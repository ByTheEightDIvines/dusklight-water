// Better Water - Dusklight native mod
//
// Stock Twilight Princess water is a flat mesh with scrolling textures, so there is nothing to
// deform. This mod keeps a coarse lattice of the game's own collision water queries around the
// camera, decides which patches are open water, and draws a real wave surface over them with the
// gfx service (see res/water.wgsl).
//
//   * Water detection: every frame a slice of a world-aligned lattice is probed with the same
//     collision water check the game uses (dBgS_WtrChk) and a ground check for water depth.
//   * Stagnant water: connected patches that are small or shallow (puddles, ponds, troughs) are
//     labelled "stagnant" and are NOT drawn over, so the stock still water stays. Water that is
//     not in the collision data at all (potion pot water is part of the pot model) is never
//     touched either.
//   * Occlusion: the surface is drawn after opaque geometry with depth writes on. The stock
//     water material tests depth (LEQUAL, no write), so it is hidden wherever the new surface is
//     in front of it.

#include "mods/service.hpp"
#include "mods/svc/camera.h"
#include "mods/svc/config.h"
#include "mods/svc/gfx.h"
#include "mods/svc/log.h"
#include "mods/svc/resource.h"
#include "mods/svc/ui.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>
#include <webgpu/webgpu.h>

// Game includes
#include "d/d_bg_s.h"
#include "d/d_bg_s_gnd_chk.h"
#include "d/d_bg_s_wtr_chk.h"
#include "d/d_com_inf_game.h"
#include "d/d_kankyo.h"

DEFINE_MOD();
IMPORT_SERVICE(LogService, svc_log);
IMPORT_SERVICE(ConfigService, svc_config);
IMPORT_SERVICE(ResourceService, svc_resource);
IMPORT_SERVICE(UiService, svc_ui);
IMPORT_SERVICE(GfxService, svc_gfx);
IMPORT_SERVICE(CameraService, svc_camera);

namespace {

// ---------------------------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------------------------
constexpr int kLatticeWidth = 64;       // cells per side, both tiers
constexpr float kFineCell = 64.0f;      // fine tier: window 4096 units (+-2048 around the camera)
constexpr float kCoarseCell = 256.0f;   // coarse tier: window 16384 units (+-8192)
constexpr int kFineBudget = 256;        // collision probes per frame (fine window: 16 frames)
constexpr int kCoarseBudget = 128;      // collision probes per frame (coarse window: 32 frames)
constexpr float kFineRadius = 1850.0f;  // the shader uses the fine tier within this distance
constexpr int kGridQuads = 256;         // surface grid quads per side
constexpr float kGridStep = 16.0f;      // grid spacing at the camera; widens toward the horizon
constexpr float kPuddleMaxDepth = 60.0f;   // patches never deeper than this are stagnant...
constexpr int kPuddleMinCells = 6;         // ...and so are patches smaller than this many cells
constexpr float kSameBodyHeightTol = 30.0f; // neighbours within this height difference are one body

ConfigVarHandle g_cvarEnabled = 0;
ConfigVarHandle g_cvarWaveHeight = 0;
ConfigVarHandle g_cvarNormals = 0;
ConfigVarHandle g_cvarClarity = 0;
ConfigVarHandle g_cvarFog = 0;
ConfigVarHandle g_cvarDebug = 0;
ConfigVarHandle g_cvarMode = 0;
ConfigVarHandle g_cvarOverlay = 0;
ConfigVarHandle g_cvarRefract = 0;

GfxDrawTypeHandle g_drawType = 0;
GfxStageHookHandle g_stageHook = 0;
GfxStageHookHandle g_hudHook = 0;
GfxDrawTypeHandle g_sunDrawType = 0;
ResourceBuffer g_shaderSource = RESOURCE_BUFFER_INIT;
GfxDeviceInfo g_deviceInfo = GFX_DEVICE_INFO_INIT;
GfxRenderTargetLayout g_layout = GFX_RENDER_TARGET_LAYOUT_INIT;
WGPURenderPipeline g_pipeline = nullptr;
WGPUBindGroupLayout g_bindLayout = nullptr;
WGPUSampler g_sampler = nullptr;
WGPUSampler g_repeatSampler = nullptr;
WGPUTexture g_waveTex = nullptr;
WGPUTextureView g_waveView = nullptr;
ResourceBuffer g_waveRaw = RESOURCE_BUFFER_INIT;
WGPURenderPipeline g_sunPipelines[2] = {nullptr, nullptr};
WGPUBindGroupLayout g_sunLayouts[2] = {nullptr, nullptr};
GfxRenderTargetLayout g_sunPipelineLayout = GFX_RENDER_TARGET_LAYOUT_INIT;

// Per-frame state handed from the after-opaque hook to the before-HUD hook (game thread).
CameraInfo g_frameCamera = CAMERA_INFO_INIT;
float g_framePlayer[3] = {0.0f, 0.0f, 0.0f};
bool g_frameValid = false;
bool g_frameAnyOpen = false;
bool g_warnedNoResolve = false;
bool g_loggedFirstDraw = false;

// ---------------------------------------------------------------------------------------------
// Water lattices (game thread only)
//
// Two world-aligned tiers are kept around the camera: a fine one for detail near the player and
// a coarse one for reach. The shader uses the fine tier close to the camera and the coarse tier
// beyond it.
// ---------------------------------------------------------------------------------------------
struct Cell {
    int ix = INT32_MIN;
    int iz = INT32_MIN;
    float height = 0.0f;
    float depth = 0.0f;
    bool found = false;
};

struct Lattice {
    float cellSize = 64.0f;
    int budget = 256;
    std::array<Cell, kLatticeWidth * kLatticeWidth> cache;
    unsigned cursor = 0;
    int ix0 = 0;
    int iz0 = 0;
};

Lattice g_fine{kFineCell, kFineBudget, {}, 0, 0, 0};
Lattice g_coarse{kCoarseCell, kCoarseBudget, {}, 0, 0, 0};
// Fine tier then coarse tier, uploaded as one storage buffer.
std::array<std::array<float, 4>, 2 * kLatticeWidth * kLatticeWidth> g_snapshot;
float g_lastWaterY = 0.0f;
bool g_haveLastWaterY = false;

int wrap_index(int v) {
    return ((v % kLatticeWidth) + kLatticeWidth) % kLatticeWidth;
}

Cell& cache_at(Lattice& lat, int ix, int iz) {
    return lat.cache[wrap_index(iz) * kLatticeWidth + wrap_index(ix)];
}

// Same query the game uses (fopAcM_wt_c::waterCheck) but with our own check object so the game's
// shared state is never disturbed.
bool probe_water(float x, float y, float z, float& outHeight) {
    static dBgS_WtrChk chk;
    cXyz pos(x, y - 500.0f, z);
    chk.Set(pos, y + 500.0f);
    if (!dComIfG_Bgsp().WaterChk(&chk)) {
        return false;
    }
    if (dComIfG_Bgsp().GetPolyAtt0(chk) == 6) {
        return false;
    }
    outHeight = chk.GetHeight();
    return true;
}

float probe_depth(float x, float waterY, float z) {
    static dBgS_GndChk gnd;
    cXyz pos(x, waterY - 2.0f, z);
    gnd.SetPos(&pos);
    const float groundY = dComIfG_Bgsp().GroundCross(&gnd);
    if (groundY == -G_CM3D_F_INF) {
        return 2000.0f; // no ground below the water: treat as very deep
    }
    return std::max(waterY - groundY, 0.0f);
}

void probe_cell(Lattice& lat, int ix, int iz, float playerY, float eyeY) {
    Cell& cell = cache_at(lat, ix, iz);
    cell.ix = ix;
    cell.iz = iz;
    cell.found = false;
    const float x = (static_cast<float>(ix) + 0.5f) * lat.cellSize;
    const float z = (static_cast<float>(iz) + 0.5f) * lat.cellSize;

    float candidates[3] = {g_lastWaterY, playerY, eyeY};
    const int count = g_haveLastWaterY ? 3 : 2;
    const float* list = g_haveLastWaterY ? candidates : candidates + 1;
    for (int i = 0; i < count; ++i) {
        float h = 0.0f;
        if (probe_water(x, list[i], z, h)) {
            cell.found = true;
            cell.height = h;
            cell.depth = probe_depth(x, h, z);
            g_lastWaterY = h;
            g_haveLastWaterY = true;
            return;
        }
    }
}

// Writes one tier into `out` (kLatticeWidth^2 entries): .x height, .y depth, .z state
// (0 none, 1 stagnant, 2 open). Returns true if any cell is open water.
bool build_snapshot(Lattice& lat, std::array<float, 4>* out) {
    constexpr int W = kLatticeWidth;
    std::array<uint8_t, W * W> state{};
    std::array<float, W * W> height{};
    std::array<float, W * W> depth{};
    for (int j = 0; j < W; ++j) {
        for (int i = 0; i < W; ++i) {
            const Cell& c = cache_at(lat, lat.ix0 + i, lat.iz0 + j);
            const int idx = j * W + i;
            if (c.ix == lat.ix0 + i && c.iz == lat.iz0 + j && c.found) {
                state[idx] = 1;
                height[idx] = c.height;
                depth[idx] = c.depth;
            }
        }
    }

    // Label connected bodies; small or shallow bodies stay stagnant.
    std::array<uint8_t, W * W> visited{};
    std::vector<int> stack;
    std::vector<int> members;
    bool anyOpen = false;
    for (int start = 0; start < W * W; ++start) {
        if (state[start] == 0 || visited[start]) {
            continue;
        }
        stack.clear();
        members.clear();
        stack.push_back(start);
        visited[start] = 1;
        bool touchesEdge = false;
        float maxDepth = 0.0f;
        while (!stack.empty()) {
            const int cur = stack.back();
            stack.pop_back();
            members.push_back(cur);
            const int ci = cur % W;
            const int cj = cur / W;
            maxDepth = std::max(maxDepth, depth[cur]);
            if (ci == 0 || cj == 0 || ci == W - 1 || cj == W - 1) {
                touchesEdge = true;
            }
            const int dx[4] = {1, -1, 0, 0};
            const int dz[4] = {0, 0, 1, -1};
            for (int k = 0; k < 4; ++k) {
                const int ni = ci + dx[k];
                const int nj = cj + dz[k];
                if (ni < 0 || nj < 0 || ni >= W || nj >= W) {
                    continue;
                }
                const int nidx = nj * W + ni;
                if (state[nidx] == 0 || visited[nidx] ||
                    std::fabs(height[nidx] - height[cur]) > kSameBodyHeightTol)
                {
                    continue;
                }
                visited[nidx] = 1;
                stack.push_back(nidx);
            }
        }
        const bool open = touchesEdge ||
                          (maxDepth >= kPuddleMaxDepth &&
                              static_cast<int>(members.size()) >= kPuddleMinCells);
        anyOpen = anyOpen || open;
        for (int m : members) {
            state[m] = open ? 2 : 1;
        }
    }

    for (int idx = 0; idx < W * W; ++idx) {
        out[idx] = {height[idx], depth[idx], static_cast<float>(state[idx]), 0.0f};
    }
    return anyOpen;
}

// Probes a slice of the tier around the camera, then refreshes its snapshot.
bool update_lattice(Lattice& lat, std::array<float, 4>* out, const CameraInfo& camera,
    float playerY) {
    lat.ix0 = static_cast<int>(std::floor(camera.eye[0] / lat.cellSize)) - kLatticeWidth / 2;
    lat.iz0 = static_cast<int>(std::floor(camera.eye[2] / lat.cellSize)) - kLatticeWidth / 2;
    for (int n = 0; n < lat.budget; ++n) {
        const unsigned idx = lat.cursor++ % (kLatticeWidth * kLatticeWidth);
        probe_cell(lat, lat.ix0 + static_cast<int>(idx % kLatticeWidth),
            lat.iz0 + static_cast<int>(idx / kLatticeWidth), playerY, camera.eye[1]);
    }
    return build_snapshot(lat, out);
}

// ---------------------------------------------------------------------------------------------
// GPU side
// ---------------------------------------------------------------------------------------------
struct Uniforms {
    float proj_from_world[16];
    float world_from_proj[16];
    float eye[4];
    float sky[4];
    float horizon[4];
    float amb[4];
    float sun[4];
    float params[4];
    float grid[4];
    float lat[4];
    float screen[4];
    float lat2[4];
    float warp[4];
    float sun0[4];
    float sun1[4];
    float sun2[4];
};
static_assert(sizeof(Uniforms) % 16 == 0);

struct DrawPayload {
    WGPUTextureView color;
    WGPUTextureView depth;
    uint32_t uniform_offset;
    uint32_t uniform_size;
    uint32_t storage_offset;
    uint32_t storage_size;
    uint32_t vertex_count;
    uint32_t _pad;
};
static_assert(sizeof(DrawPayload) <= GFX_INLINE_DRAW_PAYLOAD_SIZE);
static_assert(std::is_trivially_copyable_v<DrawPayload>);

int64_t get_int_option(ConfigVarHandle handle, int64_t fallback) {
    int64_t value = fallback;
    if (handle == 0 || svc_config->get_int(mod_ctx, handle, &value) != MOD_OK) {
        return fallback;
    }
    return value;
}

bool get_bool_option(ConfigVarHandle handle, bool fallback) {
    bool value = fallback;
    if (handle == 0 || svc_config->get_bool(mod_ctx, handle, &value) != MOD_OK) {
        return fallback;
    }
    return value;
}

void release_pipeline() {
    if (g_pipeline != nullptr) {
        wgpuRenderPipelineRelease(g_pipeline);
        g_pipeline = nullptr;
    }
    if (g_bindLayout != nullptr) {
        wgpuBindGroupLayoutRelease(g_bindLayout);
        g_bindLayout = nullptr;
    }
    g_layout = GFX_RENDER_TARGET_LAYOUT_INIT;
}

bool build_pipeline(const GfxRenderTargetLayout& layout) {
    WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl.code = {static_cast<const char*>(g_shaderSource.data), g_shaderSource.size};
    WGPUShaderModuleDescriptor moduleDesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    moduleDesc.nextInChain = &wgsl.chain;
    moduleDesc.label = {"Better Water", WGPU_STRLEN};
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(g_deviceInfo.device, &moduleDesc);
    if (module == nullptr) {
        return false;
    }

    WGPUColorTargetState colorTargets[GFX_MAX_COLOR_ATTACHMENTS];
    // Opaque write (no blend): the surface owns its pixels and writes depth.
    const uint32_t colorTargetCount =
        gfx_init_color_target_states(&layout, colorTargets, nullptr, WGPUColorWriteMask_All);

    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = module;
    fragment.entryPoint = {"fs_main", WGPU_STRLEN};
    fragment.targetCount = colorTargetCount;
    fragment.targets = colorTargets;

    WGPUDepthStencilState depthStencil = WGPU_DEPTH_STENCIL_STATE_INIT;
    depthStencil.format = layout.depth_stencil_format;
    depthStencil.depthWriteEnabled = WGPUOptionalBool_True;
    depthStencil.depthCompare = g_deviceInfo.uses_reversed_z ? WGPUCompareFunction_GreaterEqual :
                                                               WGPUCompareFunction_LessEqual;

    WGPURenderPipelineDescriptor desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    desc.label = {"Better Water", WGPU_STRLEN};
    desc.vertex.module = module;
    desc.vertex.entryPoint = {"vs_main", WGPU_STRLEN};
    desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    desc.primitive.cullMode = WGPUCullMode_None;
    desc.depthStencil = &depthStencil;
    desc.multisample.count = layout.sample_count;
    desc.fragment = &fragment;
    g_pipeline = wgpuDeviceCreateRenderPipeline(g_deviceInfo.device, &desc);
    wgpuShaderModuleRelease(module);
    if (g_pipeline == nullptr) {
        return false;
    }
    g_bindLayout = wgpuRenderPipelineGetBindGroupLayout(g_pipeline, 0);
    if (g_bindLayout == nullptr) {
        release_pipeline();
        return false;
    }
    g_layout = layout;
    return true;
}

bool ensure_pipeline(const GfxRenderTargetLayout& layout) {
    if (g_pipeline != nullptr && g_layout.key == layout.key) {
        return true;
    }
    release_pipeline();
    if (!build_pipeline(layout)) {
        release_pipeline();
        svc_log->error(mod_ctx, "failed to build water pipeline (shader error?)");
        return false;
    }
    return true;
}

// Render worker thread.
void on_draw(
    ModContext*, const GfxDrawContext* ctx, const void* payload, size_t payloadSize, void*) {
    if (payloadSize != sizeof(DrawPayload) || !ensure_pipeline(ctx->layout)) {
        return;
    }
    DrawPayload data;
    std::memcpy(&data, payload, sizeof(data));
    if (data.color == nullptr || data.depth == nullptr || g_sampler == nullptr) {
        return;
    }

    WGPUBindGroupEntry entries[5] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT,
        WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
    entries[0].binding = 0;
    entries[0].buffer = ctx->uniform_buffer;
    entries[0].offset = data.uniform_offset;
    entries[0].size = data.uniform_size;
    entries[1].binding = 1;
    entries[1].buffer = ctx->storage_buffer;
    entries[1].offset = data.storage_offset;
    entries[1].size = data.storage_size;
    entries[2].binding = 2;
    entries[2].textureView = data.color;
    entries[3].binding = 3;
    entries[3].textureView = data.depth;
    entries[4].binding = 4;
    entries[4].sampler = g_sampler;
    WGPUBindGroupDescriptor bindDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bindDesc.layout = g_bindLayout;
    bindDesc.entryCount = 5;
    bindDesc.entries = entries;
    WGPUBindGroup group = wgpuDeviceCreateBindGroup(ctx->device, &bindDesc);
    if (group == nullptr) {
        return;
    }
    wgpuRenderPassEncoderSetPipeline(ctx->pass, g_pipeline);
    wgpuRenderPassEncoderSetBindGroup(ctx->pass, 0, group, 0, nullptr);
    wgpuRenderPassEncoderDraw(ctx->pass, data.vertex_count, 1, 0, 0);
    wgpuBindGroupRelease(group);
}

struct SunPayload {
    WGPUTextureView depth;
    WGPUTextureView color;
    uint32_t uniform_offset;
    uint32_t uniform_size;
    uint32_t storage_offset;
    uint32_t storage_size;
    uint32_t vertex_count;
    uint32_t mode; // 0 = refraction layer, 1 = foam overlay
};
static_assert(sizeof(SunPayload) <= GFX_INLINE_DRAW_PAYLOAD_SIZE);
static_assert(std::is_trivially_copyable_v<SunPayload>);

constexpr uint32_t kSunCols = 25;
constexpr uint32_t kSunRows = 26;

void release_sun_pipeline() {
    for (int i = 0; i < 2; ++i) {
        if (g_sunPipelines[i] != nullptr) {
            wgpuRenderPipelineRelease(g_sunPipelines[i]);
            g_sunPipelines[i] = nullptr;
        }
        if (g_sunLayouts[i] != nullptr) {
            wgpuBindGroupLayoutRelease(g_sunLayouts[i]);
            g_sunLayouts[i] = nullptr;
        }
    }
    g_sunPipelineLayout = GFX_RENDER_TARGET_LAYOUT_INIT;
}

bool build_sun_pipeline(const GfxRenderTargetLayout& layout, uint32_t mode) {
    WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl.code = {static_cast<const char*>(g_shaderSource.data), g_shaderSource.size};
    WGPUShaderModuleDescriptor moduleDesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    moduleDesc.nextInChain = &wgsl.chain;
    moduleDesc.label = {"Better Water (Sunshine)", WGPU_STRLEN};
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(g_deviceInfo.device, &moduleDesc);
    if (module == nullptr) {
        return false;
    }

    // GXSetBlendMode(BLEND, SRCALPHA, SRCCLR): with a white source colour this is additive.
    WGPUBlendState blend{
        .color = {.operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_SrcAlpha,
            .dstFactor = mode == 0 ? WGPUBlendFactor_OneMinusSrcAlpha : WGPUBlendFactor_One},
        .alpha = {.operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_Zero,
            .dstFactor = WGPUBlendFactor_One},
    };
    WGPUColorTargetState colorTargets[GFX_MAX_COLOR_ATTACHMENTS];
    const uint32_t colorTargetCount =
        gfx_init_color_target_states(&layout, colorTargets, &blend, WGPUColorWriteMask_All);

    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = module;
    fragment.entryPoint = {mode == 0 ? "fs_sea" : "fs_sun", WGPU_STRLEN};
    fragment.targetCount = colorTargetCount;
    fragment.targets = colorTargets;

    // Depth is tested manually in the shader against the scene snapshot; no depth write
    // (GXSetZMode(TRUE, LEQUAL, FALSE)).
    WGPUDepthStencilState depthStencil = WGPU_DEPTH_STENCIL_STATE_INIT;
    depthStencil.format = layout.depth_stencil_format;
    depthStencil.depthWriteEnabled = WGPUOptionalBool_False;
    depthStencil.depthCompare = WGPUCompareFunction_Always;

    WGPURenderPipelineDescriptor desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    desc.label = {"Better Water (Sunshine)", WGPU_STRLEN};
    desc.vertex.module = module;
    desc.vertex.entryPoint = {"vs_sun", WGPU_STRLEN};
    desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    desc.primitive.cullMode = WGPUCullMode_None;
    desc.depthStencil = &depthStencil;
    desc.multisample.count = layout.sample_count;
    desc.fragment = &fragment;
    g_sunPipelines[mode] = wgpuDeviceCreateRenderPipeline(g_deviceInfo.device, &desc);
    wgpuShaderModuleRelease(module);
    if (g_sunPipelines[mode] == nullptr) {
        return false;
    }
    g_sunLayouts[mode] = wgpuRenderPipelineGetBindGroupLayout(g_sunPipelines[mode], 0);
    return g_sunLayouts[mode] != nullptr;
}

bool ensure_sun_pipeline(const GfxRenderTargetLayout& layout) {
    if (g_sunPipelines[0] != nullptr && g_sunPipelines[1] != nullptr &&
        g_sunPipelineLayout.key == layout.key)
    {
        return true;
    }
    release_sun_pipeline();
    if (!build_sun_pipeline(layout, 0) || !build_sun_pipeline(layout, 1)) {
        release_sun_pipeline();
        svc_log->error(mod_ctx, "failed to build Sunshine wave pipeline (shader error?)");
        return false;
    }
    g_sunPipelineLayout = layout;
    return true;
}

// Render worker thread.
void on_sun_draw(
    ModContext*, const GfxDrawContext* ctx, const void* payload, size_t payloadSize, void*) {
    if (payloadSize != sizeof(SunPayload) || !ensure_sun_pipeline(ctx->layout)) {
        return;
    }
    SunPayload data;
    std::memcpy(&data, payload, sizeof(data));
    if (data.depth == nullptr || g_waveView == nullptr || g_repeatSampler == nullptr ||
        data.mode > 1 || (data.mode == 0 && data.color == nullptr))
    {
        return;
    }

    WGPUBindGroupEntry entries[6] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT,
        WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT,
        WGPU_BIND_GROUP_ENTRY_INIT};
    entries[0].binding = 0;
    entries[0].buffer = ctx->uniform_buffer;
    entries[0].offset = data.uniform_offset;
    entries[0].size = data.uniform_size;
    entries[1].binding = 1;
    entries[1].buffer = ctx->storage_buffer;
    entries[1].offset = data.storage_offset;
    entries[1].size = data.storage_size;
    entries[2].binding = 3;
    entries[2].textureView = data.depth;
    entries[3].binding = 5;
    entries[3].textureView = g_waveView;
    entries[4].binding = 6;
    entries[4].sampler = g_repeatSampler;
    uint32_t entryCount = 5;
    if (data.mode == 0) {
        entries[5].binding = 2;
        entries[5].textureView = data.color;
        entryCount = 6;
    }
    WGPUBindGroupDescriptor bindDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bindDesc.layout = g_sunLayouts[data.mode];
    bindDesc.entryCount = entryCount;
    bindDesc.entries = entries;
    WGPUBindGroup group = wgpuDeviceCreateBindGroup(ctx->device, &bindDesc);
    if (group == nullptr) {
        return;
    }
    wgpuRenderPassEncoderSetPipeline(ctx->pass, g_sunPipelines[data.mode]);
    wgpuRenderPassEncoderSetBindGroup(ctx->pass, 0, group, 0, nullptr);
    wgpuRenderPassEncoderDraw(ctx->pass, data.vertex_count, 1, 0, 0);
    wgpuBindGroupRelease(group);
}

// wave.bti from Super Mario Sunshine: 128x256 I4 intensity, decoded to R8 with a box-filtered
// mip chain.
bool create_wave_texture() {
    constexpr uint32_t kW = 128;
    constexpr uint32_t kH = 256;
    if (g_waveRaw.data == nullptr || g_waveRaw.size < kW * kH) {
        return false;
    }
    uint32_t levels = 1;
    for (uint32_t m = std::max(kW, kH); m > 1; m >>= 1) {
        ++levels;
    }
    WGPUTextureDescriptor texDesc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    texDesc.label = {"Better Water wave texture", WGPU_STRLEN};
    texDesc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    texDesc.dimension = WGPUTextureDimension_2D;
    texDesc.size = {kW, kH, 1};
    texDesc.format = WGPUTextureFormat_R8Unorm;
    texDesc.mipLevelCount = levels;
    texDesc.sampleCount = 1;
    g_waveTex = wgpuDeviceCreateTexture(g_deviceInfo.device, &texDesc);
    if (g_waveTex == nullptr) {
        return false;
    }

    std::vector<uint8_t> level(static_cast<const uint8_t*>(g_waveRaw.data),
        static_cast<const uint8_t*>(g_waveRaw.data) + kW * kH);
    uint32_t w = kW;
    uint32_t h = kH;
    for (uint32_t mip = 0; mip < levels; ++mip) {
        WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        dst.texture = g_waveTex;
        dst.mipLevel = mip;
        dst.aspect = WGPUTextureAspect_All;
        WGPUTexelCopyBufferLayout layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
        layout.offset = 0;
        layout.bytesPerRow = w;
        layout.rowsPerImage = h;
        WGPUExtent3D size = {w, h, 1};
        wgpuQueueWriteTexture(g_deviceInfo.queue, &dst, level.data(), level.size(), &layout, &size);

        const uint32_t nw = std::max(w / 2, 1u);
        const uint32_t nh = std::max(h / 2, 1u);
        std::vector<uint8_t> next(static_cast<size_t>(nw) * nh);
        for (uint32_t y = 0; y < nh; ++y) {
            for (uint32_t x = 0; x < nw; ++x) {
                const uint32_t x1 = std::min(x * 2 + 1, w - 1);
                const uint32_t y1 = std::min(y * 2 + 1, h - 1);
                const uint32_t sum = level[y * 2 * w + x * 2] + level[y * 2 * w + x1] +
                                     level[y1 * w + x * 2] + level[y1 * w + x1];
                next[y * nw + x] = static_cast<uint8_t>((sum + 2) / 4);
            }
        }
        level = std::move(next);
        w = nw;
        h = nh;
    }
    g_waveView = wgpuTextureCreateView(g_waveTex, nullptr);
    return g_waveView != nullptr;
}

void set4(float (&dst)[4], float a, float b, float c, float d) {
    dst[0] = a;
    dst[1] = b;
    dst[2] = c;
    dst[3] = d;
}

void set_color(float (&dst)[4], const GXColorS10& c) {
    set4(dst, std::clamp(c.r / 255.0f, 0.0f, 1.0f), std::clamp(c.g / 255.0f, 0.0f, 1.0f),
        std::clamp(c.b / 255.0f, 0.0f, 1.0f), 1.0f);
}

float elapsed_seconds() {
    static const auto kStart = std::chrono::steady_clock::now();
    return std::chrono::duration<float>(std::chrono::steady_clock::now() - kStart).count();
}

void fill_uniforms(Uniforms& uni, const CameraInfo& camera, uint32_t width, uint32_t height) {
    const float time = elapsed_seconds();
    std::memcpy(uni.proj_from_world, camera.proj_from_world, sizeof(uni.proj_from_world));
    std::memcpy(uni.world_from_proj, camera.world_from_proj, sizeof(uni.world_from_proj));
    set4(uni.eye, camera.eye[0], camera.eye[1], camera.eye[2], 1.0f);

    dScnKy_env_light_c& env = g_env_light;
    set_color(uni.sky, env.vrbox_sky_col);
    set_color(uni.horizon, env.fog_col);
    set_color(uni.amb, env.bg_amb_col[0]);
    // Direction to the sun; fades out as it nears/falls below the horizon.
    float sx = env.sun_pos.x - camera.eye[0];
    float sy = env.sun_pos.y - camera.eye[1];
    float sz = env.sun_pos.z - camera.eye[2];
    const float sl = std::sqrt(sx * sx + sy * sy + sz * sz);
    if (sl > 1.0f) {
        sx /= sl;
        sy /= sl;
        sz /= sl;
    } else {
        sx = 0.0f;
        sy = 1.0f;
        sz = 0.0f;
    }
    set4(uni.sun, sx, sy, sz, std::clamp(sy * 6.0f, 0.0f, 1.0f));

    const float waveHeight =
        static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarWaveHeight, 100), 0, 300)) /
        100.0f;
    const float normals =
        static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarNormals, 100), 0, 400)) /
        100.0f;
    const float clarity =
        static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarClarity, 100), 10, 400)) /
        100.0f;
    const float fog =
        static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarFog, 100), 0, 1000)) /
        100.0f * 1.5e-5f;
    set4(uni.params, time, waveHeight, normals * 3.0f, fog);
    set4(uni.grid, 0.0f, 0.0f, kGridStep, static_cast<float>(kGridQuads));
    set4(uni.lat, static_cast<float>(g_fine.ix0), static_cast<float>(g_fine.iz0),
        static_cast<float>(kLatticeWidth), kFineCell);
    set4(uni.lat2, static_cast<float>(g_coarse.ix0), static_cast<float>(g_coarse.iz0),
        static_cast<float>(kLatticeWidth), kCoarseCell);
    // Radial grid warp: offset = a*|g| + b*g^2 so the outermost vertex reaches the coarse extent.
    const float half = static_cast<float>(kGridQuads) * 0.5f;
    const float reach = static_cast<float>(kLatticeWidth) * kCoarseCell * 0.5f;
    const float warpB = (reach - kGridStep * half) / (half * half);
    set4(uni.warp, kGridStep, warpB, kFineRadius, g_deviceInfo.uses_reversed_z ? 1.0f : 0.0f);
    set4(uni.screen, static_cast<float>(width), static_cast<float>(height),
        static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarDebug, 0), 0, 3)), clarity);

    // Sunshine overlay (TMapObjWave). The game runs at 30 frames per second and advances each
    // phase / scroll once per frame; the starting values are arbitrary (Sunshine randomizes them).
    constexpr double kTwoPi = 6.28318;
    const double frames = static_cast<double>(time) * 30.0;
    const double phase1 = std::fmod(1.3 + 0.02 * frames, kTwoPi);
    const double phase2 = std::fmod(4.1 + 0.03 * frames, kTwoPi);
    const double scroll0 = std::fmod(0.30 + 0.0015 * frames, 1.0);
    const double scroll1 = std::fmod(0.65 + 0.0015 * frames, 1.0);
    set4(uni.sun0, g_framePlayer[0], g_framePlayer[2], static_cast<float>(phase1),
        static_cast<float>(phase2));
    // Default amplitudes from TMapObjWave::load (unk2C = 30, unk30 = 25).
    set4(uni.sun1, 30.0f, 25.0f, static_cast<float>(scroll0), static_cast<float>(scroll1));
    const float overlay =
        static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarOverlay, 100), 0, 400)) /
        100.0f;
    const float refract =
        static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarRefract, 100), 0, 400)) /
        100.0f;
    set4(uni.sun2, refract, 1.0f, 1.0f, overlay);
}

// Game thread, after opaque scene draws and before translucent overlays (including stock water).
// Keeps the water lattices up to date; in "Custom surface" mode it also draws the surface.
void on_scene_after_opaque(ModContext*, const GfxStageContext* stageCtx, void*) {
    g_frameValid = false;
    if (!get_bool_option(g_cvarEnabled, true)) {
        return;
    }
    if (stageCtx == nullptr || stageCtx->struct_size < sizeof(GfxStageContext) ||
        stageCtx->game_view == nullptr || g_drawType == 0)
    {
        return;
    }

    CameraInfo camera = CAMERA_INFO_INIT;
    if (svc_camera->get_camera(mod_ctx, stageCtx->game_view, &camera) != MOD_OK) {
        return;
    }

    float playerY = camera.eye[1];
    g_framePlayer[0] = camera.eye[0];
    g_framePlayer[1] = camera.eye[1];
    g_framePlayer[2] = camera.eye[2];
    if (fopAc_ac_c* player = dComIfGp_getPlayer(0)) {
        playerY = player->current.pos.y;
        g_framePlayer[0] = player->current.pos.x;
        g_framePlayer[1] = player->current.pos.y;
        g_framePlayer[2] = player->current.pos.z;
    }
    const bool fineOpen = update_lattice(g_fine, g_snapshot.data(), camera, playerY);
    const bool coarseOpen = update_lattice(
        g_coarse, g_snapshot.data() + kLatticeWidth * kLatticeWidth, camera, playerY);
    // Nothing open nearby: skip the draw entirely.
    if (!fineOpen && !coarseOpen) {
        return;
    }

    g_frameCamera = camera;
    g_frameValid = true;

    if (get_int_option(g_cvarMode, 0) != 1) {
        return; // Sunshine mode draws from the before-HUD hook, after the stock water.
    }

    GfxResolveDesc resolveDesc = GFX_RESOLVE_DESC_INIT;
    resolveDesc.color = true;
    resolveDesc.depth = true;
    GfxResolvedTargets resolved = GFX_RESOLVED_TARGETS_INIT;
    if (svc_gfx->resolve_pass(mod_ctx, &resolveDesc, &resolved) != MOD_OK ||
        resolved.depth == nullptr || resolved.color == nullptr)
    {
        if (!g_warnedNoResolve) {
            g_warnedNoResolve = true;
            svc_log->warn(mod_ctx, "scene snapshots unavailable yet; water waits");
        }
        return;
    }

    Uniforms uni{};
    fill_uniforms(uni, camera, resolved.width, resolved.height);

    GfxRange uniformRange{0, 0};
    GfxRange storageRange{0, 0};
    if (svc_gfx->push_uniform(mod_ctx, &uni, sizeof(uni), &uniformRange) != MOD_OK ||
        svc_gfx->push_storage(mod_ctx, g_snapshot.data(), sizeof(g_snapshot), &storageRange) !=
            MOD_OK)
    {
        return;
    }

    DrawPayload payload{};
    payload.color = resolved.color;
    payload.depth = resolved.depth;
    payload.uniform_offset = uniformRange.offset;
    payload.uniform_size = uniformRange.size;
    payload.storage_offset = storageRange.offset;
    payload.storage_size = storageRange.size;
    payload.vertex_count = static_cast<uint32_t>(kGridQuads) * kGridQuads * 6u;
    if (svc_gfx->push_draw(mod_ctx, g_drawType, &payload, sizeof(payload)) == MOD_OK &&
        !g_loggedFirstDraw)
    {
        g_loggedFirstDraw = true;
        svc_log->info(mod_ctx, "first water surface queued");
    }
}

// Game thread, after the whole 3D scene (stock water included) and before the HUD: the Sunshine
// wave overlay is drawn here, on top of the stock water, like TMapObjWave is in Sunshine.
void on_frame_before_hud(ModContext*, const GfxStageContext*, void*) {
    if (!g_frameValid) {
        return;
    }
    g_frameValid = false;
    if (get_int_option(g_cvarMode, 0) == 1 || g_sunDrawType == 0) {
        return;
    }

    GfxResolveDesc resolveDesc = GFX_RESOLVE_DESC_INIT;
    resolveDesc.color = true;
    resolveDesc.depth = true;
    GfxResolvedTargets resolved = GFX_RESOLVED_TARGETS_INIT;
    if (svc_gfx->resolve_pass(mod_ctx, &resolveDesc, &resolved) != MOD_OK ||
        resolved.depth == nullptr || resolved.color == nullptr)
    {
        if (!g_warnedNoResolve) {
            g_warnedNoResolve = true;
            svc_log->warn(mod_ctx, "scene snapshots unavailable; wave overlay waits");
        }
        return;
    }

    Uniforms uni{};
    fill_uniforms(uni, g_frameCamera, resolved.width, resolved.height);
    GfxRange uniformRange{0, 0};
    GfxRange storageRange{0, 0};
    if (svc_gfx->push_uniform(mod_ctx, &uni, sizeof(uni), &uniformRange) != MOD_OK ||
        svc_gfx->push_storage(mod_ctx, g_snapshot.data(), sizeof(g_snapshot), &storageRange) !=
            MOD_OK)
    {
        return;
    }

    SunPayload payload{};
    payload.depth = resolved.depth;
    payload.color = resolved.color;
    payload.uniform_offset = uniformRange.offset;
    payload.uniform_size = uniformRange.size;
    payload.storage_offset = storageRange.offset;
    payload.storage_size = storageRange.size;
    payload.vertex_count = kSunCols * kSunRows * 6u;
    // Refraction wobble first (Sunshine's seaindirect layer), foam on top.
    payload.mode = 0;
    if (get_int_option(g_cvarRefract, 100) > 0) {
        svc_gfx->push_draw(mod_ctx, g_sunDrawType, &payload, sizeof(payload));
    }
    payload.mode = 1;
    if (svc_gfx->push_draw(mod_ctx, g_sunDrawType, &payload, sizeof(payload)) == MOD_OK &&
        !g_loggedFirstDraw)
    {
        g_loggedFirstDraw = true;
        svc_log->info(mod_ctx, "first Sunshine wave overlay queued");
    }
}

// ---------------------------------------------------------------------------------------------
// Config / UI
// ---------------------------------------------------------------------------------------------
ModResult register_bool_option(
    const char* name, bool defaultValue, ConfigVarHandle& outHandle, ModError* error) {
    ConfigVarDesc desc = CONFIG_VAR_DESC_INIT;
    desc.name = name;
    desc.type = CONFIG_VAR_BOOL;
    desc.default_bool = defaultValue;
    if (svc_config->register_var(mod_ctx, &desc, &outHandle) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to register option");
    }
    return MOD_OK;
}

ModResult register_int_option(
    const char* name, int64_t defaultValue, ConfigVarHandle& outHandle, ModError* error) {
    ConfigVarDesc desc = CONFIG_VAR_DESC_INIT;
    desc.name = name;
    desc.type = CONFIG_VAR_INT;
    desc.default_int = defaultValue;
    if (svc_config->register_var(mod_ctx, &desc, &outHandle) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to register option");
    }
    return MOD_OK;
}

void add_number(UiElementHandle pane, const char* label, const char* help, ConfigVarHandle cvar,
    double min, double max, double step, const char* suffix) {
    UiControlDesc control = UI_CONTROL_DESC_INIT;
    control.kind = UI_CONTROL_NUMBER;
    control.label = label;
    control.help_rml = help;
    control.binding = UI_BINDING_CONFIG_VAR;
    control.config_var = cvar;
    control.min = min;
    control.max = max;
    control.step = step;
    control.suffix = suffix;
    svc_ui->pane_add_control(mod_ctx, pane, &control, nullptr);
}

ModResult build_panel(ModContext*, UiElementHandle panel, void*, ModError*) {
    UiControlDesc control = UI_CONTROL_DESC_INIT;
    control.kind = UI_CONTROL_TOGGLE;
    control.label = "Enabled";
    control.help_rml = "Replaces open water with the animated wave surface.";
    control.binding = UI_BINDING_CONFIG_VAR;
    control.config_var = g_cvarEnabled;
    svc_ui->pane_add_control(mod_ctx, panel, &control, nullptr);

    static const char* kModeOptions[] = {"Sunshine waves", "Custom surface"};
    control = UI_CONTROL_DESC_INIT;
    control.kind = UI_CONTROL_SELECT;
    control.label = "Style";
    control.help_rml = "Sunshine waves: the wave grid from Super Mario Sunshine drawn over the "
                       "game's own water.<br/>Custom surface: an experimental replacement water "
                       "surface with refraction and reflections.";
    control.binding = UI_BINDING_CONFIG_VAR;
    control.config_var = g_cvarMode;
    control.options = kModeOptions;
    control.option_count = 2;
    svc_ui->pane_add_control(mod_ctx, panel, &control, nullptr);

    add_number(panel, "Foam Intensity", "Brightness of the Sunshine wave foam.", g_cvarOverlay, 0,
        400, 10, "%");
    add_number(panel, "Refraction", "Sunshine-style wobble of the view through the water.",
        g_cvarRefract, 0, 400, 10, "%");
    add_number(panel, "Wave Height", "Size of the swell on open, deep water.", g_cvarWaveHeight, 0,
        300, 10, "%");
    add_number(panel, "Ripple Strength", "How strongly the surface ripples shade and reflect.",
        g_cvarNormals, 0, 400, 10, "%");
    add_number(panel, "Water Clarity", "Higher is clearer; lower tints and darkens sooner.",
        g_cvarClarity, 10, 400, 10, "%");
    add_number(panel, "Distance Haze", "How quickly distant water fades to the fog colour.",
        g_cvarFog, 0, 1000, 10, "%");

    static const char* kDebugOptions[] = {"Off", "Water mask", "Thickness", "Normals"};
    control = UI_CONTROL_DESC_INIT;
    control.kind = UI_CONTROL_SELECT;
    control.label = "Debug View";
    control.help_rml = "Water mask: green where the mod draws water.<br/>Thickness: water "
                       "column above the bed.<br/>Normals: surface normals.";
    control.binding = UI_BINDING_CONFIG_VAR;
    control.config_var = g_cvarDebug;
    control.options = kDebugOptions;
    control.option_count = 4;
    svc_ui->pane_add_control(mod_ctx, panel, &control, nullptr);
    return MOD_OK;
}

}  // namespace

extern "C" {

MOD_EXPORT ModResult mod_initialize(ModError* error) {
    ModResult result = svc_resource->load(mod_ctx, "water.wgsl", &g_shaderSource);
    if (result != MOD_OK) {
        return mods::set_error(error, result, "failed to load water shader");
    }
    result = svc_resource->load(mod_ctx, "wave.raw", &g_waveRaw);
    if (result != MOD_OK) {
        return mods::set_error(error, result, "failed to load wave texture");
    }

    if (register_bool_option("waterEnabled", true, g_cvarEnabled, error) != MOD_OK ||
        register_int_option("waveHeight", 100, g_cvarWaveHeight, error) != MOD_OK ||
        register_int_option("rippleStrength", 100, g_cvarNormals, error) != MOD_OK ||
        register_int_option("clarity", 100, g_cvarClarity, error) != MOD_OK ||
        register_int_option("distanceHaze", 100, g_cvarFog, error) != MOD_OK ||
        register_int_option("debugView", 0, g_cvarDebug, error) != MOD_OK ||
        register_int_option("style", 0, g_cvarMode, error) != MOD_OK ||
        register_int_option("foamIntensity", 100, g_cvarOverlay, error) != MOD_OK ||
        register_int_option("refraction", 100, g_cvarRefract, error) != MOD_OK)
    {
        return MOD_ERROR;
    }

    if (svc_gfx->get_device_info(mod_ctx, &g_deviceInfo) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to query device info");
    }

    WGPUSamplerDescriptor samplerDesc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    samplerDesc.addressModeU = WGPUAddressMode_ClampToEdge;
    samplerDesc.addressModeV = WGPUAddressMode_ClampToEdge;
    samplerDesc.addressModeW = WGPUAddressMode_ClampToEdge;
    samplerDesc.magFilter = WGPUFilterMode_Linear;
    samplerDesc.minFilter = WGPUFilterMode_Linear;
    g_sampler = wgpuDeviceCreateSampler(g_deviceInfo.device, &samplerDesc);
    if (g_sampler == nullptr) {
        return mods::set_error(error, MOD_ERROR, "failed to create sampler");
    }
    WGPUSamplerDescriptor repeatDesc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    repeatDesc.addressModeU = WGPUAddressMode_Repeat;
    repeatDesc.addressModeV = WGPUAddressMode_Repeat;
    repeatDesc.addressModeW = WGPUAddressMode_Repeat;
    repeatDesc.magFilter = WGPUFilterMode_Linear;
    repeatDesc.minFilter = WGPUFilterMode_Linear;
    repeatDesc.mipmapFilter = WGPUMipmapFilterMode_Linear;
    g_repeatSampler = wgpuDeviceCreateSampler(g_deviceInfo.device, &repeatDesc);
    if (g_repeatSampler == nullptr || !create_wave_texture()) {
        return mods::set_error(error, MOD_ERROR, "failed to create wave texture");
    }

    GfxDrawTypeDesc drawDesc = GFX_DRAW_TYPE_DESC_INIT;
    drawDesc.label = "Better Water";
    drawDesc.draw = on_draw;
    if (svc_gfx->register_draw_type(mod_ctx, &drawDesc, &g_drawType) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to register draw type");
    }
    GfxDrawTypeDesc sunDrawDesc = GFX_DRAW_TYPE_DESC_INIT;
    sunDrawDesc.label = "Better Water (Sunshine waves)";
    sunDrawDesc.draw = on_sun_draw;
    if (svc_gfx->register_draw_type(mod_ctx, &sunDrawDesc, &g_sunDrawType) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to register Sunshine draw type");
    }
    GfxStageHookDesc hudDesc = GFX_STAGE_HOOK_DESC_INIT;
    hudDesc.callback = on_frame_before_hud;
    if (svc_gfx->register_stage_hook(
            mod_ctx, GFX_STAGE_FRAME_BEFORE_HUD, &hudDesc, &g_hudHook) != MOD_OK)
    {
        return mods::set_error(error, MOD_ERROR, "failed to register before-HUD hook");
    }
    GfxStageHookDesc stageDesc = GFX_STAGE_HOOK_DESC_INIT;
    stageDesc.callback = on_scene_after_opaque;
    if (svc_gfx->register_stage_hook(
            mod_ctx, GFX_STAGE_SCENE_AFTER_OPAQUE, &stageDesc, &g_stageHook) != MOD_OK)
    {
        return mods::set_error(error, MOD_ERROR, "failed to register stage hook");
    }

    UiModsPanelDesc panelDesc = UI_MODS_PANEL_DESC_INIT;
    panelDesc.build = build_panel;
    svc_ui->register_mods_panel(mod_ctx, &panelDesc);

    svc_log->info(mod_ctx, "Better Water ready");
    return MOD_OK;
}

MOD_EXPORT ModResult mod_update(ModError*) {
    return MOD_OK;
}

MOD_EXPORT ModResult mod_shutdown(ModError*) {
    svc_resource->free(mod_ctx, &g_shaderSource);
    svc_resource->free(mod_ctx, &g_waveRaw);
    release_pipeline();
    release_sun_pipeline();
    if (g_waveView != nullptr) {
        wgpuTextureViewRelease(g_waveView);
        g_waveView = nullptr;
    }
    if (g_waveTex != nullptr) {
        wgpuTextureRelease(g_waveTex);
        g_waveTex = nullptr;
    }
    if (g_repeatSampler != nullptr) {
        wgpuSamplerRelease(g_repeatSampler);
        g_repeatSampler = nullptr;
    }
    if (g_sampler != nullptr) {
        wgpuSamplerRelease(g_sampler);
        g_sampler = nullptr;
    }
    g_cvarEnabled = g_cvarWaveHeight = g_cvarNormals = g_cvarClarity = g_cvarFog = g_cvarDebug = 0;
    g_cvarMode = g_cvarOverlay = g_cvarRefract = 0;
    g_drawType = g_sunDrawType = 0;
    g_stageHook = g_hudHook = 0;
    return MOD_OK;
}
}
