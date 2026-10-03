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
constexpr int kLatticeWidth = 64;       // lattice cells per side
constexpr float kCellSize = 64.0f;      // world units per lattice cell (window = 4096 units)
constexpr int kGridQuads = 256;         // surface grid quads per side (step = window / quads)
constexpr int kProbeBudget = 256;       // collision probes per frame (window refreshed in 16 frames)
constexpr float kPuddleMaxDepth = 60.0f;   // patches never deeper than this are stagnant...
constexpr int kPuddleMinCells = 6;         // ...and so are patches smaller than this many cells
constexpr float kSameBodyHeightTol = 30.0f; // neighbours within this height difference are one body

ConfigVarHandle g_cvarEnabled = 0;
ConfigVarHandle g_cvarWaveHeight = 0;
ConfigVarHandle g_cvarNormals = 0;
ConfigVarHandle g_cvarClarity = 0;
ConfigVarHandle g_cvarFog = 0;
ConfigVarHandle g_cvarDebug = 0;

GfxDrawTypeHandle g_drawType = 0;
GfxStageHookHandle g_stageHook = 0;
ResourceBuffer g_shaderSource = RESOURCE_BUFFER_INIT;
GfxDeviceInfo g_deviceInfo = GFX_DEVICE_INFO_INIT;
GfxRenderTargetLayout g_layout = GFX_RENDER_TARGET_LAYOUT_INIT;
WGPURenderPipeline g_pipeline = nullptr;
WGPUBindGroupLayout g_bindLayout = nullptr;
WGPUSampler g_sampler = nullptr;
bool g_warnedNoResolve = false;
bool g_loggedFirstDraw = false;

// ---------------------------------------------------------------------------------------------
// Water lattice (game thread only)
// ---------------------------------------------------------------------------------------------
struct Cell {
    int ix = INT32_MIN;
    int iz = INT32_MIN;
    float height = 0.0f;
    float depth = 0.0f;
    bool found = false;
};

std::array<Cell, kLatticeWidth * kLatticeWidth> g_cache;
std::array<std::array<float, 4>, kLatticeWidth * kLatticeWidth> g_snapshot;
unsigned g_cursor = 0;
float g_lastWaterY = 0.0f;
bool g_haveLastWaterY = false;

int wrap_index(int v) {
    return ((v % kLatticeWidth) + kLatticeWidth) % kLatticeWidth;
}

Cell& cache_at(int ix, int iz) {
    return g_cache[wrap_index(iz) * kLatticeWidth + wrap_index(ix)];
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

void probe_cell(int ix, int iz, float playerY, float eyeY) {
    Cell& cell = cache_at(ix, iz);
    cell.ix = ix;
    cell.iz = iz;
    cell.found = false;
    const float x = (static_cast<float>(ix) + 0.5f) * kCellSize;
    const float z = (static_cast<float>(iz) + 0.5f) * kCellSize;

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

// Builds the snapshot sent to the GPU: .x height, .y depth, .z state (0 none, 1 stagnant, 2 open).
void build_snapshot(int ix0, int iz0) {
    constexpr int W = kLatticeWidth;
    std::array<uint8_t, W * W> state{};
    std::array<float, W * W> height{};
    std::array<float, W * W> depth{};
    for (int j = 0; j < W; ++j) {
        for (int i = 0; i < W; ++i) {
            const Cell& c = cache_at(ix0 + i, iz0 + j);
            const int idx = j * W + i;
            if (c.ix == ix0 + i && c.iz == iz0 + j && c.found) {
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
        for (int m : members) {
            state[m] = open ? 2 : 1;
        }
    }

    for (int idx = 0; idx < W * W; ++idx) {
        g_snapshot[idx] = {height[idx], depth[idx], static_cast<float>(state[idx]), 0.0f};
    }
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

// Game thread, after opaque scene draws and before translucent overlays (including stock water).
void on_scene_after_opaque(ModContext*, const GfxStageContext* stageCtx, void*) {
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

    // Lattice window around the camera, refreshed a slice per frame.
    const int ix0 = static_cast<int>(std::floor(camera.eye[0] / kCellSize)) - kLatticeWidth / 2;
    const int iz0 = static_cast<int>(std::floor(camera.eye[2] / kCellSize)) - kLatticeWidth / 2;
    float playerY = camera.eye[1];
    if (fopAc_ac_c* player = dComIfGp_getPlayer(0)) {
        playerY = player->current.pos.y;
    }
    for (int n = 0; n < kProbeBudget; ++n) {
        const unsigned idx = g_cursor++ % (kLatticeWidth * kLatticeWidth);
        probe_cell(ix0 + static_cast<int>(idx % kLatticeWidth),
            iz0 + static_cast<int>(idx / kLatticeWidth), playerY, camera.eye[1]);
    }
    build_snapshot(ix0, iz0);

    // Nothing open nearby: skip the draw entirely.
    bool anyOpen = false;
    for (const auto& s : g_snapshot) {
        if (s[2] > 1.5f) {
            anyOpen = true;
            break;
        }
    }
    if (!anyOpen) {
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
            svc_log->warn(mod_ctx, "scene snapshots unavailable yet; water waits");
        }
        return;
    }

    static const auto kStart = std::chrono::steady_clock::now();
    const float time = std::chrono::duration<float>(std::chrono::steady_clock::now() - kStart).count();

    const float step = (static_cast<float>(kLatticeWidth) * kCellSize) / kGridQuads;

    Uniforms uni{};
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
    set4(uni.params, time, waveHeight, normals * 6.0f, fog);
    // Grid is snapped to the step so vertices do not swim as the camera moves.
    set4(uni.grid, std::floor(camera.eye[0] / step) * step, std::floor(camera.eye[2] / step) * step,
        step, static_cast<float>(kGridQuads));
    set4(uni.lat, static_cast<float>(ix0), static_cast<float>(iz0),
        static_cast<float>(kLatticeWidth), kCellSize);
    set4(uni.screen, static_cast<float>(resolved.width), static_cast<float>(resolved.height),
        static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarDebug, 0), 0, 3)), clarity);

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

    if (register_bool_option("enabled", true, g_cvarEnabled, error) != MOD_OK ||
        register_int_option("waveHeight", 100, g_cvarWaveHeight, error) != MOD_OK ||
        register_int_option("rippleStrength", 100, g_cvarNormals, error) != MOD_OK ||
        register_int_option("clarity", 100, g_cvarClarity, error) != MOD_OK ||
        register_int_option("distanceHaze", 100, g_cvarFog, error) != MOD_OK ||
        register_int_option("debugView", 0, g_cvarDebug, error) != MOD_OK)
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

    GfxDrawTypeDesc drawDesc = GFX_DRAW_TYPE_DESC_INIT;
    drawDesc.label = "Better Water";
    drawDesc.draw = on_draw;
    if (svc_gfx->register_draw_type(mod_ctx, &drawDesc, &g_drawType) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to register draw type");
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
    release_pipeline();
    if (g_sampler != nullptr) {
        wgpuSamplerRelease(g_sampler);
        g_sampler = nullptr;
    }
    g_cvarEnabled = g_cvarWaveHeight = g_cvarNormals = g_cvarClarity = g_cvarFog = g_cvarDebug = 0;
    g_drawType = 0;
    g_stageHook = 0;
    return MOD_OK;
}
}
