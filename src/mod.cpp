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
#include "mods/svc/hook.hpp"
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
#include "JSystem/J3DGraphAnimator/J3DModelData.h"
#include "JSystem/J3DGraphBase/J3DMaterial.h"
#include "JSystem/J3DGraphBase/J3DShape.h"
#include "JSystem/J3DGraphBase/J3DShapeDraw.h"
#include "JSystem/JUtility/JUTNameTab.h"
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <aurora/dl.hpp>

DEFINE_MOD();
IMPORT_SERVICE(LogService, svc_log);
IMPORT_SERVICE(ConfigService, svc_config);
IMPORT_SERVICE(ResourceService, svc_resource);
IMPORT_SERVICE(UiService, svc_ui);
IMPORT_SERVICE(GfxService, svc_gfx);
IMPORT_SERVICE(CameraService, svc_camera);
IMPORT_SERVICE(HookService, svc_hook);

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
ConfigVarHandle g_cvarSwim = 0;
ConfigVarHandle g_cvarWaveHeight = 0;
ConfigVarHandle g_cvarNormals = 0;
ConfigVarHandle g_cvarClarity = 0;
ConfigVarHandle g_cvarFog = 0;
ConfigVarHandle g_cvarDebug = 0;
ConfigVarHandle g_cvarMode = 0;
ConfigVarHandle g_cvarOverlay = 0;
ConfigVarHandle g_cvarRefract = 0;
ConfigVarHandle g_cvarSpray = 0;
ConfigVarHandle g_cvarColor = 0;

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
WGPURenderPipeline g_sunPipelines[3] = {nullptr, nullptr, nullptr}; // 0 refraction, 1 foam, 2 spray
WGPUBindGroupLayout g_sunLayouts[3] = {nullptr, nullptr, nullptr};
float g_spray[8][4] = {};  // waterfall spray emitters: xyz foot position, w strength
int g_sprayCount = 0;
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
float g_rippleIntensity = 0.0f; // swim ripple strength, smoothed
bool g_inOwnProbe = false;  // our lattice probes must see the raw, flat water
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
    g_inOwnProbe = true;
    const bool hit = dComIfG_Bgsp().WaterChk(&chk);
    g_inOwnProbe = false;
    if (!hit) {
        return false;
    }
    if (dComIfG_Bgsp().GetPolyAtt0(chk) == 6) {
        return false;
    }
    outHeight = chk.GetHeight();
    return true;
}

// Height of the water Link is in or standing beside (or his own height when there is none). Water
// surfaces far above this (cave ceilings, upper pools) are not part of the sea and are ignored.
float g_refY = 0.0f;
constexpr float kMaxAboveRef = 150.0f;

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
        if (probe_water(x, list[i], z, h) && h <= g_refY + kMaxAboveRef) {
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
// Wave collision
//
// Every water height the game uses (Link swimming, floating items, splashes) comes out of
// dBgS::SplGrpChk, which returns the flat height of the water triangle under a point. Wind Waker
// keeps its sea mesh and its collision on one shared height function; here the same effect comes
// from a post-hook that adds the wave height to that result on open water. The wave function is a
// CPU mirror of waves() in res/water.wgsl and must stay in sync with it.
// ---------------------------------------------------------------------------------------------
float elapsed_seconds();
int64_t get_int_option(ConfigVarHandle handle, int64_t fallback);
bool get_bool_option(ConfigVarHandle handle, bool fallback);

constexpr float kBaseAmplitude = 14.0f;
constexpr double kWaveDecay = 0.70;
constexpr float kBaseWavelength = 900.0f;
constexpr float kSwellScale = 0.72f;
constexpr int kWaveCountVertex = 5;

bool g_snapshotReady = false;

// Waterfall spray emitters: places where a fine-tier water cell has a much higher water cell right
// next to it (the pool at the foot of a fall and the ledge above it).
void update_spray(const CameraInfo& camera) {
    constexpr int W = kLatticeWidth;
    struct Cand {
        float x, y, z, strength, d2;
    };
    std::array<Cand, 64> cands;
    int n = 0;
    const std::array<float, 4>* tier = g_snapshot.data();
    for (int j = 1; j < W - 1; ++j) {
        for (int i = 1; i < W - 1; ++i) {
            const std::array<float, 4>& c = tier[j * W + i];
            if (c[2] < 0.5f) {
                continue;
            }
            static const int kDx[4] = {1, -1, 0, 0};
            static const int kDz[4] = {0, 0, 1, -1};
            for (int k = 0; k < 4; ++k) {
                const std::array<float, 4>& nb = tier[(j + kDz[k]) * W + (i + kDx[k])];
                const float step = nb[0] - c[0];
                if (nb[2] < 0.5f || step < 40.0f || step > 700.0f) {
                    continue;
                }
                const float x = (static_cast<float>(g_fine.ix0 + i) + 0.5f + 0.5f * kDx[k]) * kFineCell;
                const float z = (static_cast<float>(g_fine.iz0 + j) + 0.5f + 0.5f * kDz[k]) * kFineCell;
                const float dx = x - camera.eye[0];
                const float dz = z - camera.eye[2];
                const float d2 = dx * dx + dz * dz;
                if (d2 > 3000.0f * 3000.0f || n >= static_cast<int>(cands.size())) {
                    continue;
                }
                cands[n++] = {x, c[0] + 2.0f, z, std::clamp(step / 200.0f, 0.35f, 1.0f), d2};
            }
        }
    }
    std::sort(cands.begin(), cands.begin() + n,
        [](const Cand& a, const Cand& b) { return a.d2 < b.d2; });
    g_sprayCount = 0;
    for (int i = 0; i < n && g_sprayCount < 8; ++i) {
        bool near = false;
        for (int k = 0; k < g_sprayCount; ++k) {
            const float dx = g_spray[k][0] - cands[i].x;
            const float dz = g_spray[k][2] - cands[i].z;
            if (dx * dx + dz * dz < 180.0f * 180.0f) {
                near = true;
                break;
            }
        }
        if (near) {
            continue;
        }
        g_spray[g_sprayCount][0] = cands[i].x;
        g_spray[g_sprayCount][1] = cands[i].y;
        g_spray[g_sprayCount][2] = cands[i].z;
        g_spray[g_sprayCount][3] = cands[i].strength;
        ++g_sprayCount;
    }
}
float g_collisionEye[2] = {0.0f, 0.0f};

float smoothstep_f(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Mirror of the shader's waves(): sum of Sunshine's two swells and the sharp-crested octaves.
float cpu_wave_height(float x, float z, float t) {
    constexpr double kTwoPi = 6.28318;
    const double frames = static_cast<double>(t) * 30.0;
    const double phase1 = std::fmod(1.3 + 0.02 * frames, kTwoPi);
    const double phase2 = std::fmod(4.1 + 0.03 * frames, kTwoPi);
    const double k1 = 0.02 / 6.28318;
    const double k2 = 0.03 / 6.28318;
    double h = 30.0 * kSwellScale * std::sin(k1 * x + phase1) +
               25.0 * kSwellScale * std::sin(k2 * z + phase2);
    for (int i = 0; i < kWaveCountVertex; ++i) {
        const double angle = 0.6 + i * 2.399963;
        const double dx = std::cos(angle);
        const double dz = std::sin(angle);
        const double wl = kBaseWavelength * std::pow(0.55, i);
        const double k = 6.2831853 / wl;
        const double omega = std::sqrt(980.0 * k) * 0.6;
        const double amp = kBaseAmplitude * std::pow(kWaveDecay, i);
        const double th = k * (dx * x + dz * z) - omega * t + i * 1.7;
        h += amp * (std::exp(std::sin(th) - 1.0) - 0.466);
    }
    return static_cast<float>(h);
}

// Mirror of surface_info(): bilinear (height, depth, openness) from the uploaded snapshot.
bool cpu_surface_info(float x, float z, float& openness, float& depth) {
    const bool fine = std::max(std::fabs(x - g_collisionEye[0]), std::fabs(z - g_collisionEye[1])) <
                      kFineRadius;
    const Lattice& lat = fine ? g_fine : g_coarse;
    const std::array<float, 4>* tier = g_snapshot.data() + (fine ? 0 : kLatticeWidth * kLatticeWidth);
    const float cx = x / lat.cellSize - static_cast<float>(lat.ix0) - 0.5f;
    const float cz = z / lat.cellSize - static_cast<float>(lat.iz0) - 0.5f;
    const float bx = std::floor(cx);
    const float bz = std::floor(cz);
    const float fx = cx - bx;
    const float fz = cz - bz;
    const int ix = static_cast<int>(bx);
    const int iz = static_cast<int>(bz);
    const float w[4] = {(1 - fx) * (1 - fz), fx * (1 - fz), (1 - fx) * fz, fx * fz};
    const int di[4] = {0, 1, 0, 1};
    const int dj[4] = {0, 0, 1, 1};
    float sum = 0.0f;
    float d = 0.0f;
    float open = 0.0f;
    for (int n = 0; n < 4; ++n) {
        const int i = ix + di[n];
        const int j = iz + dj[n];
        if (i < 0 || j < 0 || i >= kLatticeWidth || j >= kLatticeWidth) {
            continue;
        }
        const std::array<float, 4>& c = tier[j * kLatticeWidth + i];
        if (c[2] > 0.5f) {
            sum += w[n];
            d += c[1] * w[n];
        }
        if (c[2] > 1.5f) {
            open += w[n];
        }
    }
    if (sum < 1e-4f) {
        return false;
    }
    openness = open;
    depth = d / sum;
    return open >= 0.01f;
}


// ---------------------------------------------------------------------------------------------
// Stage water capture
//
// dKy_bg_MAxx_proc runs every frame for every stage / actor model that carries water materials. A
// post-hook picks out the stock water layers by material name (MA06 murky base, MA09 shimmering
// top), hides those shapes, and remembers their triangles so we can draw our own water on exactly
// the same surface.
// ---------------------------------------------------------------------------------------------
DEFINE_HOOK(&dKy_bg_MAxx_proc, BgMaxxProc);

struct V3 {
    float x, y, z;
};
struct WTri {
    V3 p[3];
};

struct LocalWater {
    const void* names = nullptr;
    const void* posArray = nullptr;
    unsigned matNum = 0;
    unsigned gen = 0;
    std::vector<J3DShape*> hide; // every stock water layer shape in this model
    std::vector<WTri> tris;      // drawable layers, model space, MA06 duplicates of MA09 removed
};
std::unordered_map<const void*, LocalWater> g_localWater;
unsigned g_genCounter = 0;

struct Instance {
    const void* md;
    float mtx[12];
    float scale[3];
};
std::vector<Instance> g_instances;

bool starts_with(const char* s, const char* prefix) {
    return s != nullptr && std::strncmp(s, prefix, std::strlen(prefix)) == 0;
}

// Dungeons keep their stock water, except the Morpheel arena.
bool stage_uses_water_mesh() {
    const char* stage = dComIfGp_getStartStageName();
    if (stage == nullptr) {
        return true;
    }
    if (starts_with(stage, "D_MN01B")) {
        return true;
    }
    return !starts_with(stage, "D_");
}

bool in_morpheel_arena() {
    return starts_with(dComIfGp_getStartStageName(), "D_MN01B");
}

// Layer role from the material name: 0 = not water, 1 = hidden only (shore / wave strips),
// 2 = drawable base layer (MA06), 3 = drawable top layer (MA09).
int water_role(const char* nm) {
    if (nm == nullptr || std::strlen(nm) < 8 || nm[3] != 'M' || nm[4] != 'A' || nm[5] != '0') {
        return 0;
    }
    if (nm[6] != '6' && nm[6] != '9') {
        return 0;
    }
    static const char* const kNot[] = {"Taki", "aki", "plash", "unsui", "awa", "Oil", "Kasan",
        "Sunbeam", "wall", "Jyozan"};
    for (const char* bad : kNot) {
        if (std::strstr(nm, bad) != nullptr) {
            return 0;
        }
    }
    if (std::strstr(nm, "giwa") != nullptr || std::strstr(nm, "nami") != nullptr) {
        return 1;
    }
    return nm[6] == '9' ? 3 : 2;
}

void extract_shape_tris(J3DModelData* md, J3DShape* sh, std::vector<WTri>& out) {
    J3DVertexData* vd = &md->getVertexData();
    const u8* pos = static_cast<const u8*>(vd->getVtxPosArray());
    if (pos == nullptr || sh == nullptr) {
        return;
    }
    const int ptype = vd->getVtxPosType();
    const float frac = 1.0f / static_cast<float>(1 << vd->getVtxPosFrac());
    const unsigned groups = sh->getMtxGroupNum();
    for (unsigned g = 0; g < groups; ++g) {
        J3DShapeDraw* d = sh->getShapeDraw(g);
        if (d == nullptr || d->getDisplayList() == nullptr) {
            continue;
        }
        aurora::gx::dl::Reader rd(
            reinterpret_cast<const u8*>(d->getDisplayList()), d->getDisplayListSize(),
            sh->getVtxDesc());
        while (auto c = rd.next()) {
            if (c->kind != aurora::gx::dl::Command::Kind::Draw) {
                continue;
            }
            auto P = [&](u32 v) {
                const u32 ix = c->draw.attr_idx(v, GX_VA_POS);
                V3 o;
                if (ptype == GX_F32) {
                    const float* f = reinterpret_cast<const float*>(pos + ix * 12);
                    o = {f[0], f[1], f[2]};
                } else {
                    const s16* f = reinterpret_cast<const s16*>(pos + ix * 6);
                    o = {f[0] * frac, f[1] * frac, f[2] * frac};
                }
                return o;
            };
            aurora::gx::dl::expand_triangles(c->draw.prim, c->draw.vtxCount,
                [&](u16 a, u16 b, u16 cc) {
                    WTri t{{P(a), P(b), P(cc)}};
                    const float ar = std::fabs((t.p[1].x - t.p[0].x) * (t.p[2].z - t.p[0].z) -
                                               (t.p[1].z - t.p[0].z) * (t.p[2].x - t.p[0].x));
                    if (ar > 1e-3f) {
                        out.push_back(t);
                    }
                });
        }
    }
}

int64_t centroid_key(const WTri& t) {
    const int64_t cx = std::llround((t.p[0].x + t.p[1].x + t.p[2].x) / 3.0f);
    const int64_t cz = std::llround((t.p[0].z + t.p[1].z + t.p[2].z) / 3.0f);
    return (cx << 32) ^ (cz & 0xFFFFFFFF);
}

void scan_model(J3DModelData* md, LocalWater& lw) {
    lw.names = md->getMaterialName();
    lw.posArray = md->getVertexData().getVtxPosArray();
    lw.matNum = md->getMaterialNum();
    lw.gen = ++g_genCounter;
    lw.hide.clear();
    lw.tris.clear();
    if (!stage_uses_water_mesh()) {
        return;
    }
    JUTNameTab* names = md->getMaterialName();
    if (names == nullptr) {
        return;
    }
    std::vector<WTri> top;
    std::vector<WTri> base;
    for (u16 i = 0; i < lw.matNum; ++i) {
        const int role = water_role(names->getName(i));
        if (role == 0) {
            continue;
        }
        J3DMaterial* mat = md->getMaterialNodePointer(i);
        J3DShape* sh = mat != nullptr ? mat->getShape() : nullptr;
        if (sh == nullptr) {
            continue;
        }
        lw.hide.push_back(sh);
        if (role == 3) {
            extract_shape_tris(md, sh, top);
        } else if (role == 2) {
            extract_shape_tris(md, sh, base);
        }
    }
    std::unordered_set<int64_t> have;
    for (const WTri& t : top) {
        have.insert(centroid_key(t));
        lw.tris.push_back(t);
    }
    for (const WTri& t : base) {
        if (have.find(centroid_key(t)) == have.end()) {
            lw.tris.push_back(t);
        }
    }
    char line[200];
    std::snprintf(line, sizeof(line), "water model %p: %zu layers hidden, %zu triangles", (void*)md,
        lw.hide.size(), lw.tris.size());
    svc_log->info(mod_ctx, line);
}

void on_bg_maxx_post(ModContext*, void* args, void*, void*) {
    if (!get_bool_option(g_cvarEnabled, true)) {
        return;
    }
    J3DModel* model = mods::arg<J3DModel*>(args, 0);
    if (model == nullptr) {
        return;
    }
    J3DModelData* md = model->getModelData();
    if (md == nullptr) {
        return;
    }
    LocalWater& lw = g_localWater[md];
    if (lw.gen == 0 || lw.names != static_cast<const void*>(md->getMaterialName()) ||
        lw.matNum != md->getMaterialNum() ||
        lw.posArray != md->getVertexData().getVtxPosArray())
    {
        scan_model(md, lw);
    }
    if (lw.hide.empty()) {
        return;
    }
    for (J3DShape* sh : lw.hide) {
        sh->hide();
    }
    if (lw.tris.empty()) {
        return;
    }
    Instance inst{};
    inst.md = md;
    MtxP m = model->getBaseTRMtx();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            inst.mtx[r * 4 + c] = m[r][c];
        }
    }
    const Vec* s = model->getBaseScale();
    inst.scale[0] = s->x;
    inst.scale[1] = s->y;
    inst.scale[2] = s->z;
    for (const Instance& other : g_instances) {
        if (other.md == inst.md && std::memcmp(other.mtx, inst.mtx, sizeof(inst.mtx)) == 0) {
            return;
        }
    }
    g_instances.push_back(inst);
}

// ---------------------------------------------------------------------------------------------
// Water bodies: connected pieces of the captured surface, each with its own style.
// ---------------------------------------------------------------------------------------------
enum BodyStyle { STYLE_OPEN = 0, STYLE_STILL = 1, STYLE_FLAT = 2, STYLE_FLOWING = 3 };

constexpr float kOpenMinArea = 1.2e6f;     // smaller bodies never get waves
constexpr float kOpenMinDepth = 120.0f;    // nor do shallow ones
constexpr float kWaveEdgeNear = 60.0f;     // waves fade out toward the shore over this range...
constexpr float kWaveEdgeFar = 600.0f;
constexpr size_t kMaxFrameVerts = 200000;
constexpr size_t kMaxBodyTris = 40000;

struct Seg {
    float ax, az, bx, bz;
};

struct BodyMesh {
    std::vector<WTri> tris; // base triangles, world space
    float area = 0.0f;
    float ymin = 1e30f, ymax = -1e30f;
    float x0 = 1e30f, z0 = 1e30f, x1 = -1e30f, z1 = -1e30f;
    int style = -1;
    bool built = false;
    std::vector<float> verts; // x y z w, 4 floats per vertex; w = style * 8 + wave amplitude
    std::vector<Seg> segs;    // shoreline: base-mesh edges used by one triangle only
    std::unordered_map<int64_t, std::vector<int>> segGrid;
    int deferred = 0;
};

constexpr float kSegCell = 800.0f;

int64_t cell_key(int ix, int iz) {
    return (static_cast<int64_t>(ix) << 32) ^ static_cast<int64_t>(static_cast<uint32_t>(iz));
}

float boundary_dist(const BodyMesh& b, float x, float z) {
    auto it = b.segGrid.find(cell_key(static_cast<int>(std::floor(x / kSegCell)),
        static_cast<int>(std::floor(z / kSegCell))));
    if (it == b.segGrid.end()) {
        return 1e9f;
    }
    float best = 1e18f;
    for (int si : it->second) {
        const Seg& s = b.segs[si];
        const float dx = s.bx - s.ax;
        const float dz = s.bz - s.az;
        const float len2 = dx * dx + dz * dz;
        float t = len2 > 1e-6f ? ((x - s.ax) * dx + (z - s.az) * dz) / len2 : 0.0f;
        t = std::clamp(t, 0.0f, 1.0f);
        const float px = s.ax + dx * t - x;
        const float pz = s.az + dz * t - z;
        best = std::min(best, px * px + pz * pz);
    }
    return std::sqrt(best);
}

float wave_amp_at(const BodyMesh& b, float x, float z) {
    if (b.style != STYLE_OPEN) {
        return 0.0f;
    }
    return smoothstep_f(kWaveEdgeNear, kWaveEdgeFar, boundary_dist(b, x, z));
}

int64_t vert_key(const V3& v) {
    const int64_t ix = std::llround(v.x) & 0x1FFFFF;
    const int64_t iy = std::llround(v.y) & 0x1FFFFF;
    const int64_t iz = std::llround(v.z) & 0x1FFFFF;
    return (ix << 42) | (iz << 21) | iy;
}

void build_bodies(const std::vector<WTri>& local, const Instance& inst, std::vector<BodyMesh>& out) {
    out.clear();
    std::vector<WTri> world(local.size());
    for (size_t i = 0; i < local.size(); ++i) {
        for (int k = 0; k < 3; ++k) {
            const V3& p = local[i].p[k];
            const float x = p.x * inst.scale[0];
            const float y = p.y * inst.scale[1];
            const float z = p.z * inst.scale[2];
            world[i].p[k] = {inst.mtx[0] * x + inst.mtx[1] * y + inst.mtx[2] * z + inst.mtx[3],
                inst.mtx[4] * x + inst.mtx[5] * y + inst.mtx[6] * z + inst.mtx[7],
                inst.mtx[8] * x + inst.mtx[9] * y + inst.mtx[10] * z + inst.mtx[11]};
        }
    }
    // Union triangles that share a vertex.
    std::vector<int> parent(world.size());
    for (size_t i = 0; i < parent.size(); ++i) {
        parent[i] = static_cast<int>(i);
    }
    auto find = [&](int a) {
        while (parent[a] != a) {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        return a;
    };
    std::unordered_map<int64_t, int> owner;
    for (size_t i = 0; i < world.size(); ++i) {
        for (int k = 0; k < 3; ++k) {
            auto ins = owner.emplace(vert_key(world[i].p[k]), static_cast<int>(i));
            if (!ins.second) {
                const int a = find(ins.first->second);
                const int b = find(static_cast<int>(i));
                if (a != b) {
                    parent[b] = a;
                }
            }
        }
    }
    std::unordered_map<int, size_t> bodyOf;
    for (size_t i = 0; i < world.size(); ++i) {
        const int root = find(static_cast<int>(i));
        auto it = bodyOf.find(root);
        if (it == bodyOf.end()) {
            it = bodyOf.emplace(root, out.size()).first;
            out.emplace_back();
        }
        BodyMesh& b = out[it->second];
        b.tris.push_back(world[i]);
        const WTri& t = world[i];
        b.area += 0.5f * std::fabs((t.p[1].x - t.p[0].x) * (t.p[2].z - t.p[0].z) -
                                   (t.p[1].z - t.p[0].z) * (t.p[2].x - t.p[0].x));
        for (int k = 0; k < 3; ++k) {
            b.ymin = std::min(b.ymin, t.p[k].y);
            b.ymax = std::max(b.ymax, t.p[k].y);
            b.x0 = std::min(b.x0, t.p[k].x);
            b.x1 = std::max(b.x1, t.p[k].x);
            b.z0 = std::min(b.z0, t.p[k].z);
            b.z1 = std::max(b.z1, t.p[k].z);
        }
    }
}

void build_shoreline(BodyMesh& b) {
    struct EdgeUse {
        V3 a, c;
        int count;
    };
    std::unordered_map<uint64_t, EdgeUse> edges;
    for (const WTri& t : b.tris) {
        for (int k = 0; k < 3; ++k) {
            const V3& a = t.p[k];
            const V3& c = t.p[(k + 1) % 3];
            const int64_t ka = vert_key(a);
            const int64_t kc = vert_key(c);
            const uint64_t lo = static_cast<uint64_t>(std::min(ka, kc));
            const uint64_t hi = static_cast<uint64_t>(std::max(ka, kc));
            const uint64_t key = lo * 1000003ull ^ (hi * 0x9E3779B97F4A7C15ull);
            auto it = edges.find(key);
            if (it == edges.end()) {
                edges.emplace(key, EdgeUse{a, c, 1});
            } else {
                ++it->second.count;
            }
        }
    }
    b.segs.clear();
    b.segGrid.clear();
    for (const auto& kv : edges) {
        if (kv.second.count != 1) {
            continue;
        }
        b.segs.push_back({kv.second.a.x, kv.second.a.z, kv.second.c.x, kv.second.c.z});
    }
    const float reach = kWaveEdgeFar + 1.0f;
    for (size_t i = 0; i < b.segs.size(); ++i) {
        const Seg& s = b.segs[i];
        const int ix0 = static_cast<int>(std::floor((std::min(s.ax, s.bx) - reach) / kSegCell));
        const int ix1 = static_cast<int>(std::floor((std::max(s.ax, s.bx) + reach) / kSegCell));
        const int iz0 = static_cast<int>(std::floor((std::min(s.az, s.bz) - reach) / kSegCell));
        const int iz1 = static_cast<int>(std::floor((std::max(s.az, s.bz) + reach) / kSegCell));
        for (int ix = ix0; ix <= ix1; ++ix) {
            for (int iz = iz0; iz <= iz1; ++iz) {
                b.segGrid[cell_key(ix, iz)].push_back(static_cast<int>(i));
            }
        }
    }
}

// Conforming subdivision: an edge is split exactly when it is longer than `limit`, a decision that
// depends only on the edge's endpoints, so neighbouring triangles always agree and no cracks form.
struct Subdivider {
    float limit;
    std::vector<WTri>* out;
    size_t cap;
    bool overflow = false;

    static V3 mid(const V3& a, const V3& b) {
        return {(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f, (a.z + b.z) * 0.5f};
    }
    static float len(const V3& a, const V3& b) {
        const float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    void run(V3 a, V3 b, V3 c, int depth) {
        if (overflow) {
            return;
        }
        int mask = 0;
        if (depth < 14) {
            mask = (len(a, b) > limit ? 1 : 0) | (len(b, c) > limit ? 2 : 0) |
                   (len(c, a) > limit ? 4 : 0);
        }
        if (mask == 0) {
            if (out->size() >= cap) {
                overflow = true;
                return;
            }
            out->push_back({{a, b, c}});
            return;
        }
        // Rotate so the split edges take a canonical position (AB for one, AB+BC for two).
        auto rotate = [&]() {
            const V3 t = a;
            a = b;
            b = c;
            c = t;
            mask = ((mask << 2) | (mask >> 1)) & 7; // edge AB->BC->CA->AB
        };
        const int pop = (mask & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1);
        if (pop == 1) {
            while (mask != 1) {
                rotate();
            }
            const V3 m = mid(a, b);
            run(a, m, c, depth + 1);
            run(m, b, c, depth + 1);
        } else if (pop == 2) {
            while (mask != 3) {
                rotate();
            }
            const V3 m1 = mid(a, b);
            const V3 m2 = mid(b, c);
            run(m1, b, m2, depth + 1);
            run(a, m1, m2, depth + 1);
            run(a, m2, c, depth + 1);
        } else {
            const V3 m0 = mid(a, b);
            const V3 m1 = mid(b, c);
            const V3 m2 = mid(c, a);
            run(a, m0, m2, depth + 1);
            run(m0, b, m1, depth + 1);
            run(m2, m1, c, depth + 1);
            run(m0, m1, m2, depth + 1);
        }
    }
};

void build_mesh(BodyMesh& b) {
    build_shoreline(b);
    std::vector<WTri> tris;
    if (b.style == STYLE_OPEN) {
        float limit = std::max(160.0f, std::sqrt(b.area / (0.43f * 30000.0f)));
        for (int attempt = 0; attempt < 6; ++attempt) {
            tris.clear();
            Subdivider sub{limit, &tris, kMaxBodyTris};
            for (const WTri& t : b.tris) {
                sub.run(t.p[0], t.p[1], t.p[2], 0);
            }
            if (!sub.overflow) {
                break;
            }
            limit *= 1.7f;
        }
    } else {
        tris = b.tris;
    }
    b.verts.clear();
    b.verts.reserve(tris.size() * 12);
    for (const WTri& t : tris) {
        for (int k = 0; k < 3; ++k) {
            const V3& v = t.p[k];
            const float amp = wave_amp_at(b, v.x, v.z);
            b.verts.push_back(v.x);
            b.verts.push_back(v.y);
            b.verts.push_back(v.z);
            b.verts.push_back(static_cast<float>(b.style) * 8.0f + amp);
        }
    }
    b.built = true;
}

// Depth of the ground below the water at a point; `found` is false when there is no collision.
float probe_depth_ex(float x, float waterY, float z, bool& found) {
    static dBgS_GndChk gnd;
    cXyz pos(x, waterY - 2.0f, z);
    gnd.SetPos(&pos);
    const float groundY = dComIfG_Bgsp().GroundCross(&gnd);
    found = groundY != -G_CM3D_F_INF;
    return found ? std::max(waterY - groundY, 0.0f) : 2000.0f;
}

bool decide_style(BodyMesh& b) {
    if (in_morpheel_arena()) {
        b.style = STYLE_FLAT;
        return true;
    }
    if (b.ymax - b.ymin > 60.0f) {
        b.style = STYLE_FLOWING;
        return true;
    }
    if (b.area < kOpenMinArea) {
        b.style = STYLE_STILL;
        return true;
    }
    std::vector<float> depths;
    const size_t step = std::max<size_t>(1, b.tris.size() / 16);
    for (size_t i = 0; i < b.tris.size(); i += step) {
        const WTri& t = b.tris[i];
        const float cx = (t.p[0].x + t.p[1].x + t.p[2].x) / 3.0f;
        const float cy = (t.p[0].y + t.p[1].y + t.p[2].y) / 3.0f;
        const float cz = (t.p[0].z + t.p[1].z + t.p[2].z) / 3.0f;
        bool found = false;
        const float d = probe_depth_ex(cx, cy, cz, found);
        if (found) {
            depths.push_back(d);
        }
    }
    if (depths.empty()) {
        if (++b.deferred < 120) {
            return false; // collision not loaded yet; try again next frame
        }
        b.style = STYLE_OPEN;
        return true;
    }
    std::sort(depths.begin(), depths.end());
    b.style = depths[depths.size() / 2] >= kOpenMinDepth ? STYLE_OPEN : STYLE_STILL;
    return true;
}

struct WorldWater {
    float mtx[12];
    float scale[3];
    unsigned gen = 0;
    uint64_t lastFrame = 0;
    std::vector<BodyMesh> bodies;
};
std::unordered_map<const void*, std::vector<WorldWater>> g_worldWater;
std::vector<BodyMesh*> g_activeBodies; // bodies drawn this frame (game thread only)
uint64_t g_frameNo = 0;

WorldWater* get_world_water(const Instance& inst, const LocalWater& lw) {
    std::vector<WorldWater>& list = g_worldWater[inst.md];
    for (WorldWater& w : list) {
        if (w.gen == lw.gen && std::memcmp(w.mtx, inst.mtx, sizeof(w.mtx)) == 0 &&
            std::memcmp(w.scale, inst.scale, sizeof(w.scale)) == 0)
        {
            return &w;
        }
    }
    if (list.size() >= 6) {
        list.erase(list.begin());
    }
    list.emplace_back();
    WorldWater& w = list.back();
    std::memcpy(w.mtx, inst.mtx, sizeof(w.mtx));
    std::memcpy(w.scale, inst.scale, sizeof(w.scale));
    w.gen = lw.gen;
    build_bodies(lw.tris, inst, w.bodies);
    return &w;
}

// CPU lookup used by the swim-height hook: wave amplitude (0..1) at a point if it lies on one of
// the drawn open-water bodies at about the given height.
bool water_amp_at(float x, float y, float z, float& amp) {
    for (BodyMesh* b : g_activeBodies) {
        if (b->style != STYLE_OPEN || x < b->x0 || x > b->x1 || z < b->z0 || z > b->z1) {
            continue;
        }
        for (const WTri& t : b->tris) {
            const float d = (t.p[1].z - t.p[2].z) * (t.p[0].x - t.p[2].x) +
                            (t.p[2].x - t.p[1].x) * (t.p[0].z - t.p[2].z);
            if (std::fabs(d) < 1e-6f) {
                continue;
            }
            const float l1 = ((t.p[1].z - t.p[2].z) * (x - t.p[2].x) +
                                 (t.p[2].x - t.p[1].x) * (z - t.p[2].z)) / d;
            const float l2 = ((t.p[2].z - t.p[0].z) * (x - t.p[2].x) +
                                 (t.p[0].x - t.p[2].x) * (z - t.p[2].z)) / d;
            const float l3 = 1.0f - l1 - l2;
            if (l1 < -0.001f || l2 < -0.001f || l3 < -0.001f) {
                continue;
            }
            const float h = l1 * t.p[0].y + l2 * t.p[1].y + l3 * t.p[2].y;
            if (std::fabs(h - y) > 40.0f) {
                continue;
            }
            amp = wave_amp_at(*b, x, z);
            return true;
        }
    }
    return false;
}

DEFINE_HOOK(&dBgS::SplGrpChk, SplGrpCheck);

void on_spl_grp_chk_post(ModContext*, void* args, void* retval, void*) {
    if (g_inOwnProbe || g_activeBodies.empty() || retval == nullptr || !*static_cast<bool*>(retval)) {
        return;
    }
    if (!get_bool_option(g_cvarEnabled, true) || !get_bool_option(g_cvarSwim, true)) {
        return;
    }
    dBgS_SplGrpChk* chk = mods::arg<dBgS_SplGrpChk*>(args, 1);
    if (chk == nullptr) {
        return;
    }
    const float x = chk->GetPosP().x;
    const float z = chk->GetPosP().z;
    float amp = 0.0f;
    if (!water_amp_at(x, chk->GetHeight(), z, amp) || amp <= 0.0f) {
        return;
    }
    const float heightScale =
        static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarWaveHeight, 100), 0, 300)) /
        100.0f;
    chk->SetHeight(chk->GetHeight() + cpu_wave_height(x, z, elapsed_seconds()) * heightScale * amp);
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
    float ripple[4];
    float glint[4];
    float glint_col[4];
    float spray[8][4];
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
    fragment.entryPoint = {"fs_mesh", WGPU_STRLEN};
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
    desc.vertex.entryPoint = {"vs_mesh", WGPU_STRLEN};
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

constexpr uint32_t kSprayPerEmitter = 48; // must match SPRAY_N in water.wgsl
constexpr uint32_t kSunCols = 25;
constexpr uint32_t kSunRows = 26;

void release_sun_pipeline() {
    for (int i = 0; i < 3; ++i) {
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
            .dstFactor = mode == 1 ? WGPUBlendFactor_One : WGPUBlendFactor_OneMinusSrcAlpha},
        .alpha = {.operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_Zero,
            .dstFactor = WGPUBlendFactor_One},
    };
    WGPUColorTargetState colorTargets[GFX_MAX_COLOR_ATTACHMENTS];
    const uint32_t colorTargetCount =
        gfx_init_color_target_states(&layout, colorTargets, &blend, WGPUColorWriteMask_All);

    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = module;
    fragment.entryPoint = {mode == 0 ? "fs_sea" : (mode == 1 ? "fs_sun" : "fs_spray"), WGPU_STRLEN};
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
    desc.vertex.entryPoint = {mode == 2 ? "vs_spray" : "vs_sun", WGPU_STRLEN};
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
    if (data.mode == 2) {
        // Waterfall spray: built on first use, independently of the other two layers.
        if (data.depth == nullptr) {
            return;
        }
        if (g_sunPipelines[2] == nullptr && !build_sun_pipeline(ctx->layout, 2)) {
            return;
        }
        WGPUBindGroupEntry sprayEntries[2] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
        sprayEntries[0].binding = 0;
        sprayEntries[0].buffer = ctx->uniform_buffer;
        sprayEntries[0].offset = data.uniform_offset;
        sprayEntries[0].size = data.uniform_size;
        sprayEntries[1].binding = 3;
        sprayEntries[1].textureView = data.depth;
        WGPUBindGroupDescriptor sprayDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        sprayDesc.layout = g_sunLayouts[2];
        sprayDesc.entryCount = 2;
        sprayDesc.entries = sprayEntries;
        WGPUBindGroup sprayGroup = wgpuDeviceCreateBindGroup(ctx->device, &sprayDesc);
        if (sprayGroup == nullptr) {
            return;
        }
        wgpuRenderPassEncoderSetPipeline(ctx->pass, g_sunPipelines[2]);
        wgpuRenderPassEncoderSetBindGroup(ctx->pass, 0, sprayGroup, 0, nullptr);
        wgpuRenderPassEncoderDraw(ctx->pass, data.vertex_count, 1, 0, 0);
        wgpuBindGroupRelease(sprayGroup);
        return;
    }
    if (data.depth == nullptr || g_waveView == nullptr || g_repeatSampler == nullptr ||
        data.mode > 1 || (data.mode == 0 && (data.color == nullptr || g_sampler == nullptr)))
    {
        return;
    }

    WGPUBindGroupEntry entries[7] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT,
        WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT,
        WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
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
        entries[6].binding = 4; // the refraction layer samples the scene colour snapshot
        entries[6].sampler = g_sampler;
        entryCount = 7;
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
    set4(uni.ripple, g_rippleIntensity, 0.0f, 0.0f, 0.0f);
    {
        // Glitter path: toward the sun while it is up, otherwise the moon.
        auto direction = [&](const cXyz& p, float out[3]) {
            float dx = p.x - camera.eye[0];
            float dy = p.y - camera.eye[1];
            float dz = p.z - camera.eye[2];
            const float len = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (len < 1.0f) {
                out[0] = 0.0f;
                out[1] = 1.0f;
                out[2] = 0.0f;
                return;
            }
            out[0] = dx / len;
            out[1] = dy / len;
            out[2] = dz / len;
        };
        float sd[3];
        float md[3];
        direction(env.sun_pos, sd);
        direction(env.moon_pos, md);
        if (sd[1] > 0.04f) {
            set4(uni.glint, sd[0], sd[1], sd[2], std::clamp(sd[1] * 4.0f, 0.0f, 1.0f));
            set4(uni.glint_col, 1.0f, 0.88f, 0.68f, 1.0f);
        } else if (md[1] > 0.04f) {
            set4(uni.glint, md[0], md[1], md[2], std::clamp(md[1] * 4.0f, 0.0f, 1.0f));
            set4(uni.glint_col, 0.62f, 0.76f, 1.0f, 1.0f);
        } else {
            set4(uni.glint, 0.0f, 1.0f, 0.0f, 0.0f);
            set4(uni.glint_col, 0.0f, 0.0f, 0.0f, 1.0f);
        }
    }
    for (int i = 0; i < 8; ++i) {
        const bool on = i < g_sprayCount && get_bool_option(g_cvarSpray, true);
        set4(uni.spray[i], g_spray[i][0], g_spray[i][1], g_spray[i][2], on ? g_spray[i][3] : 0.0f);
    }
    set4(uni.sun2, refract, get_int_option(g_cvarMode, 1) == 1 ? 1.0f : 0.0f,
        static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarColor, 0), 0, 2)), overlay);
}

// Game thread, after opaque scene draws and before translucent overlays. Builds the water meshes
// for the models captured this frame and draws them.
std::vector<float> g_meshScratch;
bool g_loggedBudget = false;

void on_scene_after_opaque(ModContext*, const GfxStageContext* stageCtx, void*) {
    ++g_frameNo;
    g_frameValid = false;
    g_activeBodies.clear();
    std::vector<Instance> instances;
    instances.swap(g_instances);
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
    {
        // Swim ripples: on while Link is in or at the surface of the water, stronger when moving.
        float refH = 0.0f;
        const bool inWaterZone = probe_water(g_framePlayer[0], playerY, g_framePlayer[2], refH) &&
                                 refH <= playerY + 250.0f;
        static float lastX = 0.0f;
        static float lastZ = 0.0f;
        static float lastT = -1.0f;
        const float now = elapsed_seconds();
        const float dt = now - lastT;
        float speed = 0.0f;
        if (lastT >= 0.0f && dt > 1e-4f && dt < 0.25f) {
            speed = std::hypot(g_framePlayer[0] - lastX, g_framePlayer[2] - lastZ) / dt;
        }
        lastX = g_framePlayer[0];
        lastZ = g_framePlayer[2];
        lastT = now;
        const bool swimming = inWaterZone && playerY <= refH + 30.0f && playerY >= refH - 250.0f;
        const float target = swimming ? std::clamp(0.3f + speed / 250.0f, 0.3f, 1.0f) : 0.0f;
        g_rippleIntensity += (target - g_rippleIntensity) * std::clamp(dt * 4.0f, 0.0f, 1.0f);
    }

    g_meshScratch.clear();
    for (const Instance& inst : instances) {
        auto lwIt = g_localWater.find(inst.md);
        if (lwIt == g_localWater.end() || lwIt->second.tris.empty()) {
            continue;
        }
        WorldWater* ww = get_world_water(inst, lwIt->second);
        ww->lastFrame = g_frameNo;
        for (BodyMesh& body : ww->bodies) {
            if (body.style < 0 && !decide_style(body)) {
                continue;
            }
            if (!body.built) {
                build_mesh(body);
            }
            if (g_meshScratch.size() / 4 + body.verts.size() / 4 > kMaxFrameVerts) {
                if (!g_loggedBudget) {
                    g_loggedBudget = true;
                    svc_log->warn(mod_ctx, "water mesh budget reached; some water not drawn");
                }
                continue;
            }
            g_meshScratch.insert(g_meshScratch.end(), body.verts.begin(), body.verts.end());
            g_activeBodies.push_back(&body);
        }
    }
    // Drop meshes of models that have not been drawn for a while.
    if ((g_frameNo & 255u) == 0) {
        for (auto it = g_worldWater.begin(); it != g_worldWater.end();) {
            auto& list = it->second;
            list.erase(std::remove_if(list.begin(), list.end(),
                           [](const WorldWater& w) { return g_frameNo - w.lastFrame > 900; }),
                list.end());
            it = list.empty() ? g_worldWater.erase(it) : std::next(it);
        }
    }
    if (g_meshScratch.empty()) {
        return;
    }

    g_frameCamera = camera;

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
        svc_gfx->push_storage(mod_ctx, g_meshScratch.data(), g_meshScratch.size() * sizeof(float),
            &storageRange) != MOD_OK)
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
    payload.vertex_count = static_cast<uint32_t>(g_meshScratch.size() / 4);
    if (svc_gfx->push_draw(mod_ctx, g_drawType, &payload, sizeof(payload)) == MOD_OK &&
        !g_loggedFirstDraw)
    {
        g_loggedFirstDraw = true;
        svc_log->info(mod_ctx, "first water mesh queued");
    }
}

// Game thread, after the whole 3D scene (stock water included) and before the HUD: the Sunshine
// wave overlay is drawn here, on top of the stock water, like TMapObjWave is in Sunshine.
void on_frame_before_hud(ModContext*, const GfxStageContext*, void*) {
    if (!g_frameValid) {
        return;
    }
    g_frameValid = false;
    // Rolling waves replaces the water entirely and draws its own shoreline foam; the speckle
    // overlay is only for the Subtle style.
    if (get_int_option(g_cvarMode, 1) == 1 || g_sunDrawType == 0) {
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
    if (get_int_option(g_cvarMode, 1) != 1 && get_int_option(g_cvarRefract, 100) > 0) {
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

    control = UI_CONTROL_DESC_INIT;
    control.kind = UI_CONTROL_TOGGLE;
    control.label = "Waves move swimming height";
    control.help_rml = "Link, items and splashes rise and fall with the waves on open water, like "
                       "Wind Waker's sea. Only applies in Rolling waves style.";
    control.binding = UI_BINDING_CONFIG_VAR;
    control.config_var = g_cvarSwim;
    svc_ui->pane_add_control(mod_ctx, panel, &control, nullptr);

    control = UI_CONTROL_DESC_INIT;
    control.kind = UI_CONTROL_TOGGLE;
    control.label = "Waterfall spray";
    control.help_rml = "Spray droplets, splash foam and ripples where water falls into a pool. "
                       "Only applies in Rolling waves style.";
    control.binding = UI_BINDING_CONFIG_VAR;
    control.config_var = g_cvarSpray;
    svc_ui->pane_add_control(mod_ctx, panel, &control, nullptr);

    static const char* kModeOptions[] = {"Subtle overlay", "Rolling waves"};
    control = UI_CONTROL_DESC_INIT;
    control.kind = UI_CONTROL_SELECT;
    control.label = "Style";
    control.help_rml = "Subtle overlay: Sunshine's sparkle and wobble drawn over the game's own "
                       "water.<br/>Rolling waves: a replacement water surface with visible "
                       "swell, reflections and refraction, plus Sunshine's foam.";
    control.binding = UI_BINDING_CONFIG_VAR;
    control.config_var = g_cvarMode;
    control.options = kModeOptions;
    control.option_count = 2;
    svc_ui->pane_add_control(mod_ctx, panel, &control, nullptr);

    static const char* kColorOptions[] = {"Sunshine turquoise", "Twilight green", "Deep blue"};
    control = UI_CONTROL_DESC_INIT;
    control.kind = UI_CONTROL_SELECT;
    control.label = "Water Colour";
    control.help_rml = "Body colour of the water in Rolling waves style.";
    control.binding = UI_BINDING_CONFIG_VAR;
    control.config_var = g_cvarColor;
    control.options = kColorOptions;
    control.option_count = 3;
    svc_ui->pane_add_control(mod_ctx, panel, &control, nullptr);

    add_number(panel, "Foam Intensity", "Shoreline foam (Rolling waves) or sparkle brightness (Subtle overlay).", g_cvarOverlay, 0,
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
        register_bool_option("wavesMoveSwimming", true, g_cvarSwim, error) != MOD_OK ||
        register_bool_option("waterfallSpray", true, g_cvarSpray, error) != MOD_OK ||
        register_int_option("waveHeight", 100, g_cvarWaveHeight, error) != MOD_OK ||
        register_int_option("rippleStrength", 100, g_cvarNormals, error) != MOD_OK ||
        register_int_option("clarity", 100, g_cvarClarity, error) != MOD_OK ||
        register_int_option("distanceHaze", 100, g_cvarFog, error) != MOD_OK ||
        register_int_option("debugView", 0, g_cvarDebug, error) != MOD_OK ||
        register_int_option("style", 1, g_cvarMode, error) != MOD_OK ||
        register_int_option("foamIntensity", 100, g_cvarOverlay, error) != MOD_OK ||
        register_int_option("refraction", 100, g_cvarRefract, error) != MOD_OK ||
        register_int_option("waterColor", 0, g_cvarColor, error) != MOD_OK)
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
    if (mods::hook::add_post<BgMaxxProc>(on_bg_maxx_post) != MOD_OK) {
        svc_log->warn(mod_ctx, "could not hook stage material scan");
    }
    // Wave collision: add the wave height to the game's water height queries on open water.
    if (mods::hook::add_post<SplGrpCheck>(on_spl_grp_chk_post) != MOD_OK) {
        svc_log->warn(mod_ctx, "could not hook water collision; swimming height stays flat");
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
    g_cvarSpray = g_cvarSwim = g_cvarEnabled = g_cvarWaveHeight = g_cvarNormals = g_cvarClarity = g_cvarFog = g_cvarDebug = 0;
    g_cvarMode = g_cvarOverlay = g_cvarRefract = g_cvarColor = 0;
    g_drawType = g_sunDrawType = 0;
    g_stageHook = g_hudHook = 0;
    return MOD_OK;
}
}
