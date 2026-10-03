// Better Water: wave surface drawn after opaque scene geometry.
//
// The surface is a camera-following grid generated entirely from vertex_index (no vertex buffer).
// A coarse lattice of collision water queries (built on the CPU, one vec4 per cell:
// height, depth, state, 0) decides where water is, how high it is and whether it is open water
// (waves) or stagnant (left to the stock renderer). Depth is written so the stock water material
// (depth test LEQUAL, no depth write) is hidden wherever this surface is in front of it.

struct U {
    proj_from_world: mat4x4f,
    world_from_proj: mat4x4f,
    eye: vec4f,      // xyz camera position
    sky: vec4f,      // zenith colour
    horizon: vec4f,  // horizon / fog colour
    amb: vec4f,      // ambient light colour (brightness of the scene)
    sun: vec4f,      // xyz direction to the sun, w = visibility
    params: vec4f,   // x time, y wave height scale, z normal strength, w fog density
    grid: vec4f,     // x origin, y origin, z step, w quads per side
    lat: vec4f,      // x ix0, y iz0, z width, w cell size
    screen: vec4f,   // x width, y height, z debug mode, w clarity
    lat2: vec4f,     // coarse lattice: x ix0, y iz0, z width, w cell size
    warp: vec4f,     // x linear step, y quadratic term of the grid radial warp, z fine radius, w reversed-Z
    sun0: vec4f,     // Sunshine overlay: x,y centre (player) xz, z,w wave phases
    sun1: vec4f,     // Sunshine overlay: x,y wave amplitudes, z,w texture scroll
    sun2: vec4f,     // Sunshine overlay: xyz colour tint, w intensity
}

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> lattice: array<vec4f>;
@group(0) @binding(2) var scene_color: texture_2d<f32>;
@group(0) @binding(3) var scene_depth: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var wave_tex: texture_2d<f32>;
@group(0) @binding(6) var wave_samp: sampler;

const WAVE_COUNT_VERTEX: i32 = 4;
const WAVE_COUNT_FRAG: i32 = 8;
const BASE_AMPLITUDE: f32 = 12.0;   // world units (1 unit is about 1 cm), scaled by params.y
const BASE_WAVELENGTH: f32 = 900.0;
const SWELL_SCALE: f32 = 0.6;       // fraction of Sunshine's swell amplitude used by the custom surface
const SURFACE_LIFT: f32 = 0.0;
const COVER_SLOPE: f32 = 0.012;     // extra cover per unit of camera distance
const COVER_LIFT: f32 = 12.0;       // depth is written as if the surface were at least this far above the stock water plane      // keeps the surface just above the stock water plane

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) world: vec3f,
}

// Level 0 = fine lattice (near the camera), level 1 = coarse lattice (far reach). Both live in
// one storage buffer: fine cells first, then coarse cells.
fn use_fine(xz: vec2f) -> bool {
    let d = max(abs(xz.x - u.eye.x), abs(xz.y - u.eye.z));
    return d < u.warp.z;
}

fn cell_at(level: i32, i: i32, j: i32) -> vec4f {
    let w = i32(u.lat.z);
    if i < 0 || j < 0 || i >= w || j >= w {
        return vec4f(0.0);
    }
    return lattice[u32(level * w * w + j * w + i)];
}

// Bilinear lattice query. Returns (height, depth, openness, anyWater)
fn surface_info(xz: vec2f) -> vec4f {
    var level = 1;
    var origin = u.lat2.xy;
    var cell = u.lat2.w;
    if use_fine(xz) {
        level = 0;
        origin = u.lat.xy;
        cell = u.lat.w;
    }
    let c = xz / cell - origin - vec2f(0.5);
    let b = floor(c);
    let f = c - b;
    let bi = vec2i(b);
    let s00 = cell_at(level, bi.x, bi.y);
    let s10 = cell_at(level, bi.x + 1, bi.y);
    let s01 = cell_at(level, bi.x, bi.y + 1);
    let s11 = cell_at(level, bi.x + 1, bi.y + 1);
    let w00 = (1.0 - f.x) * (1.0 - f.y);
    let w10 = f.x * (1.0 - f.y);
    let w01 = (1.0 - f.x) * f.y;
    let w11 = f.x * f.y;
    let a00 = select(0.0, w00, s00.z > 0.5);
    let a10 = select(0.0, w10, s10.z > 0.5);
    let a01 = select(0.0, w01, s01.z > 0.5);
    let a11 = select(0.0, w11, s11.z > 0.5);
    let sum = a00 + a10 + a01 + a11;
    if sum < 1e-4 {
        return vec4f(0.0);
    }
    let h = (s00.x * a00 + s10.x * a10 + s01.x * a01 + s11.x * a11) / sum;
    let d = (s00.y * a00 + s10.y * a10 + s01.y * a01 + s11.y * a11) / sum;
    let o00 = select(0.0, w00, s00.z > 1.5);
    let o10 = select(0.0, w10, s10.z > 1.5);
    let o01 = select(0.0, w01, s01.z > 1.5);
    let o11 = select(0.0, w11, s11.z > 1.5);
    return vec4f(h, d, o00 + o10 + o01 + o11, sum);
}

// Sum of sharp-crested waves, exp(sin - 1): always >= 0, so the surface never dips below the
// stock water plane (which would let it show through). Returns (height, d/dx, d/dz).
// Sunshine's two long rolling swells (TMapObjWave), shifted to be >= 0 so they only lift the
// surface above the stock water plane. Returns (height, d/dx, d/dz).
fn sun_swell(p: vec2f) -> vec3f {
    let k1 = 0.02 / 6.28318;
    let k2 = 0.03 / 6.28318;
    let a1 = u.sun1.x * SWELL_SCALE;
    let a2 = u.sun1.y * SWELL_SCALE;
    let s1 = sin(k1 * p.x + u.sun0.z);
    let s2 = sin(k2 * p.y + u.sun0.w);
    return vec3f(a1 * s1 + a2 * s2,
        a1 * k1 * cos(k1 * p.x + u.sun0.z), a2 * k2 * cos(k2 * p.y + u.sun0.w));
}

fn waves(p: vec2f, t: f32, count: i32, dist: f32) -> vec3f {
    let sw = sun_swell(p);
    var h = sw.x;
    var dx = sw.y;
    var dz = sw.z;
    for (var i = 0; i < count; i = i + 1) {
        let fi = f32(i);
        let angle = 0.6 + fi * 2.399963;
        let dir = vec2f(cos(angle), sin(angle));
        let wl = BASE_WAVELENGTH * pow(0.55, fi);
        let k = 6.2831853 / wl;
        let omega = sqrt(980.0 * k) * 0.6;
        // Short waves fade with distance: the mesh gets coarser and they would only alias.
        let lod = 1.0 - smoothstep(wl * 5.0, wl * 25.0, dist);
        let amp = BASE_AMPLITUDE * pow(0.62, fi) * lod;
        let th = k * dot(dir, p) - omega * t + fi * 1.7;
        let e = exp(sin(th) - 1.0);
        h = h + amp * (e - 0.466);
        let d = amp * e * cos(th) * k;
        dx = dx + d * dir.x;
        dz = dz + d * dir.y;
    }
    return vec3f(h, dx, dz);
}

fn amplitude_scale(open: f32, depth: f32) -> f32 {
    return u.params.y * open * smoothstep(40.0, 320.0, depth);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    var o: VOut;
    let n = u32(u.grid.w);
    let quad = vi / 6u;
    let corner = vi % 6u;
    var cx = array<u32, 6>(0u, 1u, 0u, 1u, 1u, 0u);
    var cz = array<u32, 6>(0u, 0u, 1u, 0u, 1u, 1u);
    let gx = quad % n + cx[corner];
    let gz = quad / n + cz[corner];
    let half = f32(n) * 0.5;
    let gu = f32(gx) - half;
    let gv = f32(gz) - half;
    // Dense near the camera, stretching toward the horizon.
    let x = u.eye.x + sign(gu) * (u.warp.x * abs(gu) + u.warp.y * gu * gu);
    let z = u.eye.z + sign(gv) * (u.warp.x * abs(gv) + u.warp.y * gv * gv);

    let info = surface_info(vec2f(x, z));
    if info.w < 1e-4 || info.z < 0.01 {
        o.pos = vec4f(2.0, 2.0, 2.0, 1.0); // outside clip volume: culled
        o.world = vec3f(0.0);
        return o;
    }
    let vdist = length(vec2f(x - u.eye.x, z - u.eye.z));
    let w = waves(vec2f(x, z), u.params.x, WAVE_COUNT_VERTEX, vdist);
    let y = info.x + SURFACE_LIFT + w.x * amplitude_scale(info.z, info.y);
    let world = vec3f(x, y, z);
    o.world = world;
    o.pos = u.proj_from_world * vec4f(world, 1.0);
    return o;
}

fn unproject(uv: vec2f, d: f32) -> vec3f {
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let w = u.world_from_proj * vec4f(ndc, d, 1.0);
    return w.xyz / w.w;
}

fn hash21(p: vec2f) -> f32 {
    var q = fract(p * vec2f(123.34, 456.21));
    q = q + dot(q, q + 45.32);
    return fract(q.x * q.y);
}

fn vnoise(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let s = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2f(1.0, 0.0));
    let c = hash21(i + vec2f(0.0, 1.0));
    let d = hash21(i + vec2f(1.0, 1.0));
    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

fn fbm(p: vec2f) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var q = p;
    for (var i = 0; i < 4; i = i + 1) {
        v = v + a * vnoise(q);
        q = q * 2.03 + vec2f(17.1, 9.2);
        a = a * 0.5;
    }
    return v;
}

fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.299, 0.587, 0.114));
}

fn sky_color(dir: vec3f) -> vec3f {
    let t = clamp(dir.y * 1.6, 0.0, 1.0);
    return mix(u.horizon.rgb, u.sky.rgb, t);
}

// Screen-space reflection: geometric ray march against the scene depth snapshot.
fn reflect_ray(p: vec3f, r: vec3f, fallback: vec3f) -> vec3f {
    let size = u.screen.xy;
    var t = 30.0;
    for (var i = 0; i < 32; i = i + 1) {
        let q = p + r * t;
        let clip = u.proj_from_world * vec4f(q, 1.0);
        if clip.w <= 0.0 {
            break;
        }
        let ndc = clip.xyz / clip.w;
        let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 {
            break;
        }
        let pix = vec2i(uv * size);
        let d = textureLoad(scene_depth, pix, 0).r;
        let sw = unproject(uv, d);
        let dist_scene = length(sw - u.eye.xyz);
        let dist_ray = length(q - u.eye.xyz);
        let thickness = 25.0 + t * 0.12;
        if dist_ray > dist_scene && dist_ray - dist_scene < thickness && sw.y > p.y {
            let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
            let c = textureSampleLevel(scene_color, samp, uv, 0.0).rgb;
            return mix(fallback, c, smoothstep(0.0, 0.08, edge));
        }
        t = t * 1.17;
    }
    return fallback;
}

fn shade_water(frag: vec4f, world: vec3f) -> vec4f {
    let info = surface_info(world.xz);
    // Nearest-cell mask: only open water is replaced; stagnant cells keep the stock water.
    var here: vec4f;
    if use_fine(world.xz) {
        let cell = vec2i(floor(world.xz / u.lat.w) - u.lat.xy);
        here = cell_at(0, cell.x, cell.y);
    } else {
        let cell = vec2i(floor(world.xz / u.lat2.w) - u.lat2.xy);
        here = cell_at(1, cell.x, cell.y);
    }
    if here.z < 1.5 {
        discard;
    }

    let eye = u.eye.xyz;
    let to_eye = eye - world;
    let dist = length(to_eye);
    let V = to_eye / dist;

    // Fade out toward the edge of the replaced region: dithered, because depth is written.
    let reach = u.lat2.z * u.lat2.w * 0.5;
    let edge_fade = smoothstep(0.82, 1.0, length(world.xz - eye.xz) / reach);
    if edge_fade > hash21(frag.xy) {
        discard;
    }

    let t = u.params.x;
    let a_scale = amplitude_scale(info.z, info.y);
    let shallow = smoothstep(20.0, 260.0, info.y);

    // Normal: large waves + extra small octaves that only exist as shading.
    let w = waves(world.xz, t, WAVE_COUNT_FRAG, dist);
    let w_low = waves(world.xz, t, 3, dist);
    let strength = u.params.z * mix(0.35, 1.0, shallow) * max(u.params.y, 0.15);
    // Small ripples are always present, even if the big swell is turned down, but fade out
    // with distance so far water does not shimmer.
    let ripple_fade = 1.0 - smoothstep(300.0, 2500.0, dist);
    let ripple = (vnoise(world.xz * 0.05 + vec2f(t * 0.4, -t * 0.3)) - 0.5) * ripple_fade;
    let ripple2 = (vnoise(world.xz * 0.11 - vec2f(t * 0.5, t * 0.2)) - 0.5) * ripple_fade;
    // N: full detail (glint, refraction). N_low: smooth normal for reflections and Fresnel.
    var N = normalize(vec3f(-w.y * strength - ripple * 0.05, 1.0, -w.z * strength - ripple2 * 0.05));
    let N_low = normalize(vec3f(-w_low.y * strength, 1.0, -w_low.z * strength));

    let uv = frag.xy / u.screen.xy;
    let size = vec2i(u.screen.xy);

    // Scene beneath the surface.
    let d0 = textureLoad(scene_depth, vec2i(frag.xy), 0).r;
    let scene0 = unproject(uv, d0);
    var thick0 = max(world.y - scene0.y, 0.0);
    if length(scene0 - eye) > 60000.0 {
        thick0 = 2000.0;
    }

    // Refraction: offset by the normal, but never pull in something in front of the water.
    var uv_r = uv + N.xz * 0.03 * clamp(thick0 / 120.0, 0.0, 1.0) * u.screen.w;
    uv_r = clamp(uv_r, vec2f(0.002), vec2f(0.998));
    let d_r = textureLoad(scene_depth, vec2i(uv_r * u.screen.xy), 0).r;
    let scene_r = unproject(uv_r, d_r);
    var thick = thick0;
    var under_uv = uv;
    if length(scene_r - eye) > dist * 0.98 {
        under_uv = uv_r;
        thick = max(world.y - scene_r.y, 0.0);
        if length(scene_r - eye) > 60000.0 {
            thick = 2000.0;
        }
    }
    let under = textureSampleLevel(scene_color, samp, under_uv, 0.0).rgb;

    // Absorption: red goes first, then green, blue lasts longest.
    let lum = clamp(luminance(u.amb.rgb) * 1.4, 0.12, 1.0);
    // Water body colour: shallow tint fading into the deep tint with thickness. Palette 0 is the
    // turquoise-to-blue of Super Mario Sunshine's sea.
    var shallow_col = vec3f(0.08, 0.62, 0.66);
    var deep_tint = vec3f(0.02, 0.22, 0.50);
    if u.sun2.z > 1.5 {
        shallow_col = vec3f(0.05, 0.35, 0.60);
        deep_tint = vec3f(0.01, 0.08, 0.30);
    } else if u.sun2.z > 0.5 {
        shallow_col = vec3f(0.10, 0.40, 0.35);
        deep_tint = vec3f(0.03, 0.14, 0.16);
    }
    let deep_col = (mix(shallow_col, deep_tint, smoothstep(40.0, 700.0, thick)) +
                    u.horizon.rgb * 0.06) * (0.45 + 0.85 * lum);
    let absorb = vec3f(0.0120, 0.0046, 0.0030) / max(u.screen.w, 0.1);
    let trans = exp(-absorb * thick);
    let body = under * trans + deep_col * (vec3f(1.0) - trans);

    // Reflection.
    let cosv = clamp(dot(N_low, V), 0.0, 1.0);
    let fres = 0.02 + 0.98 * pow(1.0 - cosv, 5.0);
    let R_low = reflect(-V, N_low);
    let R = reflect(-V, N);
    let sky = sky_color(vec3f(R_low.x, abs(R_low.y), R_low.z));
    let refl = reflect_ray(world, R_low, sky);

    var col = mix(body, refl, clamp(fres * 0.85, 0.0, 0.85));

    // Sun glint.
    let sunv = u.sun.xyz;
    let spec = pow(max(dot(R, sunv), 0.0), 600.0) * 3.0 + pow(max(dot(R, sunv), 0.0), 60.0) * 0.15;
    col = col + vec3f(1.0, 0.95, 0.85) * spec * u.sun.w * lum;

    // Foam: shoreline (thin water) and wave crests.
    let fn1 = fbm(world.xz * 0.045 + vec2f(t * 0.05, t * 0.03));
    let fn2 = fbm(world.xz * 0.11 - vec2f(t * 0.04, -t * 0.06));
    let shore = (1.0 - smoothstep(3.0, 34.0, thick)) * smoothstep(0.0, 8.0, thick + 3.0);
    let band = 0.5 + 0.5 * sin(thick * 0.14 - t * 1.3 + fn1 * 6.0);
    var foam = shore * smoothstep(0.40, 0.85, fn1 * 0.6 + band * 0.55) * 0.6;
    let crest = smoothstep(0.55, 0.95, (w.x - sun_swell(world.xz).x) / (BASE_AMPLITUDE * 0.9) + 0.55) * clamp(a_scale, 0.0, 1.0);
    foam = foam + crest * smoothstep(0.45, 0.75, fn2) * 0.8;
    foam = clamp(foam, 0.0, 1.0);
    col = mix(col, vec3f(0.92, 0.96, 0.98) * (0.35 + 0.65 * lum), foam);

    // Distance fog toward the scene fog colour.
    let fog = 1.0 - exp(-dist * u.params.w);
    col = mix(col, u.horizon.rgb, clamp(fog, 0.0, 1.0));

    // Debug views: 1 = water mask, 2 = depth/thickness, 3 = normals.
    if u.screen.z > 0.5 {
        if u.screen.z < 1.5 {
            return vec4f(0.1, 0.9, 0.3, 1.0);
        } else if u.screen.z < 2.5 {
            return vec4f(vec3f(clamp(thick / 400.0, 0.0, 1.0)), 1.0);
        } else {
            return vec4f(N * 0.5 + 0.5, 1.0);
        }
    }
    return vec4f(max(col, vec3f(0.0)), 1.0);
}


// ===============================================================================================
// Sunshine wave overlay (TMapObjWave, ported from the Super Mario Sunshine decompilation).
//
// A 5200 x 5200 unit grid of 200 unit cells follows the player. Height is two crossing sines,
//   y = A1 * sin(0.02 * x / 2pi + phase1) + A2 * sin(0.03 * z / 2pi + phase2),
// each phase advancing every game frame. The mesh is drawn with two copies of wave.bti (an I4
// intensity texture used as alpha) scrolling in different directions:
//   stage 0: alpha = tex0.a * vertex.a
//   stage 1: alpha = 2 * tex1.a * previous alpha          (colour = vertex colour * 2, i.e. white)
// Pixels pass the alpha test only if alpha >= 0x55 or alpha <= 0x23, and are blended as
// src * srcAlpha + dst * srcColour, which with a white source is additive.
// Vertex alpha fades linearly with Chebyshev distance from the player.
// ===============================================================================================
const SUN_CELL: f32 = 200.0;
const SUN_HALF: f32 = 2600.0;
const SUN_COLS: u32 = 25u;

struct FOut {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

// The stock water layers are flat and are hidden by depth, so wave troughs that dip below the
// stock plane must still write the depth of that plane (plus a little cover): otherwise the flat
// stock water shows through the troughs.
@fragment
fn fs_main(@builtin(position) frag: vec4f, @location(0) world: vec3f) -> FOut {
    var o: FOut;
    o.color = shade_water(frag, world);
    o.depth = frag.z;
    // The cover grows with distance: depth precision falls off, and a thin cover lets the stock
    // water's sparkle texture z-fight through as a dotted grid on far water.
    let plane_y = surface_info(world.xz).x + COVER_LIFT + length(world - u.eye.xyz) * COVER_SLOPE;
    if world.y < plane_y {
        let dir = normalize(world - u.eye.xyz);
        if abs(dir.y) > 1e-4 {
            let t = (plane_y - u.eye.y) / dir.y;
            if t > 0.0 {
                let c = u.proj_from_world * vec4f(u.eye.xyz + dir * t, 1.0);
                let d = c.z / c.w;
                if u.warp.w > 0.5 {
                    o.depth = max(frag.z, d);
                } else {
                    o.depth = min(frag.z, d);
                }
            }
        }
    }
    return o;
}

struct SunOut {
    @builtin(position) pos: vec4f,
    @location(0) uv0: vec2f,
    @location(1) uv1: vec2f,
    @location(2) va: f32,
}

@vertex
fn vs_sun(@builtin(vertex_index) vi: u32) -> SunOut {
    var o: SunOut;
    let quad = vi / 6u;
    let corner = vi % 6u;
    var cx = array<u32, 6>(0u, 1u, 0u, 1u, 1u, 0u);
    var cz = array<u32, 6>(0u, 0u, 1u, 0u, 1u, 1u);
    let gx = quad % SUN_COLS + cx[corner];
    let gz = quad / SUN_COLS + cz[corner];
    let xo = -SUN_HALF + f32(gx) * SUN_CELL;
    let zo = -SUN_HALF + f32(gz) * SUN_CELL;
    let x = u.sun0.x + xo;
    let z = u.sun0.y + zo;

    let info = surface_info(vec2f(x, z));
    if info.w < 1e-4 || info.z < 0.01 {
        o.pos = vec4f(2.0, 2.0, 2.0, 1.0); // not open water: culled
        o.uv0 = vec2f(0.0);
        o.uv1 = vec2f(0.0);
        o.va = 0.0;
        return o;
    }
    // TMapObjWave::updateHeightAndAlpha: swell and alpha shrink toward the shore.
    let depth = info.y;
    let amp = clamp(depth / 400.0, 0.0, 1.0);
    let a1 = u.sun1.x * amp;
    let a2 = u.sun1.y * amp;
    let h = a1 * sin(0.02 * (x * (1.0 / 6.28318)) + u.sun0.z)
          + a2 * sin(0.03 * (z * (1.0 / 6.28318)) + u.sun0.w);
    var world = vec3f(x, info.x + h * u.params.y, z);
    if u.sun2.y > 0.5 {
        // "Rolling waves" style: follow the opaque surface so the foam is not depth-culled by it.
        let ws = waves(vec2f(x, z), u.params.x, 2, length(vec2f(xo, zo)));
        world.y = max(info.x + SURFACE_LIFT + ws.x * amplitude_scale(info.z, info.y),
                      info.x + COVER_LIFT + length(vec2f(xo, zo)) * COVER_SLOPE) + 12.0;
    }
    o.pos = u.proj_from_world * vec4f(world, 1.0);

    // TMapObjWave::getAlpha, then the shore ramp (unk54).
    let fade = floor(255.0 * (1.0 - (1.0 / SUN_HALF) * max(abs(xo), abs(zo)))) / 255.0;
    o.va = clamp(fade, 0.0, 1.0) * clamp(depth / 150.0, 0.0, 1.0);

    o.uv0 = vec2f(x * 0.0012 + u.sun1.z, z * 0.0012);
    o.uv1 = vec2f(x * 0.0012, u.sun1.w + z * 0.0015);
    return o;
}

@fragment
fn fs_sun(in: SunOut) -> @location(0) vec4f {
    // Low-frequency warp of the texture coordinates breaks up the obvious tiling lattice.
    let wp0 = in.uv0 + (vec2f(vnoise(in.uv0 * 3.1), vnoise(in.uv0 * 3.1 + 7.3)) - 0.5) * 0.45;
    let wp1 = in.uv1 + (vec2f(vnoise(in.uv1 * 2.7 + 3.9), vnoise(in.uv1 * 2.7 + 11.1)) - 0.5) * 0.45;
    let t0 = textureSample(wave_tex, wave_samp, wp0).r;
    let t1 = textureSample(wave_tex, wave_samp, wp1).r;

    // Depth test against the scene (LEQUAL): the overlay must not show through terrain.
    let d = textureLoad(scene_depth, vec2i(in.pos.xy), 0).r;
    if u.warp.w > 0.5 {
        if in.pos.z < d - 1e-6 {
            discard;
        }
    } else {
        if in.pos.z > d + 1e-6 {
            discard;
        }
    }

    if u.screen.z > 0.5 {
        return vec4f(0.1, 0.9, 0.3, 0.5); // debug: where the overlay exists
    }

    let a0 = clamp(t0 * in.va, 0.0, 1.0);
    let a = clamp(2.0 * t1 * a0, 0.0, 1.0);
    let a8 = a * 255.0;
    // GXSetAlphaCompare(GEQUAL 0x55, OR, LEQUAL 0x23), with the cut-off softened so the specks
    // have smooth edges instead of hard blocks.
    var vis = 1.0 - smoothstep(22.0, 40.0, a8);
    if a8 >= 85.0 {
        vis = 1.0;
    }
    // Far specks only alias into rows; fade them out with distance.
    vis = vis * (1.0 - smoothstep(1500.0, 5000.0, in.pos.w));
    // RASC * 2 clamps to white; the tint lets the colour be adjusted later.
    let colour = clamp(vec3f(200.0, 200.0, 255.0) / 255.0 * 2.0, vec3f(0.0), vec3f(1.0)) * vec3f(1.0);
    return vec4f(colour, a * vis * u.sun2.w);
}


// Sunshine's indirect "seaindirect" layer: the scene behind the water is re-sampled through a
// scrolling wobble texture. Drawn over the water surface before the foam overlay.
@fragment
fn fs_sea(in: SunOut) -> @location(0) vec4f {
    let d = textureLoad(scene_depth, vec2i(in.pos.xy), 0).r;
    if u.warp.w > 0.5 {
        if in.pos.z < d - 1e-6 {
            discard;
        }
    } else {
        if in.pos.z > d + 1e-6 {
            discard;
        }
    }
    if u.screen.z > 0.5 {
        return vec4f(0.1, 0.4, 0.9, 0.5);
    }
    let n0 = textureSample(wave_tex, wave_samp, in.uv0 * 4.0 + vec2f(0.0, u.sun1.w * 4.0)).r;
    let n1 = textureSample(wave_tex, wave_samp, in.uv1 * 4.0 + vec2f(u.sun1.z * 4.0, 0.0)).r;
    // Centre the offsets; the texture's mean intensity is low.
    let off = (vec2f(n0, n1) - vec2f(0.15)) * 2.0 * u.sun2.x;
    let size = u.screen.xy;
    let px = in.pos.xy + off * 14.0 * (size.y / 720.0);
    let uv = clamp(px / size, vec2f(0.001), vec2f(0.999));
    let c = textureSampleLevel(scene_color, samp, uv, 0.0).rgb;
    return vec4f(c, clamp(in.va * 4.0, 0.0, 1.0));
}
