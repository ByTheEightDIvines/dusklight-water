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
    ripple: vec4f,   // x swim ripple intensity (0 = Link is not in the water)
    glint: vec4f,    // xyz direction to the sun or moon (whichever is up), w visibility
    glint_col: vec4f, // rgb colour of that light
    spray: array<vec4f, 8>, // waterfall spray emitters: xyz foot position, w strength (0 = unused)
}

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> lattice: array<vec4f>;
@group(0) @binding(2) var scene_color: texture_2d<f32>;
@group(0) @binding(3) var scene_depth: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var wave_tex: texture_2d<f32>;
@group(0) @binding(6) var wave_samp: sampler;

const WAVE_COUNT_VERTEX: i32 = 5;
const WAVE_COUNT_FRAG: i32 = 8;
const BASE_AMPLITUDE: f32 = 14.0;
const WAVE_DECAY: f32 = 0.70;       // amplitude ratio between successive octaves   // world units (1 unit is about 1 cm), scaled by params.y
const BASE_WAVELENGTH: f32 = 900.0;
const SWELL_SCALE: f32 = 0.72;      // fraction of Sunshine's swell amplitude used by the custom surface
const SURFACE_LIFT: f32 = 0.0;
const COVER_SLOPE: f32 = 0.012;     // extra cover per unit of camera distance
const COVER_LIFT: f32 = 12.0;       // depth is written as if the surface were at least this far above the stock water plane      // keeps the surface just above the stock water plane

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) world: vec3f,
}

// Level 0 = fine lattice (near the camera), level 1 = coarse lattice (far reach). Both live in
// one storage buffer: fine cells first, then coarse cells.
const kStepLimit: f32 = 40.0;      // largest height difference blended or spanned by one surface quad

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
    // Water at very different heights (a pool below a waterfall and the ledge above it) must never
    // be blended into a ramp: only corners near the height of the heaviest wet corner count.
    var w_ref = 0.0;
    var h_ref = 0.0;
    if s00.z > 0.5 && w00 > w_ref { w_ref = w00; h_ref = s00.x; }
    if s10.z > 0.5 && w10 > w_ref { w_ref = w10; h_ref = s10.x; }
    if s01.z > 0.5 && w01 > w_ref { w_ref = w01; h_ref = s01.x; }
    if s11.z > 0.5 && w11 > w_ref { w_ref = w11; h_ref = s11.x; }
    let a00 = select(0.0, w00, s00.z > 0.5 && abs(s00.x - h_ref) < kStepLimit);
    let a10 = select(0.0, w10, s10.z > 0.5 && abs(s10.x - h_ref) < kStepLimit);
    let a01 = select(0.0, w01, s01.z > 0.5 && abs(s01.x - h_ref) < kStepLimit);
    let a11 = select(0.0, w11, s11.z > 0.5 && abs(s11.x - h_ref) < kStepLimit);
    let sum = a00 + a10 + a01 + a11;
    if sum < 1e-4 {
        return vec4f(0.0);
    }
    let h = (s00.x * a00 + s10.x * a10 + s01.x * a01 + s11.x * a11) / sum;
    let d = (s00.y * a00 + s10.y * a10 + s01.y * a01 + s11.y * a11) / sum;
    let o00 = select(0.0, w00, s00.z > 1.5 && a00 > 0.0);
    let o10 = select(0.0, w10, s10.z > 1.5 && a10 > 0.0);
    let o01 = select(0.0, w01, s01.z > 1.5 && a01 > 0.0);
    let o11 = select(0.0, w11, s11.z > 1.5 && a11 > 0.0);
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
        let amp = BASE_AMPLITUDE * pow(WAVE_DECAY, fi) * lod;
        let th = k * dot(dir, p) - omega * t + fi * 1.7;
        let e = exp(sin(th) - 1.0);
        h = h + amp * (e - 0.466);
        let d = amp * e * cos(th) * k;
        dx = dx + d * dir.x;
        dz = dz + d * dir.y;
    }
    return vec3f(h, dx, dz);
}

// Sun / moon glints: tiny facets of the surface, each tilted a little differently and turning over
// time, catch the light for an instant when their reflection lines up with the light. Three bands
// at growing world scale keep the glints a similar size on screen from near to the horizon.
fn glint_band(p: vec2f, g: f32, t: f32, R: vec3f, L: vec3f, seed: f32) -> f32 {
    // Rotated grid, jittered centres and varied sizes: no visible lattice of dots.
    let cs = cos(0.5 + seed * 0.9);
    let sn = sin(0.5 + seed * 0.9);
    let q = vec2f(p.x * cs - p.y * sn, p.x * sn + p.y * cs) / g;
    let id = floor(q) + vec2f(seed * 17.0, seed * 31.0);
    let jit = (vec2f(hash21(id + 1.3), hash21(id + 5.7)) - vec2f(0.5)) * 0.6;
    let f = fract(q) - vec2f(0.5) - jit;
    let h = hash21(id);
    let a = hash21(id * 1.7 + 3.1) * 6.2832 + t * (0.7 + h * 1.6);
    // Small random facet tilt only: where the glints appear is decided by the swell's own slope.
    let tilt = vec3f(cos(a), 0.0, sin(a)) * (0.01 + 0.07 * hash21(id + 9.7));
    let Rf = normalize(R + tilt);
    let along = max(dot(Rf, L), 0.0);
    let spec = pow(along, 150.0) * step(0.30, h);
    let rad = 0.12 + 0.24 * hash21(id + 8.1);
    let shape = smoothstep(rad, rad * 0.25, length(f));
    let tw = 0.55 + 0.45 * sin(t * (2.0 + h * 5.0) + h * 40.0);
    return spec * shape * tw;
}

fn glints(p: vec2f, dist: f32, t: f32, R: vec3f, L: vec3f) -> f32 {
    let w1 = 1.0 - smoothstep(1200.0, 2600.0, dist);
    let w3 = smoothstep(5000.0, 9000.0, dist);
    let w2 = clamp(1.0 - w1 - w3, 0.0, 1.0);
    var s = 0.0;
    if w1 > 0.001 {
        s = s + w1 * glint_band(p, 9.0, t, R, L, 0.0);
    }
    if w2 > 0.001 {
        s = s + w2 * glint_band(p, 40.0, t, R, L, 1.0);
    }
    if w3 > 0.001 {
        s = s + w3 * glint_band(p, 180.0, t, R, L, 2.0);
    }
    return s;
}

// Nearest-cell lattice read: (height, is wet).
fn cell_h(xz: vec2f) -> vec2f {
    var c: vec4f;
    if use_fine(xz) {
        c = cell_at(0, i32(floor(xz.x / u.lat.w) - u.lat.x), i32(floor(xz.y / u.lat.w) - u.lat.y));
    } else {
        c = cell_at(1, i32(floor(xz.x / u.lat2.w) - u.lat2.x), i32(floor(xz.y / u.lat2.w) - u.lat2.y));
    }
    return vec2f(c.x, select(0.0, 1.0, c.z > 0.5));
}

// Waterfall field: 0 far from any drop, rising toward 1 at the foot of a fall. A waterfall shows up
// in the lattice as a ledge of water much higher than the pool below it; the field is a smooth
// weighted count of such ledge cells around the point, so its contours are rings about the foot.
fn fall_field(xz: vec2f, y: f32) -> f32 {
    let jit = hash21(floor(xz * 0.25)) * 1.047198; // dithers the cell-sized steps of the lattice
    var acc = 0.0;
    var tot = 0.0;
    for (var r = 0u; r < 3u; r = r + 1u) {
        let rad = 55.0 + 50.0 * f32(r);
        let wgt = 1.0 - 0.35 * f32(r);
        for (var k = 0u; k < 6u; k = k + 1u) {
            let ang = f32(k) * 1.047198 + jit;
            let hc = cell_h(xz + vec2f(cos(ang), sin(ang)) * rad);
            if hc.y > 0.5 && hc.x > y + kStepLimit {
                acc = acc + wgt;
            }
            tot = tot + wgt;
        }
    }
    return acc / tot;
}

fn amplitude_scale(open: f32, depth: f32) -> f32 {
    return u.params.y * open * smoothstep(40.0, 320.0, depth);
}

const MIN_DEPTH: f32 = 2.0;        // water shallower than this is left to the stock water

fn usable(info: vec4f) -> bool {
    return info.w >= 1e-4 && info.y >= MIN_DEPTH;
}

fn grid_xz(gx: u32, gz: u32) -> vec2f {
    let half = f32(u32(u.grid.w)) * 0.5;
    let gu = f32(gx) - half;
    let gv = f32(gz) - half;
    // Dense near the camera, stretching toward the horizon.
    return vec2f(
        u.eye.x + sign(gu) * (u.warp.x * abs(gu) + u.warp.y * gu * gu),
        u.eye.z + sign(gv) * (u.warp.x * abs(gv) + u.warp.y * gv * gv));
}

// Culling is decided per quad, never per vertex: a single vertex thrown outside the clip volume
// drags its triangles into huge clipped slivers (the grey plane). Corners without water borrow the
// height of the quad's water corners; the fragment shader's mask removes the dry part.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    var o: VOut;
    let n = u32(u.grid.w);
    let quad = vi / 6u;
    let corner = vi % 6u;
    var cx = array<u32, 6>(0u, 1u, 0u, 1u, 1u, 0u);
    var cz = array<u32, 6>(0u, 0u, 1u, 0u, 1u, 1u);
    let qx = quad % n;
    let qz = quad / n;
    let own = grid_xz(qx + cx[corner], qz + cz[corner]);

    var hsum = 0.0;
    var count = 0.0;
    var hmin = 1e9;
    var hmax = -1e9;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let p = grid_xz(qx + (k & 1u), qz + (k >> 1u));
        let ci = surface_info(p);
        if usable(ci) {
            hsum = hsum + ci.x;
            count = count + 1.0;
            hmin = min(hmin, ci.x);
            hmax = max(hmax, ci.x);
        }
    }
    // A quad spanning a step (waterfall face, ledge) is dropped so no sheet slopes across it.
    if count < 0.5 {
        o.pos = vec4f(2.0, 2.0, 2.0, 1.0); // whole quad dry: all six vertices collapse together
        o.world = vec3f(0.0);
        return o;
    }
    let info = surface_info(own);
    var base = hsum / count;
    var scale = 0.0;
    let stepped = hmax - hmin > kStepLimit;
    if usable(info) {
        base = info.x;
        scale = amplitude_scale(info.z, info.y);
    }
    // A quad spanning a step (waterfall face, ledge) is flattened to the lower pool; the fragment
    // shader throws away the part that belongs to the higher ledge.
    if stepped {
        base = hmin;
        scale = 0.0;
    }
    let vdist = length(own - u.eye.xz);
    let w = waves(own, u.params.x, WAVE_COUNT_VERTEX, vdist);
    let world = vec3f(own.x, base + SURFACE_LIFT + w.x * scale, own.y);
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
    // Quintic interpolation: no visible cell grid, so foam edges stay round instead of blocky.
    let s = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
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
    // Each octave is rotated so the value-noise axes never line up.
    let rot = mat2x2f(0.8, -0.6, 0.6, 0.8);
    for (var i = 0; i < 4; i = i + 1) {
        v = v + a * vnoise(q);
        q = rot * q * 2.03 + vec2f(17.1, 9.2);
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
    // Smooth (bilinear) openness among the wet neighbours instead of the nearest cell, so the edge
    // of the replaced water follows smooth contours instead of a staircase of lattice cells.
    // Cells that are wet but not open are still water (ponds, puddles): they get a calm, murky look.
    if info.x > world.y + 28.0 {
        discard; // belongs to a higher ledge: the quad was flattened to the pool below
    }
    let open_frac = info.z / max(info.w, 1e-4);
    let stagnant = open_frac < 0.5;
    // Only about half a lattice cell past the last wet cell is covered; the true shoreline is cut
    // by the terrain test below.
    if info.w < 0.1 || info.y < MIN_DEPTH {
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
    // One slowly drifting density field drives both the glints and the ripple detail, so the
    // surface is busier (more glints, finer ripples) in the same patches.
    let dens = smoothstep(0.30, 0.78, vnoise(world.xz * 0.0016 + vec2f(t * 0.012, -t * 0.008)));
    let ripple_fade = 1.0 - smoothstep(300.0, 2500.0, dist);
    let ripple = (vnoise(world.xz * 0.05 + vec2f(t * 0.4, -t * 0.3)) - 0.5) * ripple_fade * (0.75 + 0.6 * dens);
    let ripple2 = (vnoise(world.xz * 0.11 - vec2f(t * 0.5, t * 0.2)) - 0.5) * ripple_fade * (0.75 + 0.6 * dens);
    var fall = 0.0;
    if u.eye.y >= world.y && !stagnant {
        fall = smoothstep(0.08, 0.85, fall_field(world.xz, info.x) * 1.8);
    }
    let hi_fade = (1.0 - smoothstep(250.0, 1400.0, dist)) * (0.45 + 1.25 * dens);
    let ripple3 = (vnoise(world.xz * 0.23 + vec2f(-t * 0.7, t * 0.5)) - 0.5) * hi_fade;
    let ripple4 = (vnoise(world.xz * 0.47 + vec2f(t * 0.6, t * 0.8)) - 0.5) * hi_fade;
    // N: full detail (glint, refraction). N_low: smooth normal for reflections and Fresnel.
    let calm = select(1.0, 0.6, stagnant);
    var N = normalize(vec3f(-w.y * strength - (ripple + ripple3 * 1.1) * 0.05 * calm, 1.0, -w.z * strength - (ripple2 + ripple4 * 1.1) * 0.05 * calm));
    var N_low = normalize(vec3f(-w_low.y * strength, 1.0, -w_low.z * strength));
    let N_w = normalize(vec3f(-w.y * strength, 1.0, -w.z * strength)); // swell only, for glints

    // Waterfall foot: ripples spreading outward as ring contours of the fall field.
    let fall_ring = sin(fall * 14.0 - t * 3.6 + vnoise(world.xz * 0.03) * 2.0);
    N = normalize(vec3f(N.x + fall_ring * 0.10 * fall, N.y, N.z + fall_ring * 0.07 * fall));
    N_low = normalize(vec3f(N_low.x + fall_ring * 0.05 * fall, N_low.y, N_low.z + fall_ring * 0.04 * fall));

    // Footsteps and swimming: a turbulent swirl around Link, like the stock spring ripples. It
    // warps the reflection normal and what is seen through the water.
    let sw_r = length(world.xz - u.sun0.xy);
    let sw_env = select(1.0, 0.0, stagnant) * u.ripple.x * (1.0 - smoothstep(40.0, 380.0, sw_r)) *
                 (0.7 + 0.3 * sin(sw_r * 0.035 - t * 3.0));
    let sw = vec2f(fbm(world.xz * 0.035 + vec2f(t * 0.35, 0.0)) - 0.5,
                   fbm(world.xz * 0.035 + vec2f(5.2, -t * 0.3)) - 0.5) * sw_env;
    N = normalize(vec3f(N.x + sw.x * 1.4, N.y, N.z + sw.y * 1.4));
    N_low = normalize(vec3f(N_low.x + sw.x, N_low.y, N_low.z + sw.y));

    let uv = frag.xy / u.screen.xy;
    let size = vec2i(u.screen.xy);

    // Scene beneath the surface.
    let d0 = textureLoad(scene_depth, vec2i(frag.xy), 0).r;
    let scene0 = unproject(uv, d0);
    // Terrain above the surface (a beach, a rock face) is not water: without this the cover that
    // hides the stock plane draws the water as a glassy lip over the shore.
    var terrain_above = false;
    if u.eye.y >= world.y && scene0.y > world.y + 1.0 + dist * 0.002 && length(scene0 - eye) < 60000.0 {
        if scene0.y > world.y + 40.0 {
            discard;
        }
        terrain_above = true;
    }
    if terrain_above {
        // Shore band: show the scene as it is, but still claim the depth so the stock water's own
        // rim cannot draw a dark line over it.
        return vec4f(textureSampleLevel(scene_color, samp, uv, 0.0).rgb, 1.0);
    }
    var thick0 = max(world.y - scene0.y, 0.0);
    if length(scene0 - eye) > 60000.0 {
        thick0 = 2000.0;
    }

    // Refraction: offset by the normal, but never pull in something in front of the water.
    // Mostly the smooth swell tilts what is seen below; fine ripples only add a hint, so the
    // lakebed does not shimmer and warp.
    var uv_r = uv + (N_low.xz * 0.022 + (N.xz - N_low.xz) * 0.006) * clamp(thick0 / 120.0, 0.0, 1.0) * u.screen.w +
               sw * 0.14 * clamp(thick0 / 40.0, 0.25, 1.0) * u.screen.w;
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
    if stagnant {
        shallow_col = mix(shallow_col, vec3f(0.10, 0.26, 0.18), 0.12);
        deep_tint = mix(deep_tint, vec3f(0.03, 0.09, 0.06), 0.12);
    }
    let deep_col = (mix(shallow_col, deep_tint, smoothstep(40.0, 700.0, thick)) +
                    u.horizon.rgb * 0.06) * (0.45 + 0.85 * lum);
    let absorb = vec3f(0.0120, 0.0046, 0.0030) * select(1.0, 1.15, stagnant) / max(u.screen.w, 0.1);
    let trans = exp(-absorb * thick);
    let body = under * trans + deep_col * (vec3f(1.0) - trans);

    // Reflection.
    let cosv = clamp(dot(N_low, V), 0.0, 1.0);
    let fres = 0.02 + 0.98 * pow(1.0 - cosv, 5.0);
    let R_low = reflect(-V, N_low);
    let R = reflect(-V, N);
    let sky = sky_color(vec3f(R_low.x, abs(R_low.y), R_low.z));
    let refl = reflect_ray(world, R_low, sky);

    let shore_calm = mix(0.2, 1.0, smoothstep(4.0, 70.0, thick));
    var col = mix(body, refl, clamp(fres * 0.85 * shore_calm, 0.0, 0.85));

    // Sun glint.
    let sunv = u.sun.xyz;
    let spec = pow(max(dot(R, sunv), 0.0), 600.0) * 3.0 + pow(max(dot(R, sunv), 0.0), 60.0) * 0.15;
    col = col + vec3f(1.0, 0.95, 0.85) * spec * u.sun.w * lum;

    // Water tapers to nothing at the shoreline: the thinner it is, the more of the scene shows.
    col = mix(under, col, smoothstep(0.0, 24.0, thick));

    // Foam: shoreline (thin water) and wave crests.
    let fn1 = fbm(world.xz * 0.045 + vec2f(t * 0.05, t * 0.03));
    let fn2 = fbm(world.xz * 0.11 - vec2f(t * 0.04, -t * 0.06));
    // A lapping line at the waterline: a bright thin edge plus a broken wash that surges in and out.
    let surge = 12.0 + 7.0 * sin(t * 0.9 + fn1 * 5.0);
    let wash = 1.0 - smoothstep(0.0, surge, thick);
    let edge = 1.0 - smoothstep(0.0, 5.0, thick);
    let breakup = smoothstep(0.30, 0.62, fn2 * 0.6 + fn1 * 0.5);
    var foam = clamp(wash * breakup * 0.85 + edge * 0.7, 0.0, 1.0);

    // Shoreline foam only: it follows the Foam Intensity setting and is never drawn when looking
    // up at the water from underneath.
    foam = foam * clamp(u.sun2.w, 0.0, 2.0);
    if u.eye.y < world.y {
        foam = 0.0;
    }
    foam = clamp(foam, 0.0, 1.0);
    // Foam must read as white against whatever is under it, including sunlit sand that is brighter
    // than the ambient-scaled foam colour: it is never darker than the scene plus a little.
    let foam_col = clamp(max(vec3f(0.92, 0.96, 0.98) * (0.60 + 0.40 * lum), under * 1.08 + vec3f(0.10)), vec3f(0.0), vec3f(1.0));
    col = mix(col, foam_col, foam);

    // Waterfall foot: churning white splash, with brighter rings where the ripples crest.
    if fall > 0.001 {
        let churn = smoothstep(0.55, 0.85, fbm(world.xz * 0.16 + vec2f(t * 1.1, -t * 0.8)) * 0.7 +
                                            fbm(world.xz * 0.41 - vec2f(t * 1.6, t * 0.6)) * 0.45);
        let crest = smoothstep(0.88, 1.0, fall_ring) * 0.35;
        let splash = clamp(fall * (0.10 + 0.90 * churn) + fall * crest, 0.0, 1.0);
        col = mix(col, clamp(max(vec3f(0.94, 0.97, 0.99) * (0.60 + 0.40 * lum), under * 1.08 + vec3f(0.10)), vec3f(0.0), vec3f(1.0)), splash * 0.5);
    }

    // Wave texture: thin pale contour lines that trace the swell and stretch along the crests, as
    // in Sunshine's sea. They are contours of the actual wave height, wandered by a slow noise so
    // they break up and drift; the same density field as the glints makes them busier in patches.
    if u.eye.y >= world.y {
        // Contours of a noise field stretched along x: long, thin, sinuous lines that mostly run
        // across the view, drifting slowly, instead of closed swirls.
        let q = vec2f(world.x * 0.0011 + t * 0.012, world.z * 0.0105 - t * 0.02);
        let nf = vnoise(q) * 1.0 + vnoise(q * 2.3 + vec2f(7.0, 3.0)) * 0.5 + w.x * 0.012;
        let c1 = fract(nf * 5.0);
        let l1 = 1.0 - smoothstep(0.0, 0.06, min(c1, 1.0 - c1));
        let c2 = fract(nf * 11.0 + 0.37);
        let l2 = 1.0 - smoothstep(0.0, 0.05, min(c2, 1.0 - c2));
        let breakup2 = smoothstep(0.25, 0.60, vnoise(world.xz * 0.006 + vec2f(-t * 0.05, t * 0.03)));
        let lines = (l1 * 0.8 + l2 * (0.15 + 0.7 * dens)) * breakup2;
        let line_fade = (1.0 - smoothstep(2500.0, 7000.0, dist)) * smoothstep(20.0, 120.0, info.y) * smoothstep(8.0, 45.0, thick);
        // Pale highlights added to the water, not a darker tint: they read as light on the surface.
        col = col + vec3f(0.55, 0.60, 0.55) * (0.25 + 0.75 * lum) * clamp(lines * line_fade * (0.20 + 0.22 * dens), 0.0, 0.5);
    }

    // Distance fog toward the scene fog colour.
    let fog = 1.0 - exp(-dist * u.params.w);
    col = mix(col, u.horizon.rgb, clamp(fog, 0.0, 1.0));

    // Sparkle: animated glints where the sun or moon catches the facets of the water.
    if u.eye.y >= world.y {
        let Rg = reflect(-V, N_w);
        let g = glints(world.xz, dist, t, Rg, u.glint.xyz);
        col = col + u.glint_col.rgb * g * 6.0 * smoothstep(4.0, 30.0, thick) * (0.4 + 1.4 * dens) * u.glint.w * clamp(u.sun2.w, 0.0, 2.0) *
              (1.0 - 0.5 * clamp(fog, 0.0, 1.0));
    }

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
    let base_y = surface_info(world.xz).x;
    let cover = COVER_LIFT + length(world - u.eye.xyz) * COVER_SLOPE;
    // Seen from above the cover sits over the stock plane; seen from below (swimming, looking up)
    // it sits under it, so the stock water's scrolling texture cannot show through either way.
    let above = u.eye.y >= base_y;
    let plane_y = select(base_y - cover, base_y + cover, above);
    let behind_plane = select(world.y > plane_y, world.y < plane_y, above);
    if behind_plane {
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
    @location(3) wp: vec2f,
    @location(4) crest: f32,
}

fn sun_xz(gx: u32, gz: u32) -> vec2f {
    return vec2f(-SUN_HALF + f32(gx) * SUN_CELL, -SUN_HALF + f32(gz) * SUN_CELL);
}

@vertex
fn vs_sun(@builtin(vertex_index) vi: u32) -> SunOut {
    var o: SunOut;
    let quad = vi / 6u;
    let corner = vi % 6u;
    var cx = array<u32, 6>(0u, 1u, 0u, 1u, 1u, 0u);
    var cz = array<u32, 6>(0u, 0u, 1u, 0u, 1u, 1u);
    let qx = quad % SUN_COLS;
    let qz = quad / SUN_COLS;
    let off = sun_xz(qx + cx[corner], qz + cz[corner]);
    let xo = off.x;
    let zo = off.y;
    let x = u.sun0.x + xo;
    let z = u.sun0.y + zo;

    // Per-quad culling (see vs_main).
    var hsum = 0.0;
    var count = 0.0;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let po = sun_xz(qx + (k & 1u), qz + (k >> 1u));
        let ci = surface_info(u.sun0.xy + po);
        if usable(ci) {
            hsum = hsum + ci.x;
            count = count + 1.0;
        }
    }
    if count < 0.5 {
        o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
        o.uv0 = vec2f(0.0);
        o.uv1 = vec2f(0.0);
        o.va = 0.0;
        o.wp = vec2f(0.0);
        o.crest = 0.0;
        return o;
    }
    let info = surface_info(vec2f(x, z));
    var base = hsum / count;
    var depth = 0.0;
    var open = 0.0;
    if usable(info) {
        base = info.x;
        depth = info.y;
        open = 1.0;
    }
    // TMapObjWave::updateHeightAndAlpha: swell and alpha shrink toward the shore.
    let amp = clamp(depth / 400.0, 0.0, 1.0);
    let a1 = u.sun1.x * amp;
    let a2 = u.sun1.y * amp;
    let h = a1 * sin(0.02 * (x * (1.0 / 6.28318)) + u.sun0.z)
          + a2 * sin(0.03 * (z * (1.0 / 6.28318)) + u.sun0.w);
    let world = vec3f(x, base + h * u.params.y, z);
    o.pos = u.proj_from_world * vec4f(world, 1.0);

    // TMapObjWave::getAlpha, then the shore ramp (unk54).
    let fade = floor(255.0 * (1.0 - (1.0 / SUN_HALF) * max(abs(xo), abs(zo)))) / 255.0;
    o.va = clamp(fade, 0.0, 1.0) * clamp(depth / 150.0, 0.0, 1.0) * open;
    o.wp = vec2f(x, z);
    // Foam collects on the swell crests rather than covering the whole sea evenly.
    o.crest = smoothstep(-0.25, 0.75, h / max(u.sun1.x + u.sun1.y, 1.0));
    o.uv0 = vec2f(x * 0.0012 + u.sun1.z, z * 0.0012);
    o.uv1 = vec2f(x * 0.0012, u.sun1.w + z * 0.0015);
    return o;
}

@fragment
fn fs_sun(in: SunOut) -> @location(0) vec4f {
    // Low-frequency warp of the texture coordinates breaks up the obvious tiling lattice.
    let wt = u.params.x * 0.02;
    let wp0 = in.uv0 + (vec2f(vnoise(in.uv0 * 3.1 + wt), vnoise(in.uv0 * 3.1 + 7.3 - wt)) - 0.5) * 0.45;
    let wp1 = in.uv1 + (vec2f(vnoise(in.uv1 * 2.7 + 3.9 - wt), vnoise(in.uv1 * 2.7 + 11.1 + wt)) - 0.5) * 0.45;
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
    // Patchy, drifting coverage (world-space noise moving at its own pace) on top of the crests,
    // so the specks are not a uniform sheet scrolling in one direction.
    let tm = u.params.x;
    let drift = smoothstep(0.42, 0.72, fbm(in.wp * 0.0011 + vec2f(tm * 0.011, -tm * 0.007)));
    vis = vis * in.crest * mix(0.08, 1.0, drift);
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


// ===============================================================================================
// Waterfall spray: camera-facing droplets thrown up from the foot of each fall. Each particle is
// derived from its index alone (a repeating ballistic arc), so no particle buffer is needed.
// ===============================================================================================
const SPRAY_N: u32 = 48u;

struct SprayOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) world: vec3f,
    @location(2) alpha: f32,
}

@vertex
fn vs_spray(@builtin(vertex_index) vi: u32) -> SprayOut {
    var o: SprayOut;
    o.uv = vec2f(0.0);
    o.world = vec3f(0.0);
    o.alpha = 0.0;
    o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
    let quad = vi / 6u;
    let corner = vi % 6u;
    let e = quad / SPRAY_N;
    let pi = quad % SPRAY_N;
    if e >= 8u {
        return o;
    }
    let em = u.spray[e];
    if em.w < 0.01 {
        return o;
    }
    let seed = f32(pi) * 1.37 + f32(e) * 57.3;
    let period = 1.1 + hash21(vec2f(seed, 1.0)) * 1.1;
    let ph = fract(u.params.x / period + hash21(vec2f(seed, 2.0)));
    let tt = ph * period;
    let ang = hash21(vec2f(seed, 3.0)) * 6.2832;
    let big = hash21(vec2f(seed, 4.0)) < 0.25;     // a quarter of the particles are slow mist
    let sp = select(30.0 + 90.0 * hash21(vec2f(seed, 5.0)), 15.0 + 35.0 * hash21(vec2f(seed, 5.0)), big);
    let vy = select(220.0 + 280.0 * hash21(vec2f(seed, 6.0)), 60.0 + 120.0 * hash21(vec2f(seed, 6.0)), big) * (0.6 + 0.4 * em.w);
    let jr = 50.0 * hash21(vec2f(seed, 7.0));
    let origin = em.xyz + vec3f(cos(ang) * jr, 0.0, sin(ang) * jr);
    let pos = origin + vec3f(cos(ang) * sp * tt, vy * tt - 0.5 * 700.0 * tt * tt, sin(ang) * sp * tt);
    if pos.y < em.y {
        return o;
    }
    let size = select(5.0 + 7.0 * hash21(vec2f(seed, 8.0)), 28.0 + 30.0 * hash21(vec2f(seed, 8.0)), big) * (0.7 + 0.6 * ph);
    var cx = array<f32, 6>(-1.0, 1.0, -1.0, 1.0, 1.0, -1.0);
    var cy = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    let to_eye = u.eye.xyz - pos;
    let right = normalize(cross(vec3f(0.0, 1.0, 0.0), to_eye));
    let cam_up = normalize(cross(to_eye, right));
    let wp = pos + (right * cx[corner] + cam_up * cy[corner]) * size;
    o.pos = u.proj_from_world * vec4f(wp, 1.0);
    o.uv = vec2f(cx[corner], cy[corner]);
    o.world = pos;
    o.alpha = em.w * sin(ph * 3.14159) * select(0.75, 0.22, big) *
              (1.0 - smoothstep(1800.0, 3000.0, length(to_eye)));
    return o;
}

@fragment
fn fs_spray(in: SprayOut) -> @location(0) vec4f {
    let a = in.alpha * (1.0 - smoothstep(0.25, 1.0, length(in.uv)));
    if a < 0.003 {
        discard;
    }
    // Hidden behind scene geometry?
    let uv = in.pos.xy / u.screen.xy;
    let d0 = textureLoad(scene_depth, vec2i(in.pos.xy), 0).r;
    let scene = unproject(uv, d0);
    if length(scene - u.eye.xyz) < length(in.world - u.eye.xyz) - 6.0 {
        discard;
    }
    let lum = clamp(luminance(u.amb.rgb) * 1.4, 0.12, 1.0);
    return vec4f(vec3f(0.94, 0.97, 1.0) * (0.45 + 0.55 * lum), a);
}
