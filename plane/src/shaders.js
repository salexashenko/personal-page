// WGSL for the zeta landscape. Grid constants live here so the shaders and
// the JavaScript that drives them cannot drift apart.

import { T_RS, SIEGEL_TERMS, siegelTables } from './siegel.js';

export const FW = 512;            // field columns (sigma)
export const FH = 2048;           // field rows kept in the ring (t)
export const DT = 1 / 32;         // t step, so the ring holds 64 units of t
export const MW = FW - 1;         // surface mesh quads across
export const MH = FH - 1;         // surface mesh quads along
export const NMAX = 1024;         // most series terms a point may use (it never needs more than about 230)
export const HEIGHT = 0.45;       // world height per unit of log|zeta|
export const EPS2 = 2.5e-5;       // floors log|zeta| near -5.3 so zeros are wells, not holes
export const LMAX = 7;            // soft ceiling on log|zeta| for the pole and the left-hand wall

// Columns are spaced finely near the critical line, where everything happens,
// and stretch out toward the sides: sigma = C + A sinh(B x), x in [-1, 1].
// That covers sigma from about -19.4 to 20.4 with 1/40 resolution at the line.
export const SIG_C = 0.5, SIG_A = 2.2, SIG_B = 2.9;
export function sigmaOfCol(c) {
  return SIG_C + SIG_A * Math.sinh(SIG_B * ((c / (FW - 1)) * 2 - 1));
}

export function colOfSigma(s) {
  return (Math.asinh((s - SIG_C) / SIG_A) / SIG_B * 0.5 + 0.5) * (FW - 1);
}
/** World height of the land where zeta = re + i im: capL(logMod(jetOf(F))) below, times HEIGHT. */
export function landHeight(re, im) {
  const a = Math.max(1, Math.abs(re), Math.abs(im));
  const L = Math.log(a) + 0.5 * Math.log((re / a) ** 2 + (im / a) ** 2 + EPS2);
  return HEIGHT * (L > 0 ? LMAX * Math.tanh(L / LMAX) : L);
}

const SIGMA_MAP = /* wgsl */ `
fn sigmaOfCol(c: f32) -> f32 {
  return ${SIG_C} + ${SIG_A} * sinh(${SIG_B} * ((c / ${FW - 1}.0) * 2.0 - 1.0));
}
fn colOfSigma(s: f32) -> f32 {
  return (asinh((s - ${SIG_C}) / ${SIG_A}) / ${SIG_B} * 0.5 + 0.5) * ${FW - 1}.0;
}
`;
const HORIZON = 'vec3f(0.011, 0.010, 0.036)';

const COMMON = /* wgsl */ `
const FW: u32 = ${FW}u;
const FH: u32 = ${FH}u;
const EPS2: f32 = ${EPS2};
const LMAX: f32 = f32(${LMAX});
const PI: f32 = 3.14159265358979;

struct Globals {
  viewProj: mat4x4f,
  cam: vec4f,    // eye xyz, time
  grid: vec4f,   // unused, pixels per world unit at w = 1, dt, heightScale
  win: vec4f,    // validLo, validHi, meshRow0 (rows relative to origin), seconds this step
  flow: vec4f,   // swirlCos, swirlSin, flow step, zShift
  flow2: vec4f,  // vmax step, life rate * dt, streak, particle px at w=1
  res: vec4f,    // width, height, particle gain, grid gain
  aux: vec4f,    // unused, unused, 1 while the field is filling, unused
  misc: vec4u,   // origin row mod FH, frame, particle count, flags (1 = respawn all)
  me: vec4f,     // where the camera is looking: sigma, z, unused, unused
  me2: vec4f,    // how many ball slots are in use, unused, which way each mark's eddy turns (bits), unused
  hits: array<vec4f, 8>,   // where balls have landed: sigma, z, age in seconds, strength (0 = none)
};

fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn cdiv(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b);
}

// Cyclic palette: amber, violet, azure, green.
fn palette(phase: f32) -> vec3f {
  let h = phase / (2.0 * PI);
  return vec3f(0.55, 0.45, 0.60) + vec3f(0.45, 0.40, 0.40) * cos(2.0 * PI * (h + vec3f(0.0, 0.25, 0.55)));
}
const GOLD: vec3f = vec3f(1.0, 0.74, 0.36);
const HORIZON: vec3f = ${HORIZON};
fn haze(dist: f32) -> f32 { return exp(-pow(dist * 0.026, 2.0)); }
fn sides(sigma: f32) -> f32 { return smoothstep(19.0, 14.0, abs(sigma - 0.5)); }
${SIGMA_MAP}

// zeta and zeta' rescaled together so squaring never overflows.
struct Jet { z: vec2f, zp: vec2f, la: f32 };
fn jetOf(F: vec4f) -> Jet {
  let a = max(1.0, max(abs(F.x), abs(F.y)));
  return Jet(F.xy / a, F.zw / a, log(a));
}
fn logMod(j: Jet) -> f32 { return j.la + 0.5 * log(dot(j.z, j.z) + EPS2); }
fn capL(L: f32) -> f32 { return select(L, LMAX * tanh(L / LMAX), L > 0.0); }
// A ball landing sends a ring outward across the surface. x: how far it lifts the
// surface here, y: how brightly it glows.
fn rippleAt(p: vec2f) -> vec2f {
  var acc = vec2f(0.0);
  for (var i = 0; i < 8; i++) {
    let h = G.hits[i];
    if (h.w <= 0.0) { continue; }
    let x = (length(p - h.xy) - 2.6 * h.z) / 0.45;
    let live = exp(-h.z * 0.55) * h.w;
    // The swell starts flat under your feet and grows as it leaves; the glow is a thin line on its crest.
    acc += vec2f(0.1 * exp(-x * x) * cos(x * 2.4) * smoothstep(0.0, 0.5, h.z), exp(-60.0 * x * x) + 0.07 * exp(-3.0 * x * x)) * live;
  }
  return acc;
}
// Colour is the argument of zeta * zeta'. It winds once around every zero,
// and keeps turning across the open plain where arg(zeta) alone sits still.
fn hueOf(j: Jet) -> f32 {
  let w = vec2f(j.z.x * j.zp.x - j.z.y * j.zp.y, j.z.x * j.zp.y + j.z.y * j.zp.x);
  return atan2(w.y, w.x);
}
`;

export const NB = 48;             // balls that can be rolling at once
export const TRAIL = 128;         // trail samples kept per ball
const TRAIL_EVERY = 0.04;         // seconds between samples

const BALL_STRUCT = /* wgsl */ `
struct Ball {
  p: vec4f,   // sigma, z, height of its lowest point, state (0 none, 1 falling, 2 rolling, 3 going down a zero)
  v: vec4f,   // velocity in sigma, in z, and vertically; seconds alive
  q: vec4f,   // seconds since the last trail sample, newest trail slot, seconds nearly still, how far gone (0..1)
  c: vec4f,   // hue, radius, slope of the ground under it (sigma, z)
  n: vec4f,   // for the music: which sector of arg zeta it is in, sector lines crossed so far, which band of log|zeta|, band lines crossed
};
const NB: u32 = ${NB}u;
const TRAIL: u32 = ${TRAIL}u;
`;

const FIELD_SAMPLE = /* wgsl */ `
fn ringRow(j: i32) -> i32 { return i32((u32(j) + G.misc.x) % FH); }

// Bilinear zeta (xy) and zeta' (zw) at sigma, and z = t minus the window origin.
fn fieldAt(sigma: f32, z: f32) -> vec4f {
  let u = clamp(colOfSigma(sigma), 0.0, f32(FW) - 1.001);
  let v = max(z / G.grid.z, 0.0);
  let i = i32(floor(u));
  let j = i32(floor(v));
  let f = vec2f(u - floor(u), v - floor(v));
  let r0 = ringRow(j);
  let r1 = ringRow(j + 1);
  let a = textureLoad(field, vec2i(i, r0), 0);
  let b = textureLoad(field, vec2i(i + 1, r0), 0);
  let c = textureLoad(field, vec2i(i, r1), 0);
  let d = textureLoad(field, vec2i(i + 1, r1), 0);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
`;

// ---------------------------------------------------------------------------
// Field: zeta and zeta', one invocation per point. Low down by Euler–Maclaurin
// summation, which needs about t/2 terms; from T_RS up by the Riemann–Siegel
// formula (see siegel.js), which needs sqrt(t / 2 pi).
// The phases t*ln(n) mod 2pi are prepared in double precision on the CPU for
// the first row of each batch; the shader only adds a small offset, so single
// precision holds up at any height.
// ---------------------------------------------------------------------------
const SIEGEL_TABLE = siegelTables()
  .flat()
  .map((c) => `vec2f(${c.map((x) => (Math.abs(x) < 1e-12 ? 0 : x).toExponential(9)).join(', ')})`)
  .join(',\n  ');

export const FIELD_WGSL = /* wgsl */ `
struct FieldU {
  rowTex: u32, rows: u32,
  N: u32,        // how many entries of the table are filled
  aInt: u32,     // floor(sqrt(tRef/2pi))
  dt: f32, tRef: f32,
  phiRef: f32,   // tRef ln(tRef/2pi) - tRef, mod 4pi, from the CPU in double precision
  lnTau: f32,    // ln(tRef/2pi)
  aFrac: f32,    // sqrt(tRef/2pi) - aInt
};
@group(0) @binding(0) var<uniform> U: FieldU;
@group(0) @binding(1) var<storage, read> tbl: array<vec2f>;   // (phase at tRef, ln n)
@group(0) @binding(2) var field: texture_storage_2d<rgba32float, write>;

fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn cdiv(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b);
}
fn cinv(b: vec2f) -> vec2f { return vec2f(b.x, -b.y) / dot(b, b); }

${SIGMA_MAP}
fn cexp(a: vec2f) -> vec2f { return exp(a.x) * vec2f(cos(a.y), sin(a.y)); }
fn clog(a: vec2f) -> vec2f { return vec2f(0.5 * log(dot(a, a)), atan2(a.y, a.x)); }

struct Pair { z: vec2f, zp: vec2f };   // zeta and zeta'

// zeta(sE + i sg t) by Euler–Maclaurin: U.N terms, then the Bernoulli tail.
fn eulerMaclaurin(sE: f32, sg: f32, t: f32, dtt: f32) -> Pair {
  let N = U.N;
  var S = vec2f(0.0);
  var Sp = vec2f(0.0);
  for (var n = 1u; n < N; n++) {
    let e = tbl[n];
    let th = e.x + dtt * e.y;
    let term = exp(-sE * e.y) * vec2f(cos(th), -sg * sin(th));
    S += term;
    Sp -= e.y * term;
  }

  let eN = tbl[N];
  let lnN = eN.y;
  let Nf = f32(N);
  let thN = eN.x + dtt * lnN;
  let pN = exp(-sE * lnN) * vec2f(cos(thN), -sg * sin(thN));   // N^-s
  let s = vec2f(sE, sg * t);
  let sm1 = s - vec2f(1.0, 0.0);
  let T0 = Nf * cdiv(pN, sm1);                                 // N^(1-s)/(s-1)

  var z = S + T0 + 0.5 * pN;
  var zp = Sp - lnN * T0 - cdiv(T0, sm1) - 0.5 * lnN * pN;

  // Bernoulli tail: B_2k/(2k)! * s(s+1)...(s+2k-2) * N^(-s-2k+1)
  var C = array<f32, 6>(
    0.083333333333, -0.0013888888889, 3.3068783069e-5,
    -8.2671957672e-7, 2.0876756988e-8, -5.2841901387e-10);
  var P = cmul(s, pN) / Nf;
  var D = cinv(s);
  let inv2 = 1.0 / (Nf * Nf);
  for (var k = 0u; k < 6u; k++) {
    z += C[k] * P;
    zp += C[k] * cmul(P, D - vec2f(lnN, 0.0));
    let a = s + vec2f(f32(2u * k + 1u), 0.0);
    let b = s + vec2f(f32(2u * k + 2u), 0.0);
    P = cmul(cmul(P, a), b) * inv2;
    D += cinv(a) + cinv(b);
  }
  return Pair(z, zp);
}

// chi(s) = 2^s pi^(s-1) sin(pi s/2) Gamma(1-s), the factor in zeta(s) = chi(s) zeta(1-s),
// kept as k e^ln so its size can be held in check, with chi'/chi beside it.
struct Chi { ln: vec2f, k: vec2f, dlog: vec2f };

// Low down: Lanczos for Gamma(1-s) and its logarithmic derivative.
fn chiLow(sigma: f32, t: f32) -> Chi {
  let x = vec2f(1.0 - sigma, -t);
  var cof = array<f32, 6>(
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 1.208650973866179e-3, -5.395239384953e-6);
  var ser = vec2f(1.000000000190015, 0.0);
  var dser = vec2f(0.0);
  for (var q = 0u; q < 6u; q++) {
    let inv = cinv(x + vec2f(f32(q) + 1.0, 0.0));
    ser += cof[q] * inv;
    dser -= cof[q] * cmul(inv, inv);
  }
  let w = x + vec2f(5.5, 0.0);
  let lw = clog(w);
  let half = vec2f(0.5, 0.0);
  let lnGamma = cmul(x + half, lw) - w + clog(2.5066282746310005 * cdiv(ser, x));
  let digamma = lw + cdiv(x + half, w) - vec2f(1.0, 0.0) + cdiv(dser, ser) - cinv(x);
  // sin(pi s/2) grows like e^(pi|t|/2) and Gamma(1-s) shrinks as fast, so
  // that factor is moved into the exponent where the two cancel.
  let a = 1.5707963267948966 * sigma;
  let b = 1.5707963267948966 * t;
  let e2 = exp(-2.0 * abs(b));
  let sn = 0.5 * vec2f(sin(a) * (1.0 + e2), sign(b) * cos(a) * (1.0 - e2));
  let tb = sign(b) * (1.0 - e2) / (1.0 + e2);
  let cot = cdiv(vec2f(cos(a), -sin(a) * tb), vec2f(sin(a), cos(a) * tb));
  let ex = vec2f(sigma * 0.6931471805599453 + (sigma - 1.0) * 1.1447298858494002 + abs(b), t * 1.8378770664093453) + lnGamma;
  return Chi(ex, sn, vec2f(1.8378770664093453, 0.0) + 1.5707963267948966 * cot - digamma);
}

// t ln(t/2pi) - t mod 4pi: enormous, so it arrives from the CPU already
// reduced at the first row of the batch and is carried up from there.
fn bigPhase(dtt: f32) -> f32 {
  let r = dtt / U.tRef;
  return U.phiRef + dtt * (U.lnTau + r * (0.5 - r * (0.16666667 - r * 0.083333333)));
}

// High up (t > 200): Stirling's series for log chi.
fn chiHigh(sigma: f32, t: f32, dtt: f32) -> Chi {
  let u = 1.0 - sigma;
  let alpha = atan(u / t);
  let xx = (u / t) * (u / t);
  let eps = 0.5 * xx * (1.0 - 0.5 * xx);            // ln|1-s| - ln t
  let lnt = log(t);
  let zinv = cinv(vec2f(u, -t));                     // 1/(1-s)
  let re = (0.5 - sigma) * (lnt + eps) + t * alpha
    + sigma * 0.6931471805599453 + (sigma - 1.0) * 1.1447298858494002
    - 0.6931471805599453 - 1.0 + sigma + 0.9189385332046727 + zinv.x / 12.0;
  let im = -bigPhase(dtt) - t * eps + (0.5 - sigma) * (alpha - 1.5707963267948966)
    + 1.5707963267948966 * (1.0 - sigma) + zinv.y / 12.0;
  return Chi(vec2f(re, im), vec2f(1.0, 0.0), vec2f(1.8378770664093453 - lnt - eps, -alpha) + 0.5 * zinv + cmul(zinv, zinv) / 12.0);
}

// The six moments the Riemann–Siegel remainder is built from, as Chebyshev
// series in x = a - floor(a). Each already carries its constant: in order,
// M0, M1, M2, -(2 pi i/3) M3, i pi M4, -(2 pi^2/9) M6.
const SIEGEL = array<vec2f, ${6 * SIEGEL_TERMS}>(
  ${SIEGEL_TABLE});
fn moment(m: u32, y: f32) -> vec2f {
  let base = m * ${SIEGEL_TERMS}u;
  var b1 = vec2f(0.0);
  var b2 = vec2f(0.0);
  for (var k = ${SIEGEL_TERMS - 1}u; k >= 1u; k--) {
    let b0 = SIEGEL[base + k] + 2.0 * y * b1 - b2;
    b2 = b1;
    b1 = b0;
  }
  return SIEGEL[base] + y * b1 - b2;
}

// zeta(sigma + i t) by Riemann–Siegel, for sigma >= 0:
//   zeta(s) = R(s) + chi(s) conj(R(1 - conj s)),
//   R(s) = sum_{n <= N} n^-s + (-1)^N a^-s e^(i pi a^2) (c0 + c1/a + c2/a^2).
// The second R is carried already multiplied by |chi|, which is tiny when
// sigma is large, so that nothing overflows on the way.
fn siegel(sigma: f32, t: f32, dtt: f32) -> Pair {
  // a = sqrt(t/2pi) = aRef sqrt(1 + x), grown from the CPU's value so its fractional part keeps its digits.
  let x = dtt / U.tRef;
  let grown = U.aFrac + (f32(U.aInt) + U.aFrac) * x / (1.0 + sqrt(1.0 + x));
  let N = min(U.aInt + u32(grown), U.N);
  let frac = fract(grown);
  let a = f32(N) + frac;
  let lna = log(a);
  let ia = 1.0 / a;
  let chi = chiHigh(sigma, t, dtt);
  let tau = 1.0 - sigma;

  var A = vec2f(0.0);    // R(s)
  var Ap = vec2f(0.0);   // R'(s)
  var B = vec2f(0.0);    // |chi| R(1 - conj s)
  var Bp = vec2f(0.0);   // |chi| R'(1 - conj s)
  for (var n = 1u; n <= N; n++) {
    let e = tbl[n];
    let th = e.x + dtt * e.y;
    let w = vec2f(cos(th), -sin(th));
    let ta = exp(-sigma * e.y) * w;
    let tb = exp(chi.ln.x - tau * e.y) * w;
    A += ta;
    Ap -= e.y * ta;
    B += tb;
    Bp -= e.y * tb;
  }

  let y = 2.0 * frac - 1.0;
  let m0 = moment(0u, y);
  let m1 = moment(1u, y);
  let m2 = moment(2u, y);
  let m3 = moment(3u, y);
  let m4 = moment(4u, y);
  let m6 = moment(5u, y);
  // pi a^2 - t ln a is minus half the big phase.
  let psi = -0.5 * bigPhase(dtt);
  let turn = select(1.0, -1.0, (N & 1u) == 1u) * vec2f(cos(psi), sin(psi));

  let ca = m0 + ia * (m3 - sigma * m1 + ia * (0.5 * sigma * (sigma + 1.0) * m2 + (0.5 + sigma / 1.5) * m4 + m6));
  let da = ia * (-m1 + ia * ((sigma + 0.5) * m2 + m4 / 1.5));      // d ca / d sigma
  let pa = exp(-sigma * lna) * turn;
  let Ia = cmul(pa, ca);
  A += Ia;
  Ap += cmul(pa, da) - lna * Ia;

  let cb = m0 + ia * (m3 - tau * m1 + ia * (0.5 * tau * (tau + 1.0) * m2 + (0.5 + tau / 1.5) * m4 + m6));
  let db = ia * (-m1 + ia * ((tau + 0.5) * m2 + m4 / 1.5));
  let pb = exp(chi.ln.x - tau * lna) * turn;
  let Ib = cmul(pb, cb);
  B += Ib;
  Bp += cmul(pb, db) - lna * Ib;

  let rot = vec2f(cos(chi.ln.y), sin(chi.ln.y));
  let Bc = vec2f(B.x, -B.y);
  let Bpc = vec2f(Bp.x, -Bp.y);
  return Pair(A + cmul(rot, Bc), Ap + cmul(rot, cmul(chi.dlog, Bc) - Bpc));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= ${FW}u || gid.y >= U.rows) { return; }
  let sigma = sigmaOfCol(f32(gid.x));
  var dtt = f32(gid.y) * U.dt;
  // Step off the real axis: the pole and the tail's 1/(s+j) terms live there.
  if (abs(U.tRef + dtt) < 1.0e-3) { dtt += 2.0e-3; }
  let t = U.tRef + dtt;

  // Left of the critical strip the terms n^-s dwarf their own sum and single
  // precision drowns. There we sum zeta(1-s) instead, where the series is
  // tame, and come back through the functional equation.
  let mirror = sigma < 0.0;
  let sE = select(sigma, 1.0 - sigma, mirror);

  var p: Pair;
  if (t >= ${T_RS}.0) {
    p = siegel(sE, t, dtt);
    if (mirror) { p = Pair(vec2f(p.z.x, -p.z.y), vec2f(p.zp.x, -p.zp.y)); }
  } else {
    p = eulerMaclaurin(sE, select(1.0, -1.0, mirror), t, dtt);
  }

  if (mirror) {
    // Far up the left wall zeta outgrows single precision; only its phase and
    // the fact that it is huge matter there, so |chi| is held at e^60.
    var c: Chi;
    if (t < 200.0) { c = chiLow(sigma, t); } else { c = chiHigh(sigma, t, dtt); }
    let chi = cmul(cexp(vec2f(min(c.ln.x, 60.0), c.ln.y)), c.k);
    p = Pair(cmul(chi, p.z), cmul(chi, cmul(c.dlog, p.z) - p.zp));
  }

  let row = (U.rowTex + gid.y) % ${FH}u;
  textureStore(field, vec2u(gid.x, row), vec4f(p.z, p.zp));
}
`;

// ---------------------------------------------------------------------------
// Particles: steepest descent on log|zeta| (the Newton flow ds/dtau = -zeta/zeta'),
// optionally rotated so the paths spiral. Along each path arg(zeta) is constant
// when the swirl is zero, which is why a particle keeps its colour.
// ---------------------------------------------------------------------------
export const SIM_WGSL = /* wgsl */ `
${COMMON}
struct Particle { a: vec4f, b: vec4f };   // a = sigma, z, height, age   b = previous sigma, z, height; hue
${BALL_STRUCT}
@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var field: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> P: array<Particle>;
@group(0) @binding(3) var<storage, read> B: array<Ball>;
${FIELD_SAMPLE}

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn unit(v: u32) -> f32 { return f32(v >> 8u) / 16777216.0; }

// What the balls add to the flow, in units per second: each landing throws a
// ring outward and leaves an eddy turning where it struck, and a rolling ball
// parts the light and drags some along, harder the faster it goes.
fn disturb(p: vec2f) -> vec2f {
  var v = vec2f(0.0);
  for (var i = 0; i < 8; i++) {
    let h = G.hits[i];
    if (h.w <= 0.0) { continue; }
    let d = p - h.xy;
    let r = length(d) + 1.0e-4;
    let dir = d / r;
    let x = (r - 2.6 * h.z) / 0.45;
    let ring = exp(-x * x) * exp(-h.z * 0.55);
    let reach = 0.9 + 0.5 * h.z;
    let eddy = exp(-(r * r) / (reach * reach)) * exp(-h.z * 0.4);
    let turn = select(1.0, -1.0, ((u32(G.me2.z) >> u32(i)) & 1u) == 1u);
    // The ring must pass through the light, not sweep it along: its push is well under its own speed.
    v += h.w * (dir * ring * 2.0 + vec2f(-dir.y, dir.x) * eddy * 3.0 * turn);
  }
  let used = min(u32(G.me2.x), NB);
  for (var i = 0u; i < used; i++) {
    let at = B[i].p;
    if (at.w < 1.5) { continue; }
    let d = p - at.xy;
    let r2 = dot(d, d);
    if (r2 > 0.3) { continue; }
    let b = B[i];
    // Gentle enough that light bound for a zero still gets past a ball circling it.
    let near = exp(-r2 / 0.03) * (1.0 - b.q.w);
    v += (d * inverseSqrt(r2 + 1.0e-6) * 0.5 * length(b.v.xy) + b.v.xy * 0.8) * near;
  }
  return v;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= G.misc.z) { return; }
  let p = P[idx];

  let smin = sigmaOfCol(1.5);
  let smax = sigmaOfCol(f32(FW) - 2.5);
  let lo = (G.win.x + 1.0) * G.grid.z;
  let hi = (G.win.y - 2.0) * G.grid.z;

  var pos = vec2f(p.a.x, p.a.y - G.flow.w);
  var prev = vec3f(pos, p.a.z);
  let rate = 0.55 + 0.9 * unit(pcg(idx * 2654435761u + 17u));
  var age = p.a.w + G.flow2.y * rate;

  var alive = (G.misc.w & 1u) == 0u && age < 1.0
    && pos.x > smin && pos.x < smax && pos.y > lo && pos.y < hi;

  if (alive) {
    let j = jetOf(fieldAt(pos.x, pos.y));
    let m2 = dot(j.z, j.z);
    if (j.la == 0.0 && m2 < 4.0e-5) {
      alive = false;               // arrived at a zero
    } else if (dot(j.zp, j.zp) > 1.0e-30) {
      var w = -cdiv(j.z, j.zp);
      w = vec2f(w.x * G.flow.x - w.y * G.flow.y, w.x * G.flow.y + w.y * G.flow.x);
      var stepv = w * G.flow.z;
      let len = length(stepv);
      if (len > G.flow2.x) { stepv *= G.flow2.x / len; }
      if (len == len && len < 1.0e30) { pos += stepv; }
    }
    pos += disturb(pos) * G.win.w;
  }

  if (!alive) {
    var r = pcg(idx ^ pcg(G.misc.y * 9781u + 7u));
    let r0 = unit(r); r = pcg(r);
    let r1 = unit(r); r = pcg(r);
    let r2 = unit(r); r = pcg(r);
    let r3 = unit(r); r = pcg(r);
    let r4 = unit(r);
    // Some spread over the whole land, most gathered where the camera is looking.
    var sg = mix(smin, smax, r0);
    if (r2 < 0.6) { sg = clamp(G.me.x + (r0 + r3 - 1.0) * 6.0, smin, smax); }
    let w0 = G.win.z * G.grid.z;
    var zz = w0 + f32(FH) * G.grid.z * r1;
    let q = r1 * 2.0 - 1.0;
    if (r4 < 0.66) { zz = G.me.y + q * abs(q) * 30.0; }
    if (!(zz > lo && zz < hi)) {
      // That part of the field is not summed yet: wait there, unseen.
      P[idx] = Particle(vec4f(G.me.x, G.me.y, 0.0, 1.0), vec4f(G.me.x, G.me.y, 0.0, 0.0));
      return;
    }
    pos = vec2f(sg, zz);
    age = 0.12 * r3;
  }

  let j2 = jetOf(fieldAt(pos.x, pos.y));
  let y = G.grid.w * capL(logMod(j2)) + rippleAt(pos).x;
  let phase = hueOf(j2);
  if (!alive) { prev = vec3f(pos, y); }
  P[idx] = Particle(vec4f(pos, y, age), vec4f(prev, phase));
}
`;

// ---------------------------------------------------------------------------
// The surface y = log|zeta(sigma + it)|, drawn dark with a conformal grid:
// level lines of |zeta| and of arg zeta, which cross at right angles.
// ---------------------------------------------------------------------------
export const SURFACE_WGSL = /* wgsl */ `
${COMMON}
@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var field: texture_2d<f32>;
${FIELD_SAMPLE}

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) sz: vec2f,
  @location(1) valid: f32,
  @location(2) wpos: vec3f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var cx = array<u32, 6>(0u, 1u, 0u, 0u, 1u, 1u);
  var cy = array<u32, 6>(0u, 0u, 1u, 1u, 0u, 1u);
  let q = vi / 6u;
  let c = vi % 6u;
  let col = i32(q % ${MW}u + cx[c]);
  let rel = i32(G.win.z) + i32(q / ${MW}u + cy[c]);
  let F = textureLoad(field, vec2i(col, ringRow(rel)), 0);
  let ok = f32(rel) >= G.win.x && f32(rel) < G.win.y;
  let sigma = sigmaOfCol(f32(col));
  let z = f32(rel) * G.grid.z;
  var y = 0.0;
  if (ok) { y = G.grid.w * capL(logMod(jetOf(F))) + rippleAt(vec2f(sigma, z)).x; }
  var o: VSOut;
  o.wpos = vec3f(sigma, y, z);
  o.pos = G.viewProj * vec4f(o.wpos, 1.0);
  o.sz = vec2f(sigma, z);
  o.valid = select(0.0, 1.0, ok);
  return o;
}

fn lineMask(dist: f32, px: f32, width: f32) -> f32 {
  return 1.0 - smoothstep(width * px, (width + 1.2) * px, dist);
}

struct SurfaceOut { @location(0) col: vec4f, @location(1) far: vec4f };   // far: distance along the view, for placing clicks

@fragment
fn fs(in: VSOut) -> SurfaceOut {
  let px = max(length(dpdx(in.sz)), length(dpdy(in.sz)));
  if (in.valid < 0.999) { discard; }

  let j = jetOf(fieldAt(in.sz.x, in.sz.y));
  let m2 = dot(j.z, j.z);
  let L = logMod(j);
  let phase = atan2(j.z.y, j.z.x);

  // Gradient of the height. |grad log|zeta|| = |grad arg zeta| = |zeta'/zeta|.
  let q = vec2f(j.zp.x * j.z.x + j.zp.y * j.z.y, j.zp.y * j.z.x - j.zp.x * j.z.y) / (m2 + EPS2);
  let th = tanh(L / LMAX);
  let capd = select(1.0, 1.0 - th * th, L > 0.0);
  let slope = G.grid.w * capd * vec2f(q.x, -q.y);
  let n = normalize(vec3f(-slope.x, 1.0, -slope.y));
  let gmag = length(j.zp) / sqrt(m2 + 1.0e-12);

  let view = normalize(G.cam.xyz - in.wpos);
  let dist = length(G.cam.xyz - in.wpos);
  let hue = palette(hueOf(j));

  // Body: dark glass that holds a little of the colour, plus a grazing rim.
  let key = max(dot(n, normalize(vec3f(-0.45, 0.8, -0.35))), 0.0);
  let rim = pow(1.0 - max(dot(n, view), 0.0), 4.0);
  var col = hue * (0.03 + 0.07 * key) + mix(hue, vec3f(0.5, 0.65, 1.0), 0.5) * rim * 0.06;

  // The net: equal steps of pi/6 in log|zeta| and in arg zeta. The two
  // families cross at right angles everywhere and tighten into every zero.
  let stepv = PI / 6.0;
  let g = max(gmag, 1.0e-6);
  let dPhase = abs(fract(phase / stepv + 0.5) - 0.5) * stepv / g;
  let dMod = abs(fract(L / stepv + 0.5) - 0.5) * stepv / g;
  let cell = stepv / g;
  let dense = smoothstep(2.5 * px, 8.0 * px, cell);
  let net = max(lineMask(dPhase, px, 0.3), lineMask(dMod, px, 0.3) * 0.75) * dense;
  col += hue * net * 0.36 * G.res.w;

  // The line every zero sits on.
  let off = abs(in.sz.x - 0.5);
  col += GOLD * lineMask(off, px, 0.6) * 2.4;
  col += GOLD * exp(-off * 7.0) * 0.045;

  // Rings from where balls have landed.
  col += vec3f(1.0, 0.93, 0.8) * rippleAt(in.sz).y * 0.55;

  // Where the field is still being summed, the surface ends in a bright seam.
  let v = in.sz.y / G.grid.z;
  let edge = min(v - G.win.x, G.win.y - 1.0 - v) * G.grid.z;
  col += vec3f(0.5, 0.75, 1.0) * exp(-edge * 3.0) * 1.8 * G.aux.z;

  // Distance and the far sides melt into the colour of the horizon.
  let f = haze(dist) * sides(in.sz.x);
  return SurfaceOut(vec4f(mix(HORIZON, col, f), 1.0), vec4f(1.0 / in.pos.w, 0.0, 0.0, 0.0));
}
`;

// ---------------------------------------------------------------------------
// Balls. Dropped onto the surface, they roll downhill under gravity with a
// little friction. A zero is a funnel shaped like a logarithm, so a ball that
// misses the middle circles it at a steady speed in ever tighter turns, like
// a coin in a wishing well, before it goes down.
// ---------------------------------------------------------------------------
export const BALLS_WGSL = /* wgsl */ `
${COMMON}
${BALL_STRUCT}
struct Drops {
  d: array<vec4f, 4>,   // sigma, z, height to fall from, ball slot (-1: none)
  e: array<vec4f, 4>,   // hue, radius, unused, unused
};
@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var field: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> B: array<Ball>;
@group(0) @binding(3) var<storage, read_write> T: array<vec4f>;   // per ball: sigma, z, height, time laid down (0: nothing)
@group(0) @binding(4) var<uniform> D: Drops;
${FIELD_SAMPLE}

const GRAV: f32 = 6.5;
const FRICTION: f32 = 0.38;

// The height of the surface exactly as it is drawn: the mesh has one corner
// per field sample and flat triangles between, and a ball has to sit on those
// triangles, not on the smoother surface the samples were taken from.
fn cornerHeight(i: i32, row: i32) -> f32 {
  return capL(logMod(jetOf(textureLoad(field, vec2i(i, row), 0))));
}
fn drawnHeight(p: vec2f) -> f32 {
  let u = clamp(colOfSigma(p.x), 0.0, f32(FW) - 1.001);
  let v = max(p.y / G.grid.z, 0.0);
  let i = i32(floor(u));
  let j = i32(floor(v));
  let f = vec2f(u - floor(u), v - floor(v));
  let r0 = ringRow(j);
  let r1 = ringRow(j + 1);
  let b = cornerHeight(i + 1, r0);
  let c = cornerHeight(i, r1);
  var h: f32;
  if (f.x + f.y < 1.0) {
    let a = cornerHeight(i, r0);
    h = a + (b - a) * f.x + (c - a) * f.y;
  } else {
    let d = cornerHeight(i + 1, r1);
    h = d + (c - d) * (1.0 - f.x) + (b - d) * (1.0 - f.y);
  }
  return G.grid.w * h + rippleAt(p).x;
}

struct Ground { h: f32, grad: vec2f, L: f32, reach: f32, phase: f32 };   // reach: about how far the nearest zero is
fn groundAt(p: vec2f) -> Ground {
  let j = jetOf(fieldAt(p.x, p.y));
  let m2 = dot(j.z, j.z);
  let L = logMod(j);
  let q = vec2f(j.zp.x * j.z.x + j.zp.y * j.z.y, j.zp.y * j.z.x - j.zp.x * j.z.y) / (m2 + EPS2);
  let th = tanh(L / LMAX);
  let capd = select(1.0, 1.0 - th * th, L > 0.0);
  return Ground(drawnHeight(p), G.grid.w * capd * vec2f(q.x, -q.y), L, sqrt(m2 / max(dot(j.zp, j.zp), 1.0e-30)), atan2(j.z.y, j.z.x));
}

@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= NB) { return; }
  var b = B[i];
  let base = i * TRAIL;
  if (G.flow.w != 0.0) {
    // The window's origin moved: everything keeps its place on the land.
    b.p.y -= G.flow.w;
    for (var k = 0u; k < TRAIL; k++) { T[base + k].y -= G.flow.w; }
  }
  for (var s = 0; s < 4; s++) {
    let d = D.d[s];
    if (d.w >= 0.0 && u32(d.w) == i) {
      b.p = vec4f(d.xy, groundAt(d.xy).h + d.z, 1.0);
      b.v = vec4f(0.0, 0.0, -1.0, 0.0);
      b.q = vec4f(0.0);
      b.c = D.e[s];
      b.n = vec4f(0.0);
      for (var k = 0u; k < TRAIL; k++) { T[base + k] = vec4f(0.0); }
    }
  }
  if (b.p.w < 0.5) { B[i] = b; return; }

  let lo = (G.win.x + 2.0) * G.grid.z;
  let hi = (G.win.y - 3.0) * G.grid.z;
  let steps = 6;
  let h = G.win.w / f32(steps);
  for (var n = 0; n < steps; n++) {
    if (b.p.y < lo || b.p.y > hi || abs(b.p.x - 0.5) > 18.5) { b.p.w = 0.0; break; }
    let g = groundAt(b.p.xy);
    // The lines of the net are equal steps of pi/6 in arg zeta and in log|zeta|.
    // Which cell of the net the ball is in is what the music is played from.
    let sector = floor(g.phase / (PI / 6.0));
    let band = floor(g.L / (PI / 6.0));
    if (b.p.w < 1.5) {
      // Falling.
      b.v.z -= 2.0 * GRAV * h;
      b.p.z += b.v.z * h;
      if (b.p.z <= g.h) { b.p.z = g.h; b.p.w = 2.0; b.v.z = 0.0; b.n = vec4f(sector, 0.0, band, 0.0); }
    } else {
      // Rolling: gravity along the slope, less a little friction.
      let a = -GRAV * g.grad / (1.0 + dot(g.grad, g.grad)) - FRICTION * b.v.xy;
      var v = b.v.xy + a * h;
      let sp = length(v);
      if (sp > 8.0) { v *= 8.0 / sp; }
      b.v = vec4f(v, 0.0, b.v.w);
      let at = b.p.xy + v * h;
      b.p = vec4f(at, drawnHeight(at), b.p.w);
      b.c = vec4f(b.c.xy, g.grad);
      if (b.p.w < 2.5) {
        if (sector != b.n.x) { b.n.x = sector; b.n.y += 1.0; }
        if (band != b.n.z) { b.n.z = band; b.n.w += 1.0; }
        b.q.z = select(0.0, b.q.z + h, sp < 0.06);
        // It goes when the funnel of a zero has narrowed to its own width,
        // or when it has come to rest on the flat, or is simply old.
        if ((g.reach < 1.15 * b.c.y && g.L < -1.0) || b.q.z > 2.5 || b.v.w > 150.0) { b.p.w = 3.0; }
      } else {
        // Going: it dwindles where it is, still on the surface, and is gone.
        b.q.w += h / 0.4;
        if (b.q.w >= 1.0) { b.p.w = 0.0; break; }
      }
    }
  }

  b.v.w += G.win.w;
  b.q.x += G.win.w;
  if (b.p.w > 1.5 && b.q.x >= ${TRAIL_EVERY}) {
    b.q.x = 0.0;
    let head = (u32(b.q.y) + 1u) % TRAIL;
    b.q.y = f32(head);
    T[base + head] = vec4f(b.p.xyz, G.cam.w);
  }
  B[i] = b;
}
`;

// The balls as pearls of light, and the tracks they leave.
export const BALL_DRAW_WGSL = /* wgsl */ `
${COMMON}
${BALL_STRUCT}
@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> B: array<Ball>;
@group(0) @binding(2) var<storage, read> T: array<vec4f>;

struct BOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) col: vec3f,
  @location(2) gain: f32,
};

@vertex
fn vsBall(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> BOut {
  let b = B[ii];
  var o: BOut;
  o.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  if (b.p.w < 0.5) { return o; }
  // It rests on the slope: its centre is one radius out along the surface's normal.
  let r = b.c.y * (1.0 - b.q.w);
  let up = select(vec3f(0.0, 1.0, 0.0), normalize(vec3f(-b.c.z, 1.0, -b.c.w)), b.p.w > 1.5);
  let centre = vec3f(b.p.x, b.p.z, b.p.y) + up * r;
  let c = G.viewProj * vec4f(centre, 1.0);
  if (c.w < 0.1) { return o; }
  // Depth of the point of the ball nearest the eye, so the ground behind it never cuts into it.
  let front = G.viewProj * vec4f(centre + normalize(G.cam.xyz - centre) * r, 1.0);
  let px = max(r * G.grid.y / c.w, 1.6);
  let corner = vec2f(f32(vi & 1u), f32(vi >> 1u)) * 2.0 - 1.0;
  o.pos = vec4f(c.xy + corner * px / (G.res.xy * 0.5) * c.w, front.z / front.w * c.w, c.w);
  o.uv = corner;
  o.col = palette(b.c.x);
  o.gain = (1.0 + 5.0 * b.q.w) * mix(0.35, 1.0, haze(length(G.cam.xyz - centre)));
  return o;
}

@fragment
fn fsBall(in: BOut) -> @location(0) vec4f {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let n = vec3f(in.uv, sqrt(1.0 - r2));
  let key = max(dot(n, normalize(vec3f(-0.4, 0.6, 0.7))), 0.0);
  let rim = pow(1.0 - n.z, 2.0);
  let glint = pow(max(dot(n, normalize(vec3f(-0.3, 0.5, 0.8))), 0.0), 48.0);
  let col = vec3f(1.0, 0.95, 0.86) * (0.3 + 0.55 * key) + in.col * (0.3 + 1.8 * rim) + vec3f(1.0) * glint * 2.5;
  return vec4f(col * in.gain, 1.0);
}

struct TOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) col: vec3f,
  @location(2) hl: f32,
};

// One piece of track per instance, from one sample to the next; the newest
// piece runs from the last sample to the ball itself.
@vertex
fn vsTrail(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> TOut {
  let bi = ii / TRAIL;
  let j = ii % TRAIL;
  let b = B[bi];
  let a = T[bi * TRAIL + j];
  var z = T[bi * TRAIL + (j + 1u) % TRAIL];
  var o: TOut;
  o.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  if (a.w <= 0.0) { return o; }
  if (j == u32(b.q.y)) {
    if (b.p.w < 0.5) { return o; }
    z = vec4f(b.p.xyz, G.cam.w);
  } else if (z.w < a.w) { return o; }
  let fade = 1.0 - (G.cam.w - a.w) / ${(TRAIL_EVERY * (TRAIL - 3)).toFixed(3)};
  if (fade <= 0.0) { return o; }

  let lift = b.c.y * 0.45;
  let p0 = vec3f(a.x, a.z + lift, a.y);
  let p1 = vec3f(z.x, z.z + lift, z.y);
  let c0 = G.viewProj * vec4f(p0, 1.0);
  let c1 = G.viewProj * vec4f(p1, 1.0);
  if (c0.w < 0.08 || c1.w < 0.08) { return o; }
  let half = G.res.xy * 0.5;
  let s0 = c0.xy / c0.w * half;
  let s1 = c1.xy / c1.w * half;
  var dir = s1 - s0;
  let len = length(dir);
  dir = select(vec2f(1.0, 0.0), dir / max(len, 1.0e-5), len > 1.0e-3);
  let nrm = vec2f(-dir.y, dir.x);
  let r = clamp(0.4 * b.c.y * G.grid.y / c1.w, 1.1, 7.0);
  let along = f32(vi & 1u);
  let side = f32(vi >> 1u) * 2.0 - 1.0;
  let px = mix(s0, s1, along) + dir * (along * 2.0 - 1.0) * r + nrm * side * r;
  let base = mix(c0, c1, along);
  o.pos = vec4f(px / half * base.w, base.z, base.w);
  o.hl = 0.5 * len / r;
  o.uv = vec2f((along * 2.0 - 1.0) * (o.hl + 1.0), side);
  o.col = mix(palette(b.c.x), vec3f(1.0, 0.95, 0.86), 0.25 * fade) * fade * fade * 0.55 * haze(length(G.cam.xyz - p1));
  return o;
}

@fragment
fn fsTrail(in: TOut) -> @location(0) vec4f {
  let d = length(vec2f(max(abs(in.uv.x) - in.hl, 0.0), in.uv.y));
  let core = max(1.0 - d * d, 0.0);
  return vec4f(in.col * core * core, 0.0);
}
`;

export const PARTICLE_WGSL = /* wgsl */ `
${COMMON}
struct Particle { a: vec4f, b: vec4f };
@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> P: array<Particle>;

struct POut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) col: vec3f,
  @location(2) hl: f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> POut {
  let p = P[ii];
  let lift = 0.012;
  let cur = vec3f(p.a.x, p.a.z + lift, p.a.y);
  let was = vec3f(p.b.x, p.b.z + lift, p.b.y);
  let tail = cur - (cur - was) * G.flow2.z;
  let c1 = G.viewProj * vec4f(cur, 1.0);
  let c0 = G.viewProj * vec4f(tail, 1.0);
  var o: POut;
  // Skip what is behind the eye, and particles still waiting for their part of the field.
  if (c1.w < 0.08 || c0.w < 0.08 || p.a.w >= 1.0) {
    o.pos = vec4f(0.0, 0.0, -1.0, 1.0);
    return o;
  }
  let half = G.res.xy * 0.5;
  let s1 = c1.xy / c1.w * half;
  let s0 = c0.xy / c0.w * half;
  var dir = s1 - s0;
  let len = length(dir);
  dir = select(vec2f(1.0, 0.0), dir / max(len, 1.0e-5), len > 1.0e-3);
  let nrm = vec2f(-dir.y, dir.x);

  let ideal = G.flow2.w / c1.w;
  let r = clamp(ideal, 0.75, 5.0);
  let energy = min(1.0, (ideal * ideal) / (r * r));

  let along = f32(vi & 1u);
  let side = f32(vi >> 1u) * 2.0 - 1.0;
  let px = mix(s0, s1, along) + dir * (along * 2.0 - 1.0) * r + nrm * side * r;
  let base = mix(c0, c1, along);
  o.pos = vec4f(px / half * base.w, base.z, base.w);
  let hl = 0.5 * len / r;
  o.uv = vec2f((along * 2.0 - 1.0) * (hl + 1.0), side);
  o.hl = hl;

  let age = p.a.w;
  let fade = smoothstep(0.0, 0.08, age) * (1.0 - smoothstep(0.75, 1.0, age));
  let dist = length(G.cam.xyz - cur);
  // A few particles burn much brighter than the rest.
  let u = fract(f32(ii) * 0.61803398875);
  let spark = 0.14 + 9.0 * pow(u, 10.0);
  // Going down a zero the light dims, so the mouth of each one is dark.
  let sunk = mix(0.1, 1.0, smoothstep(-1.4, -0.3, p.a.z));
  o.col = palette(p.b.w) * fade * sunk * haze(dist) * sides(p.a.x) * energy * spark * G.res.z / (1.0 + 0.3 * hl);
  return o;
}

@fragment
fn fs(in: POut) -> @location(0) vec4f {
  // A comet: full width and brightness at the head, thinning to nothing behind.
  let along = clamp(in.uv.x / (in.hl + 1.0) * 0.5 + 0.5, 0.0, 1.0);
  let taper = mix(1.0, mix(0.3, 1.0, along), clamp(in.hl, 0.0, 1.0));
  let d = length(vec2f(max(abs(in.uv.x) - in.hl, 0.0), in.uv.y / taper));
  let core = max(1.0 - d * d, 0.0);
  let tail = mix(1.0, along * along, clamp(in.hl, 0.0, 1.0));
  return vec4f(in.col * core * core * tail, 0.0);
}
`;

// ---------------------------------------------------------------------------
// Post: bloom chain, tone mapping, grade.
// ---------------------------------------------------------------------------
export const POST_WGSL = /* wgsl */ `
struct PostU {
  exposure: f32, bloom: f32, time: f32, fade: f32,
  right: vec4f,   // camera right, tan(fov/2) * aspect
  up: vec4f,      // camera up, tan(fov/2)
  fwd: vec4f,
};
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var src2: texture_2d<f32>;
@group(0) @binding(3) var<uniform> U: PostU;
const HORIZON: vec3f = ${HORIZON};

struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var o: VOut;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}

fn tap(uv: vec2f) -> vec3f { return textureSampleLevel(src, samp, uv, 0.0).rgb; }

// 13-tap downsample: stable under motion, no fireflies from single hot pixels.
fn down13(uv: vec2f) -> vec3f {
  let ts = 1.0 / vec2f(textureDimensions(src));
  let a = tap(uv + ts * vec2f(-2.0, -2.0));
  let b = tap(uv + ts * vec2f(0.0, -2.0));
  let c = tap(uv + ts * vec2f(2.0, -2.0));
  let d = tap(uv + ts * vec2f(-2.0, 0.0));
  let e = tap(uv);
  let f = tap(uv + ts * vec2f(2.0, 0.0));
  let g = tap(uv + ts * vec2f(-2.0, 2.0));
  let h = tap(uv + ts * vec2f(0.0, 2.0));
  let i = tap(uv + ts * vec2f(2.0, 2.0));
  let j = tap(uv + ts * vec2f(-1.0, -1.0));
  let k = tap(uv + ts * vec2f(1.0, -1.0));
  let l = tap(uv + ts * vec2f(-1.0, 1.0));
  let m = tap(uv + ts * vec2f(1.0, 1.0));
  return e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
}

@fragment
fn fsPrefilter(in: VOut) -> @location(0) vec4f {
  let c = down13(in.uv);
  let br = max(c.r, max(c.g, c.b));
  let knee = 0.5;
  let thr = 0.55;
  let soft = clamp(br - thr + knee, 0.0, 2.0 * knee);
  let w = max(soft * soft / (4.0 * knee + 1.0e-4), br - thr) / max(br, 1.0e-4);
  return vec4f(min(c * w, vec3f(24.0)), 1.0);
}

@fragment
fn fsDown(in: VOut) -> @location(0) vec4f { return vec4f(down13(in.uv), 1.0); }

@fragment
fn fsUp(in: VOut) -> @location(0) vec4f {
  let ts = 1.0 / vec2f(textureDimensions(src));
  var c = tap(in.uv) * 4.0;
  c += (tap(in.uv + ts * vec2f(1.0, 0.0)) + tap(in.uv - ts * vec2f(1.0, 0.0))
      + tap(in.uv + ts * vec2f(0.0, 1.0)) + tap(in.uv - ts * vec2f(0.0, 1.0))) * 2.0;
  c += tap(in.uv + ts * vec2f(1.0, 1.0)) + tap(in.uv + ts * vec2f(-1.0, 1.0))
     + tap(in.uv + ts * vec2f(1.0, -1.0)) + tap(in.uv + ts * vec2f(-1.0, -1.0));
  return vec4f(c / 16.0, 1.0);
}

fn hash21(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
// Sparse stars, fixed in the sky, thinning out toward the horizon haze.
fn stars(dir: vec3f) -> f32 {
  if (dir.y < 0.03) { return 0.0; }
  let uv = vec2f(atan2(dir.x, dir.z), asin(clamp(dir.y, -1.0, 1.0))) * 46.0;
  let cell = floor(uv);
  let h = hash21(cell);
  if (h < 0.86) { return 0.0; }
  let spot = vec2f(hash21(cell + 17.0), hash21(cell + 43.0)) * 0.7 + 0.15;
  let d = length(fract(uv) - spot);
  let twinkle = 0.75 + 0.25 * sin(U.time * (1.0 + 3.0 * h) + h * 40.0);
  return smoothstep(0.07, 0.0, d) * (h - 0.86) * 7.0 * twinkle * smoothstep(0.03, 0.3, dir.y);
}

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fsComposite(in: VOut) -> @location(0) vec4f {
  let scene = textureSampleLevel(src, samp, in.uv, 0.0);
  let bloom = textureSampleLevel(src2, samp, in.uv, 0.0).rgb;
  let q = in.uv - 0.5;
  // Sky: a low band of light on the horizon, the same colour the distance fades to.
  let dir = normalize(U.fwd.xyz + U.right.xyz * (q.x * 2.0 * U.right.w) - U.up.xyz * (q.y * 2.0 * U.up.w));
  // Dusk above the horizon. Below it, the same haze the land dissolves into,
  // so the edges of the computed world are never seen.
  let sky = HORIZON * (0.08 + 0.92 * exp(-max(dir.y, 0.0) * 3.4)) + vec3f(0.75, 0.82, 1.0) * stars(dir) * 0.5;
  var c = (scene.rgb + sky * (1.0 - clamp(scene.a, 0.0, 1.0)) + bloom * U.bloom) * U.exposure * U.fade;
  c = aces(c);
  c *= 1.0 - 0.9 * dot(q, q);
  c = pow(c, vec3f(1.0 / 2.2));
  let n = fract(sin(dot(in.pos.xy + U.time * 61.0, vec2f(12.9898, 78.233))) * 43758.5453);
  c += (n - 0.5) / 255.0;
  return vec4f(c, 1.0);
}
`;
