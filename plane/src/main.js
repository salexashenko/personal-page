import {
  FW, FH, DT, MW, MH, NMAX, HEIGHT, NB, TRAIL, sigmaOfCol,
  FIELD_WGSL, SIM_WGSL, SURFACE_WGSL, BALLS_WGSL, BALL_DRAW_WGSL, PARTICLE_WGSL, POST_WGSL,
} from './shaders.js';
import { termsFor, zeta } from './zeta.js';
import { T_RS } from './siegel.js';
import { createSound } from './sound.js';

const TWO_PI = 2 * Math.PI;
const TERM_BUDGET = 3e7;        // series terms summed per frame while filling, before adapting
const SIEGEL_WORK = 40;         // what a Riemann–Siegel point costs beyond its sum, in terms
const BLOOM_LEVELS = 6;
const T_MAX = 120000;
const PARTICLES = 1 << 21;
const HITS = 8;                 // landing rings alive at once
const DROP = 2.4;               // height a ball falls from
const POUR_EVERY = 0.13;        // seconds between balls while the button is held
const NEAR = 0.05, FAR = 240;
const FOV = (50 * Math.PI) / 180;

// Heights worth visiting, on keys 1 to 5: the first zeros, a stretch further
// up, zero 1,000, Lehmer's pair (two zeros that almost collide), zero 100,000.
const PLACES = [22, 60, 1418, 7004, 74920];

const canvas = document.getElementById('gl');
const placard = document.getElementById('placard');
const sEl = document.getElementById('s');
const tEl = document.getElementById('t');
const hint = document.getElementById('hint');
const calm = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const sound = createSound();    // silent until the first click or key, as browsers require

// Look, exposed for tuning from the console.
const look = { gain: 0.17, gridGain: 1.0, exposure: 1.15, bloom: 0.6, streak: 6.0, flow: 1.5, size: 0.011, swirl: 0.7 };

// The camera looks at the point 1/2 + i t on the line from `dist` away,
// turned `yaw` from straight up the line and tipped `pitch` down.
// `want` is where it is going, `view` where it is; one eases toward the other.
const want = { t: PLACES[0], dist: 13, yaw: 0, pitch: 0.62 };
const view = { ...want };

const keys = new Set();
let hits = [];                  // landing rings: { s, t, born, strength, turn }
let drops = [];                 // balls waiting to be handed to the GPU: { s, t }
let picks = [];                 // clicks waiting to be turned into points on the land: { x, y } in pixels
let lastMark = -1e9, markCount = 0, dropCount = 0;
let time = 0;
let fade = 0, fadeTarget = 1, pendingPlace = null;
let newest = null;              // the ball last dropped, as far as we know: { slot, s, t, state }

function fail(message) {
  document.getElementById('unsupported').hidden = false;
  document.getElementById('why').textContent = message;
  placard.hidden = hint.hidden = true;
}

// ---------------------------------------------------------------------------
// Small column-major matrix helpers (left-handed view, depth 0..1).
// ---------------------------------------------------------------------------
function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect; m[5] = f;
  m[10] = far / (far - near); m[11] = 1;
  m[14] = -(far * near) / (far - near);
  return m;
}
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
/** View matrix and camera basis for an eye looking along `f`. */
function viewFrom(eye, f) {
  const r = norm(cross([0, 1, 0], f));
  const u = cross(f, r);
  const m = new Float32Array(16);
  m[0] = r[0]; m[4] = r[1]; m[8] = r[2]; m[12] = -dot3(r, eye);
  m[1] = u[0]; m[5] = u[1]; m[9] = u[2]; m[13] = -dot3(u, eye);
  m[2] = f[0]; m[6] = f[1]; m[10] = f[2]; m[14] = -dot3(f, eye);
  m[15] = 1;
  return { m, r, u, f };
}
function mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  }
  return o;
}
const mod = (a, n) => ((a % n) + n) % n;
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const ease = (dt, tau) => 1 - Math.exp(-dt / tau);

// ---------------------------------------------------------------------------
async function start() {
  if (!navigator.gpu) return fail('This browser has no WebGPU. Open the page in a current Chrome, Edge or Safari.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return fail('WebGPU is present but no graphics adapter is available. Check that hardware acceleration is on.');

  const device = await adapter.requestDevice();
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU:', e.error.message));
  device.lost.then((info) => fail(`The graphics device was lost (${info.message || info.reason}). Reload the page to start again.`));

  const context = canvas.getContext('webgpu');
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format: canvasFormat, alphaMode: 'opaque' });

  async function shader(code, label) {
    const m = device.createShaderModule({ code, label });
    const info = await m.getCompilationInfo();
    for (const msg of info.messages) {
      if (msg.type === 'error') console.error(`${label}:${msg.lineNum}:${msg.linePos} ${msg.message}`);
    }
    return m;
  }
  const [fieldMod, simMod, surfaceMod, ballsMod, ballDrawMod, particleMod, postMod] = await Promise.all([
    shader(FIELD_WGSL, 'field'), shader(SIM_WGSL, 'sim'), shader(SURFACE_WGSL, 'surface'),
    shader(BALLS_WGSL, 'balls'), shader(BALL_DRAW_WGSL, 'ball draw'), shader(PARTICLE_WGSL, 'particles'), shader(POST_WGSL, 'post'),
  ]);

  // --- resources that never change size -----------------------------------
  const GLOBALS = 224 + HITS * 16;
  const globals = new ArrayBuffer(GLOBALS);
  const gF = new Float32Array(globals);
  const gU = new Uint32Array(globals);
  const globalsBuf = device.createBuffer({ size: GLOBALS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const fieldTex = device.createTexture({
    size: [FW, FH], format: 'rgba32float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const fieldView = fieldTex.createView();
  const tableBuf = device.createBuffer({ size: 8 * (NMAX + 1), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const fieldUBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const postU = new Float32Array(16);
  const postUBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const particleBuf = device.createBuffer({ size: PARTICLES * 32, usage: GPUBufferUsage.STORAGE });
  const BALL_BYTES = 80, BALL_FLOATS = BALL_BYTES / 4;
  const ballBuf = device.createBuffer({ size: NB * BALL_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const ballRead = device.createBuffer({ size: NB * BALL_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const trackBuf = device.createBuffer({ size: NB * TRAIL * 16, usage: GPUBufferUsage.STORAGE });
  const dropsU = new Float32Array(32);
  const dropsBuf = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const lnTable = new Float64Array(NMAX + 1);
  for (let n = 1; n <= NMAX; n++) lnTable[n] = Math.log(n);
  const phaseTable = new Float32Array(2 * (NMAX + 1));

  // --- layouts -------------------------------------------------------------
  const C = GPUShaderStage.COMPUTE, V = GPUShaderStage.VERTEX, F = GPUShaderStage.FRAGMENT;
  const fieldBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: C, buffer: { type: 'uniform' } },
    { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: C, storageTexture: { access: 'write-only', format: 'rgba32float' } },
  ] });
  const simBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: C, buffer: { type: 'uniform' } },
    { binding: 1, visibility: C, texture: { sampleType: 'unfilterable-float' } },
    { binding: 2, visibility: C, buffer: { type: 'storage' } },
    { binding: 3, visibility: C, buffer: { type: 'read-only-storage' } },
  ] });
  const ballsBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: C, buffer: { type: 'uniform' } },
    { binding: 1, visibility: C, texture: { sampleType: 'unfilterable-float' } },
    { binding: 2, visibility: C, buffer: { type: 'storage' } },
    { binding: 3, visibility: C, buffer: { type: 'storage' } },
    { binding: 4, visibility: C, buffer: { type: 'uniform' } },
  ] });
  const surfaceBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: V | F, buffer: { type: 'uniform' } },
    { binding: 1, visibility: V | F, texture: { sampleType: 'unfilterable-float' } },
  ] });
  const particleBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: V | F, buffer: { type: 'uniform' } },
    { binding: 1, visibility: V, buffer: { type: 'read-only-storage' } },
  ] });
  const ballDrawBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: V | F, buffer: { type: 'uniform' } },
    { binding: 1, visibility: V, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: V, buffer: { type: 'read-only-storage' } },
  ] });
  const blurBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: F, sampler: { type: 'filtering' } },
    { binding: 1, visibility: F, texture: { sampleType: 'float' } },
  ] });
  const compositeBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: F, sampler: { type: 'filtering' } },
    { binding: 1, visibility: F, texture: { sampleType: 'float' } },
    { binding: 2, visibility: F, texture: { sampleType: 'float' } },
    { binding: 3, visibility: F, buffer: { type: 'uniform' } },
  ] });
  const layout = (bgl) => device.createPipelineLayout({ bindGroupLayouts: [bgl] });

  // --- pipelines -----------------------------------------------------------
  const HDR = 'rgba16float';
  const add = { srcFactor: 'one', dstFactor: 'one', operation: 'add' };
  const glow = { format: HDR, blend: { color: add, alpha: add } };
  // A second target holds how far away the land is at each pixel, so a click can be turned into a place.
  const PICK = 'r32float';
  const noPick = { format: PICK, writeMask: 0 };
  const solid = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' };
  const overSurface = { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'less-equal' };
  const fieldPipe = device.createComputePipeline({ layout: layout(fieldBGL), compute: { module: fieldMod, entryPoint: 'main' } });
  const simPipe = device.createComputePipeline({ layout: layout(simBGL), compute: { module: simMod, entryPoint: 'main' } });
  const ballsPipe = device.createComputePipeline({ layout: layout(ballsBGL), compute: { module: ballsMod, entryPoint: 'main' } });
  const surfacePipe = device.createRenderPipeline({
    layout: layout(surfaceBGL),
    vertex: { module: surfaceMod, entryPoint: 'vs' },
    fragment: { module: surfaceMod, entryPoint: 'fs', targets: [{ format: HDR }, { format: PICK }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: solid,
  });
  const particlePipe = device.createRenderPipeline({
    layout: layout(particleBGL),
    vertex: { module: particleMod, entryPoint: 'vs' },
    fragment: { module: particleMod, entryPoint: 'fs', targets: [glow, noPick] },
    primitive: { topology: 'triangle-strip' },
    depthStencil: overSurface,
  });
  const ballPipe = device.createRenderPipeline({
    layout: layout(ballDrawBGL),
    vertex: { module: ballDrawMod, entryPoint: 'vsBall' },
    fragment: { module: ballDrawMod, entryPoint: 'fsBall', targets: [{ format: HDR }, noPick] },
    primitive: { topology: 'triangle-strip' },
    depthStencil: solid,
  });
  const trackPipe = device.createRenderPipeline({
    layout: layout(ballDrawBGL),
    vertex: { module: ballDrawMod, entryPoint: 'vsTrail' },
    fragment: { module: ballDrawMod, entryPoint: 'fsTrail', targets: [glow, noPick] },
    primitive: { topology: 'triangle-strip' },
    depthStencil: overSurface,
  });
  const blurPipe = (entry, blend) => device.createRenderPipeline({
    layout: layout(blurBGL),
    vertex: { module: postMod, entryPoint: 'vs' },
    fragment: { module: postMod, entryPoint: entry, targets: [{ format: HDR, blend }] },
  });
  const prefilterPipe = blurPipe('fsPrefilter');
  const downPipe = blurPipe('fsDown');
  const upPipe = blurPipe('fsUp', { color: add, alpha: add });
  const compositePipe = device.createRenderPipeline({
    layout: layout(compositeBGL),
    vertex: { module: postMod, entryPoint: 'vs' },
    fragment: { module: postMod, entryPoint: 'fsComposite', targets: [{ format: canvasFormat }] },
  });

  const fieldBG = device.createBindGroup({ layout: fieldBGL, entries: [
    { binding: 0, resource: { buffer: fieldUBuf } },
    { binding: 1, resource: { buffer: tableBuf } },
    { binding: 2, resource: fieldView },
  ] });
  const surfaceBG = device.createBindGroup({ layout: surfaceBGL, entries: [
    { binding: 0, resource: { buffer: globalsBuf } },
    { binding: 1, resource: fieldView },
  ] });
  const simBG = device.createBindGroup({ layout: simBGL, entries: [
    { binding: 0, resource: { buffer: globalsBuf } },
    { binding: 1, resource: fieldView },
    { binding: 2, resource: { buffer: particleBuf } },
    { binding: 3, resource: { buffer: ballBuf } },
  ] });
  const ballsBG = device.createBindGroup({ layout: ballsBGL, entries: [
    { binding: 0, resource: { buffer: globalsBuf } },
    { binding: 1, resource: fieldView },
    { binding: 2, resource: { buffer: ballBuf } },
    { binding: 3, resource: { buffer: trackBuf } },
    { binding: 4, resource: { buffer: dropsBuf } },
  ] });
  const particleBG = device.createBindGroup({ layout: particleBGL, entries: [
    { binding: 0, resource: { buffer: globalsBuf } },
    { binding: 1, resource: { buffer: particleBuf } },
  ] });
  const ballDrawBG = device.createBindGroup({ layout: ballDrawBGL, entries: [
    { binding: 0, resource: { buffer: globalsBuf } },
    { binding: 1, resource: { buffer: ballBuf } },
    { binding: 2, resource: { buffer: trackBuf } },
  ] });
  let respawnAll = true;

  // --- size-dependent targets ---------------------------------------------
  let W = 0, H = 0, hdrView, depthView, pickTex, pickView, mips = [], blurBGs = [], prefilterBG, compositeBG;
  let targets = [];
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(2, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(2, Math.round(canvas.clientHeight * dpr));
    if (w === W && h === H) return;
    W = w; H = h;
    canvas.width = W; canvas.height = H;
    for (const t of targets) t.destroy();
    targets = [];
    const make = (size, format, extra = 0) => {
      const t = device.createTexture({ size, format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | extra });
      targets.push(t);
      return t;
    };
    hdrView = make([W, H], HDR).createView();
    depthView = make([W, H], 'depth32float').createView();
    pickTex = make([W, H], PICK, GPUTextureUsage.COPY_SRC);
    pickView = pickTex.createView();
    mips = [];
    for (let i = 0; i < BLOOM_LEVELS; i++) {
      mips.push(make([Math.max(1, W >> (i + 1)), Math.max(1, H >> (i + 1))], HDR).createView());
    }
    const blurBG = (v) => device.createBindGroup({ layout: blurBGL, entries: [
      { binding: 0, resource: sampler }, { binding: 1, resource: v },
    ] });
    prefilterBG = blurBG(hdrView);
    blurBGs = mips.map(blurBG);
    compositeBG = device.createBindGroup({ layout: compositeBGL, entries: [
      { binding: 0, resource: sampler }, { binding: 1, resource: hdrView },
      { binding: 2, resource: mips[0] }, { binding: 3, resource: { buffer: postUBuf } },
    ] });
  }

  // --- the sliding window of computed rows --------------------------------
  // Row R of the field is t = R * DT and lives at texture row R mod FH, so the
  // journey can go on indefinitely and each frame only sums the rows that are new.
  let originRow = 0;                 // world z = (R - originRow) * DT
  let validLo = 0, validHi = 0;      // rows [validLo, validHi) hold current data
  let filling = 0;
  let lead = 10;                     // how far past the point you look at, along t, the window is centred
  // The fill rate adapts to what this GPU manages in a frame: it grows while
  // frames stay as quick as the quickest recently seen, and backs off when they slow.
  let fillScale = 1, dtSmooth = 1 / 60, baseDt = 1 / 60;

  function computeRows(startRow, rows) {
    const tRef = startRow * DT;
    const tMax = Math.max(Math.abs(tRef), Math.abs(tRef + rows * DT));
    // Above T_RS a point sums only as far as sqrt(t/2pi); below, Euler–Maclaurin's t/2.
    const N = Math.min(NMAX, tRef >= T_RS ? Math.floor(Math.sqrt(tMax / TWO_PI)) + 1 : termsFor(tMax));
    // t * ln(n) mod 2pi in double precision; the shader adds only a small offset.
    for (let n = 1; n <= N; n++) {
      const ln = lnTable[n];
      const x = tRef * ln;
      phaseTable[2 * n] = x - TWO_PI * Math.floor(x / TWO_PI);
      phaseTable[2 * n + 1] = ln;
    }
    device.queue.writeBuffer(tableBuf, 0, phaseTable, 0, 2 * (N + 1));
    const u = new ArrayBuffer(48);
    // The one huge phase in the functional equation, reduced here in double precision
    // (mod 4pi, since Riemann–Siegel wants half of it), and sqrt(t/2pi) split so its fraction keeps its digits.
    const lnTau = Math.log(Math.max(tRef, 1) / TWO_PI);
    const phi = Math.max(tRef, 1) * (lnTau - 1);
    const a = Math.sqrt(Math.max(tRef, 1) / TWO_PI);
    new Uint32Array(u, 0, 4).set([mod(startRow, FH), rows, N, Math.floor(a)]);
    new Float32Array(u, 16, 5).set([DT, tRef, phi - 2 * TWO_PI * Math.floor(phi / (2 * TWO_PI)), lnTau, a - Math.floor(a)]);
    device.queue.writeBuffer(fieldUBuf, 0, u);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(fieldPipe);
    pass.setBindGroup(0, fieldBG);
    pass.dispatchWorkgroups(FW / 64, rows);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  function updateWindow(dtReal) {
    // The window leans toward wherever the camera is facing.
    lead += (10 * Math.cos(view.yaw) - lead) * ease(dtReal, 1.5);
    const d0 = Math.floor((view.t + lead) / DT - FH / 2);
    const d1 = d0 + FH;
    const newOrigin = Math.floor(d0 / 256) * 256;
    const zShift = (newOrigin - originRow) * DT;
    originRow = newOrigin;

    if (validHi <= d0 || validLo >= d1 || validHi <= validLo) {
      // Nothing usable (first frame, or a jump): grow outward from where you are looking.
      validLo = validHi = clamp(Math.round(view.t / DT), d0, d1);
    }
    validLo = Math.max(validLo, d0);
    validHi = Math.min(validHi, d1);

    const top = Math.abs(view.t) + FH * DT;
    const terms = top < T_RS + 2 * FH * DT ? termsFor(top) : Math.sqrt(top / TWO_PI) + SIEGEL_WORK;
    const needUp = d1 - validHi, needDown = validLo - d0;
    filling = needUp + needDown;
    dtSmooth += (dtReal - dtSmooth) * 0.15;
    baseDt = Math.min(baseDt * 1.001, dtSmooth);
    if (filling > 0) {
      if (dtSmooth < baseDt * 1.12) fillScale = Math.min(fillScale * 1.1, 256);
      else if (dtSmooth > baseDt * 1.35) fillScale = Math.max(fillScale * 0.8, 0.5);
    }
    const budget = Math.max(1, Math.floor((TERM_BUDGET * fillScale) / (FW * terms)));
    const up = Math.min(needUp, needDown > 0 ? Math.max(1, Math.round(budget * 0.6)) : budget, 256);
    const down = Math.min(needDown, Math.max(1, budget - up), 256);
    if (up > 0) { computeRows(validHi, up); validHi += up; }
    if (down > 0) { computeRows(validLo - down, down); validLo -= down; }
    return { d0, zShift };
  }

  // --- balls ---------------------------------------------------------------
  // The GPU rolls them. Here we only decide where each one starts and which
  // slot it takes, and now and then read back which slots are still in use.
  const slotState = new Uint8Array(NB);     // as of the last read-back
  const slotBorn = new Float64Array(NB).fill(-1e9);
  // What each ball was last heard doing: its state, the sector of the net it was in, how many lines it had crossed.
  // `at` is when that was last brought up to date, `last` when it last sounded.
  const heard = Array.from({ length: NB }, () => ({ state: 0, sector: 0, crossed: 0, at: 0, last: 0 }));
  let reading = false;
  let heardFrom = null;                     // the camera, as the ear: { viewProj, originRow }
  let notes = 0;

  /** Where a point of the land sits in the picture (-1 left, 1 right) and how near it is (0..1). */
  function hear(s, y, t) {
    if (!heardFrom) return { pan: 0, near: 0.5 };
    const m = heardFrom.viewProj, z = t - heardFrom.originRow * DT;
    const x = m[0] * s + m[4] * y + m[8] * z + m[12];
    const w = m[3] * s + m[7] * y + m[11] * z + m[15];
    if (w < 0.1) return { pan: 0, near: 0.15 };
    const pan = clamp(x / w, -1.3, 1.3);
    return { pan, near: (Math.abs(pan) > 1.05 ? 0.5 : 1) / (1 + (w / 16) ** 2) };
  }

  function takeSlot() {
    // The lowest empty slot (unless it was only just handed out), so the slots in
    // use stay packed at the front; if every slot is full, the oldest ball goes.
    let best = -1, oldest = 0;
    for (let i = 0; i < NB; i++) {
      if (best < 0 && slotState[i] === 0 && time - slotBorn[i] > 1) best = i;
      if (slotBorn[i] < slotBorn[oldest]) oldest = i;
    }
    if (best < 0) best = oldest;
    slotBorn[best] = time;
    slotState[best] = 1;
    Object.assign(heard[best], { state: 1, sector: 0, crossed: 0, at: time });
    return best;
  }
  /** One past the highest slot that may hold a ball. */
  function slotsUsed() {
    let n = 0;
    for (let i = 0; i < NB; i++) if (slotState[i] > 0 || time - slotBorn[i] < 1) n = i + 1;
    return n;
  }

  /** Drop a ball over the point s + i t. */
  function drop(s, t) {
    if (drops.length < 16) drops.push({ s, t });
  }

  function readBalls() {
    if (reading) return;
    reading = true;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(ballBuf, 0, ballRead, 0, NB * BALL_BYTES);
    device.queue.submit([enc.finish()]);
    const origin = originRow;
    ballRead.mapAsync(GPUMapMode.READ).then(() => {
      const f = new Float32Array(ballRead.getMappedRange());
      for (let i = 0; i < NB; i++) {
        if (time - slotBorn[i] < 0.3) continue;       // this may still be the slot's previous ball
        const o = i * BALL_FLOATS;
        const state = f[o + 3];
        slotState[i] = state;
        listen(i, state, f, o, origin);
      }
      if (newest && time - slotBorn[newest.slot] > 0.3) {
        const o = newest.slot * BALL_FLOATS;
        newest.state = f[o + 3];
        if (newest.state > 0) { newest.s = f[o]; newest.t = f[o + 1] + origin * DT; }
      }
      ballRead.unmap();
      reading = false;
    }).catch(() => { reading = false; });
  }

  /** A ring on the land where a ball has just come down. */
  function mark(s, t) {
    if (time - lastMark <= 0.55) return;
    lastMark = time;
    hits.push({ s, t, born: time, strength: 0.5, turn: markCount++ % 2 ? 1 : -1 });
    if (hits.length > HITS) hits.shift();
  }

  /** Turn what one ball has done since it was last read into sound. */
  function listen(i, state, f, o, origin) {
    const was = heard[i];
    const s = f[o], t = f[o + 1] + origin * DT, y = f[o + 2];
    const sector = f[o + 16], crossed = f[o + 17], band = f[o + 18];
    if (was.state === 1 && state >= 2) {
      // It has come down. The GPU says when and into which cell of the net, so
      // the ring and the sound stay with the ball however the frames are running.
      mark(s, t);
      sound.plip({ line: sector, band, ...hear(s, y, t) });
      notes++;
    }
    let caughtUp = true;
    if (state === 2 && was.state === 2 && crossed > was.crossed) {
      if (time - was.last > 0.03) {
        // It crossed lines of the net: one note each, in the order it met them, spread over the time since we last looked.
        const { pan, near } = hear(s, y, t);
        const speed = Math.hypot(f[o + 4], f[o + 5]);
        const count = Math.min(crossed - was.crossed, 3);
        const since = clamp(time - was.at, 0.005, 0.12);
        const turn = (((sector - was.sector) % 12) + 18) % 12 - 6;     // -6..5, the short way round
        const dir = turn === 0 ? 0 : Math.sign(turn);
        let at = sector - dir * count;
        for (let k = 0; k < count; k++) {
          const line = dir >= 0 ? at + 1 : at;
          at += dir;
          sound.pluck({ line, band, speed, register: i, pan, near, after: (k * since) / count });
          notes++;
        }
        was.last = time;
      } else caughtUp = false;       // too soon after its last notes: these crossings wait for the next look
    }
    if (state === 3 && (was.state === 1 || was.state === 2) && band <= -2) {
      // A zero has it.
      sound.bell({ gamma: t, ...hear(s, y, t) });
      notes++;
    }
    was.state = state;
    if (caughtUp) { was.sector = sector; was.crossed = crossed; was.at = time; }
  }

  /** Find the point on the land under a pixel, from the distance the frame just drew there, and drop a ball on it. */
  function pick(enc, px, py, cam, origin) {
    const buf = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyTextureToBuffer({ texture: pickTex, origin: [clamp(px, 0, W - 1), clamp(py, 0, H - 1)] }, { buffer: buf }, [1, 1]);
    return () => buf.mapAsync(GPUMapMode.READ).then(() => {
      const z = new Float32Array(buf.getMappedRange())[0];
      buf.destroy();
      if (!(z > 0)) return;                // sky
      const nx = (2 * (px + 0.5)) / cam.W - 1, ny = 1 - (2 * (py + 0.5)) / cam.H;
      const kx = nx * cam.tanF * cam.aspect, ky = ny * cam.tanF;
      const s = cam.eye[0] + z * (cam.f[0] + cam.r[0] * kx + cam.u[0] * ky);
      const zz = cam.eye[2] + z * (cam.f[2] + cam.r[2] * kx + cam.u[2] * ky);
      drop(s, zz + origin * DT);
    }).catch(() => {});
  }

  // --- input ---------------------------------------------------------------
  // Left button: drop a ball; keep it down and they keep coming, wherever the
  // pointer goes. Right button, or shift: turn and tip the view. Wheel: travel
  // along the line; with ctrl (a pinch on a trackpad), move in and out.
  const pointer = { x: 0, y: 0, pouring: false, turning: false, lastPour: 0 };
  const pixel = (x, y) => {
    const k = W / canvas.clientWidth;
    return { x: Math.floor(x * k), y: Math.floor(y * k) };
  };
  canvas.addEventListener('pointerdown', (e) => {
    sound.start();
    canvas.setPointerCapture(e.pointerId);
    pointer.x = e.clientX; pointer.y = e.clientY;
    if (e.button === 0 && !e.shiftKey) {
      pointer.pouring = true;
      pointer.lastPour = time;
      picks.push(pixel(e.clientX, e.clientY));
      hint.classList.remove('on');
    } else pointer.turning = true;
  });
  canvas.addEventListener('pointermove', (e) => {
    if (pointer.turning) {
      want.yaw += (e.clientX - pointer.x) * 0.005;
      want.pitch = clamp(want.pitch + (e.clientY - pointer.y) * 0.004, 0.16, 1.45);
    }
    pointer.x = e.clientX; pointer.y = e.clientY;
  });
  const release = () => { pointer.pouring = false; pointer.turning = false; };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const unit = e.deltaMode === 1 ? 32 : 1;
    if (e.ctrlKey) want.dist = clamp(want.dist * Math.exp(e.deltaY * unit * 0.01), 3.5, 34);
    else {
      want.t = clamp(want.t - e.deltaY * unit * 0.0012 * want.dist, 2, T_MAX - 40);
      want.yaw += e.deltaX * unit * 0.0015;
    }
  }, { passive: false });
  window.addEventListener('keydown', (e) => {
    if (e.code.startsWith('Arrow') || e.code === 'Space') e.preventDefault();
    if (e.code === 'KeyM' && !e.repeat) sound.toggle(); else sound.start();
    keys.add(e.code);
    const place = PLACES[parseInt(e.key, 10) - 1];
    if (place !== undefined && !e.repeat) { pendingPlace = place; fadeTarget = 0; }
  });
  window.addEventListener('keyup', (e) => keys.delete(e.code));
  window.addEventListener('blur', () => { keys.clear(); release(); });

  placard.classList.add('on');
  hint.classList.add('on');
  setTimeout(() => hint.classList.remove('on'), 16000);

  // A few balls to begin with, so the first thing you see is what they do.
  const opening = [[2.6, 15.4, 1.6], [-1.6, 19.2, 2.3], [2.2, 23.6, 3.0], [-1.2, 27.4, 3.8], [3.0, 31.4, 4.6]];

  // --- frame ---------------------------------------------------------------
  let last = performance.now(), frameNo = 0, fpsEma = 60, textClock = 0;

  function frame(now) {
    const dtReal = clamp((now - last) / 1000, 0.001, 0.1);
    last = now;
    time += dtReal;
    fpsEma += (1 / dtReal - fpsEma) * 0.05;
    resize();

    // A jump between places is wrapped in a fade to black so the cut is never seen.
    const rate = fadeTarget < fade ? 4 : 0.9;
    fade += Math.sign(fadeTarget - fade) * Math.min(Math.abs(fadeTarget - fade), rate * dtReal);
    if (pendingPlace !== null && fade <= 0.001) {
      want.t = view.t = pendingPlace;
      hits = []; drops = []; newest = null;
      opening.length = 0;
      pendingPlace = null;
      fadeTarget = 1;
      respawnAll = true;
    }

    // Camera.
    const key = (a, b) => (keys.has(a) || keys.has(b) ? 1 : 0);
    want.t = clamp(want.t + (key('ArrowUp', 'KeyW') - key('ArrowDown', 'KeyS')) * dtReal * 0.6 * want.dist, 2, T_MAX - 40);
    want.yaw += (key('ArrowRight', 'KeyD') - key('ArrowLeft', 'KeyA')) * dtReal * 0.9;
    want.dist = clamp(want.dist * Math.exp((key('Minus', 'KeyQ') - key('Equal', 'KeyE')) * dtReal * 1.2), 3.5, 34);
    const k = ease(dtReal, calm ? 0.05 : 0.22);
    view.t += (want.t - view.t) * k;
    view.yaw += (want.yaw - view.yaw) * k;
    view.pitch += (want.pitch - view.pitch) * k;
    view.dist += (want.dist - view.dist) * k;

    while (opening.length && time > opening[0][2]) { const [s, t] = opening.shift(); drop(s, t); }
    if (pointer.pouring && time - pointer.lastPour > POUR_EVERY && picks.length < 2) {
      pointer.lastPour = time;
      picks.push(pixel(pointer.x, pointer.y));
    }

    hits = hits.filter((h) => time - h.born < 6);

    const { d0, zShift } = updateWindow(dtReal);

    const zFocus = view.t - originRow * DT;
    const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
    const f = [Math.sin(view.yaw) * cp, -sp, Math.cos(view.yaw) * cp];
    const eye = [0.5 - f[0] * view.dist, Math.max(-f[1] * view.dist, 1.3), zFocus - f[2] * view.dist];
    const cam = viewFrom(eye, f);
    const aspect = W / H, tanF = Math.tan(FOV / 2);
    const viewProj = mul(perspective(FOV, aspect, NEAR, FAR), cam.m);

    // Hand the waiting balls to the GPU, four a frame at most.
    dropsU.fill(0);
    for (let i = 0; i < 4; i++) {
      const d = drops.shift();
      if (!d || d.t < (validLo + 4) * DT || d.t > (validHi - 5) * DT) { dropsU[i * 4 + 3] = -1; continue; }
      const slot = takeSlot();
      dropsU.set([d.s, d.t - originRow * DT, DROP, slot], i * 4);
      dropsU.set([dropCount * 2.399963, 0.075 + 0.03 * ((dropCount * 0.618034) % 1), 0, 0], 16 + i * 4);
      dropCount++;
      newest = { slot, s: d.s, t: d.t, state: 1 };
    }
    device.queue.writeBuffer(dropsBuf, 0, dropsU);

    const simDt = Math.min(dtReal, 1 / 30);
    gF.set(viewProj, 0);
    gF.set([eye[0], eye[1], eye[2], time], 16);
    gF.set([0, (H / 2) / tanF, DT, HEIGHT], 20);
    gF.set([validLo - originRow, validHi - originRow, d0 - originRow, simDt], 24);
    gF.set([Math.cos(look.swirl), Math.sin(look.swirl), simDt * look.flow, zShift], 28);
    gF.set([simDt * 3.2, simDt / 6.5, look.streak, look.size * (H / 2) / tanF], 32);
    gF.set([W, H, look.gain, look.gridGain], 36);
    gF.set([0, 0, filling > 0 ? 1 : 0, 0], 40);
    gU.set([mod(originRow, FH), frameNo, PARTICLES, respawnAll ? 1 : 0], 44);
    gF.set([0.5, zFocus, 0, 0], 48);
    let turns = 0;
    hits.forEach((h, i) => { if (h.turn < 0) turns |= 1 << i; });
    gF.set([slotsUsed(), 0, turns, 0], 52);
    for (let i = 0; i < HITS; i++) {
      const h = hits[i];
      gF.set(h ? [h.s, h.t - originRow * DT, time - h.born, h.strength] : [0, 0, 0, 0], 56 + i * 4);
    }
    device.queue.writeBuffer(globalsBuf, 0, globals);
    postU.set([look.exposure, look.bloom, time, fade * fade, ...cam.r, tanF * aspect, ...cam.u, tanF, ...cam.f, 0]);
    device.queue.writeBuffer(postUBuf, 0, postU);
    respawnAll = false;

    const enc = device.createCommandEncoder();
    {
      const pass = enc.beginComputePass();
      pass.setPipeline(ballsPipe);
      pass.setBindGroup(0, ballsBG);
      pass.dispatchWorkgroups(Math.ceil(NB / 16));
      pass.end();
    }
    {
      const pass = enc.beginComputePass();
      pass.setPipeline(simPipe);
      pass.setBindGroup(0, simBG);
      pass.dispatchWorkgroups(Math.ceil(PARTICLES / 256));
      pass.end();
    }
    {
      const pass = enc.beginRenderPass({
        colorAttachments: [
          { view: hdrView, clearValue: [0, 0, 0, 0], loadOp: 'clear', storeOp: 'store' },
          { view: pickView, clearValue: [0, 0, 0, 0], loadOp: 'clear', storeOp: 'store' },
        ],
        depthStencilAttachment: { view: depthView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
      });
      pass.setPipeline(surfacePipe);
      pass.setBindGroup(0, surfaceBG);
      pass.draw(MW * MH * 6);
      pass.setPipeline(ballPipe);
      pass.setBindGroup(0, ballDrawBG);
      pass.draw(4, NB);
      pass.setPipeline(particlePipe);
      pass.setBindGroup(0, particleBG);
      pass.draw(4, PARTICLES);
      pass.setPipeline(trackPipe);
      pass.setBindGroup(0, ballDrawBG);
      pass.draw(4, NB * TRAIL);
      pass.end();
    }
    // Clicks become points on the land by reading the distance just drawn under them.
    const camNow = { eye, r: cam.r, u: cam.u, f: cam.f, tanF, aspect, W, H };
    const after = picks.splice(0, 2).map((p) => pick(enc, p.x, p.y, camNow, originRow));

    const blit = (pipe, bg, target, load) => {
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: target, loadOp: load ? 'load' : 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }] });
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    };
    blit(prefilterPipe, prefilterBG, mips[0]);
    for (let i = 1; i < BLOOM_LEVELS; i++) blit(downPipe, blurBGs[i - 1], mips[i]);
    for (let i = BLOOM_LEVELS - 1; i > 0; i--) blit(upPipe, blurBGs[i], mips[i - 1], true);
    blit(compositePipe, compositeBG, context.getCurrentTexture().createView());
    device.queue.submit([enc.finish()]);
    for (const go of after) go();
    heardFrom = { viewProj, originRow };
    readBalls();
    sound.place(view.t);

    // The placard follows the ball last dropped; as it circles a zero you can watch it close in.
    textClock += dtReal;
    if (textClock > 0.1) {
      textClock = 0;
      const following = newest && newest.state > 0;
      const s = following ? newest.s : 0.5, t = following ? newest.t : view.t;
      const digits = following ? 3 : 2;
      sEl.textContent = s.toFixed(digits).replace('-', '−');
      tEl.textContent = t.toFixed(digits).replace('-', '−');
    }
    frameNo++;
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // --- for testing, and for checking the GPU field against the CPU reference
  window.__zeta = {
    want, view, look, drop,
    click: (x, y) => picks.push(pixel(x, y)),
    clearRings: () => { hits = []; },
    sound,
    stats: () => ({ notes, voices: sound.voices, sounding: sound.running, fps: fpsEma, validLo, validHi, originRow, W, H, filling, hits: hits.length, balls: slotState.reduce((n, s) => n + (s > 0 ? 1 : 0), 0), newest }),
    async check(t) {
      const row = Math.round(t / DT);
      if (row < validLo || row >= validHi) return { error: 'row not computed', validLo, validHi, row };
      const buf = device.createBuffer({ size: FW * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = device.createCommandEncoder();
      enc.copyTextureToBuffer({ texture: fieldTex, origin: [0, mod(row, FH)] }, { buffer: buf, bytesPerRow: FW * 16 }, [FW, 1]);
      device.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(buf.getMappedRange().slice(0));
      buf.destroy();
      // Compared only where the double-precision reference is itself reliable.
      let worst = 0, worstD = 0, at = 0;
      for (let i = 0; i < FW; i++) {
        const sigma = sigmaOfCol(i);
        if (sigma < -8) continue;
        const ref = zeta(sigma, row * DT || 2e-3);
        if (Math.hypot(ref[0], ref[1]) > 1e24) continue;   // the GPU holds sizes this large at e^60 on purpose
        const err = Math.hypot(data[4 * i] - ref[0], data[4 * i + 1] - ref[1]) / Math.max(Math.hypot(ref[0], ref[1]), 1e-3);
        const derr = Math.hypot(data[4 * i + 2] - ref[2], data[4 * i + 3] - ref[3]) / Math.max(Math.hypot(ref[2], ref[3]), 1e-3);
        if (err > worst) { worst = err; at = sigma; }
        worstD = Math.max(worstD, derr);
      }
      return { t: row * DT, worstRelErr: worst, worstAtSigma: at, worstRelErrDeriv: worstD };
    },
  };
}

start().catch((err) => {
  console.error(err);
  fail(`The scene could not start: ${err.message}`);
});
