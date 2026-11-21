// The music. The net drawn on the land is the instrument: its lines are equal
// steps of arg zeta, twelve to a turn, and a ball plucks a note each time it
// crosses one. Around a zero the twelve lines meet, so a ball circling a zero
// plays an arpeggio, up six notes and back down six, faster as the circle
// tightens and higher as it sinks. When the zero takes the ball, the zero
// rings: each has its own low bell. Everything is in one pentatonic key over
// a quiet drone, so any number of balls sound well together.

const PENT = [0, 2, 4, 7, 9];
const ROOT = 146.83;            // D3
/** Frequency of the n-th note of the scale, counted from the root. */
const note = (n) => ROOT * 2 ** ((12 * Math.floor(n / 5) + PENT[((n % 5) + 5) % 5]) / 12);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
/** The twelve lines round a zero, folded so the notes go up six and come back. */
const fold = (line) => { const m = ((line % 12) + 12) % 12; return m <= 6 ? m : 12 - m; };

const MAX_VOICES = 44;

export function createSound() {
  let ctx = null, master = null, dry = null, wet = null, echo = null;
  let muted = false, voices = 0, colour = [], colourNow = -1, meter = null;

  /** A few seconds of decaying noise: the room everything is heard in. */
  function room(seconds) {
    const n = Math.floor(ctx.sampleRate * seconds);
    const buf = ctx.createBuffer(2, n, ctx.sampleRate);
    for (let c = 0; c < 2; c++) {
      const d = buf.getChannelData(c);
      let smooth = 0;
      for (let i = 0; i < n; i++) {
        const t = i / n;
        // Darker as it dies away, as a real room is.
        const k = 0.25 + 0.6 * t;
        smooth += (Math.random() * 2 - 1 - smooth) * (1 - k);
        d[i] = smooth * Math.pow(1 - t, 2.4) * (i < 400 ? i / 400 : 1);
      }
    }
    return buf;
  }

  function start() {
    if (ctx) { if (ctx.state === 'suspended') ctx.resume(); return; }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    ctx = new AC();

    master = ctx.createGain();
    master.gain.value = 0;
    master.gain.setTargetAtTime(muted ? 0 : 0.9, ctx.currentTime, 0.4);
    const limit = ctx.createDynamicsCompressor();
    limit.threshold.value = -16; limit.knee.value = 14; limit.ratio.value = 5;
    limit.attack.value = 0.004; limit.release.value = 0.3;
    master.connect(limit).connect(ctx.destination);
    meter = ctx.createAnalyser();
    meter.fftSize = 2048;
    limit.connect(meter);

    dry = ctx.createGain();
    dry.connect(master);

    wet = ctx.createGain();
    const pre = ctx.createBiquadFilter(); pre.type = 'lowpass'; pre.frequency.value = 5200;
    const hall = ctx.createConvolver(); hall.buffer = room(4.2);
    const post = ctx.createBiquadFilter(); post.type = 'highpass'; post.frequency.value = 160;
    const hallGain = ctx.createGain(); hallGain.gain.value = 0.9;
    wet.connect(pre).connect(hall).connect(post).connect(hallGain).connect(master);

    // One soft echo, a dotted beat behind, feeding the room.
    echo = ctx.createGain();
    const delay = ctx.createDelay(1);
    delay.delayTime.value = 0.42;
    const back = ctx.createGain(); back.gain.value = 0.32;
    const dull = ctx.createBiquadFilter(); dull.type = 'lowpass'; dull.frequency.value = 2400;
    echo.connect(delay).connect(dull).connect(back).connect(delay);
    dull.connect(wet);
    const echoOut = ctx.createGain(); echoOut.gain.value = 0.5;
    dull.connect(echoOut).connect(master);

    drone();
  }

  /** The ground under the music: the root and its fifth, barely moving, and one colour note that changes as you travel. */
  function drone() {
    const bus = ctx.createGain();
    bus.gain.value = 0;
    bus.gain.setTargetAtTime(1, ctx.currentTime + 0.5, 3);
    const soft = ctx.createBiquadFilter(); soft.type = 'lowpass'; soft.frequency.value = 900;
    bus.connect(soft);
    soft.connect(dry);
    const send = ctx.createGain(); send.gain.value = 0.7;
    soft.connect(send).connect(wet);

    const voice = (f, level, type, breathe) => {
      const o = ctx.createOscillator(); o.type = type; o.frequency.value = f;
      const g = ctx.createGain(); g.gain.value = level;
      // A slow breath on each voice, each at its own pace.
      const lfo = ctx.createOscillator(); lfo.frequency.value = breathe;
      const depth = ctx.createGain(); depth.gain.value = level * 0.45;
      lfo.connect(depth).connect(g.gain);
      o.connect(g).connect(bus);
      o.start(); lfo.start();
      return g;
    };
    voice(note(-5), 0.05, 'sine', 0.05);          // D2
    voice(note(-5) * 1.003, 0.018, 'triangle', 0.071);
    voice(note(-2), 0.03, 'sine', 0.083);         // A2
    voice(note(0), 0.016, 'sine', 0.061);         // D3
    // Third, sixth, second: only one sounds at a time.
    colour = [2, 4, 6].map((n, i) => {
      const g = voice(note(n), 0.0001, 'sine', 0.09 + 0.02 * i);
      return g;
    });
  }

  /** Where along the line you are decides which colour note the drone holds. */
  function place(t) {
    if (!ctx) return;
    const want = Math.floor(Math.abs(t) / 9) % 3;
    if (want === colourNow) return;
    colourNow = want;
    colour.forEach((g, i) => g.gain.setTargetAtTime(i === want ? 0.014 : 0.0001, ctx.currentTime, 2.5));
  }

  /** One sounding thing: its own pan, some of it into the room, some into the echo. */
  function out(pan, level, room, echoes) {
    const p = ctx.createStereoPanner ? ctx.createStereoPanner() : ctx.createGain();
    if (p.pan) p.pan.value = clamp(pan, -1, 1) * 0.85;
    const g = ctx.createGain(); g.gain.value = level;
    p.connect(g);
    g.connect(dry);
    const s = ctx.createGain(); s.gain.value = room; g.connect(s).connect(wet);
    if (echoes > 0) { const e = ctx.createGain(); e.gain.value = echoes; g.connect(e).connect(echo); }
    return p;
  }

  /** A sine that starts at once and dies away. */
  function partial(dest, f, when, peak, decay, attack = 0.004) {
    const o = ctx.createOscillator(); o.type = 'sine'; o.frequency.value = f;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, when);
    g.gain.linearRampToValueAtTime(peak, when + attack);
    g.gain.setTargetAtTime(0, when + attack, decay / 4.6);
    o.connect(g).connect(dest);
    o.start(when);
    o.stop(when + attack + decay * 1.6);
    return o;
  }

  function claim(seconds) {
    if (!ctx || ctx.state !== 'running' || voices >= MAX_VOICES) return false;
    voices++;
    setTimeout(() => { voices--; }, seconds * 1000);
    return true;
  }

  /**
   * A ball crossing a line of the net.
   * line: which of the twelve (any integer), band: how deep in log|zeta| (lower is deeper),
   * speed: how fast the ball is going, register: 0, 1 or 2 for the ball's own octave,
   * pan: -1..1 across the picture, near: 0..1 how close it is, after: seconds from now.
   */
  function pluck({ line, band, speed, register, pan, near, after = 0 }) {
    const n = clamp(5 + fold(line) + clamp(-band, -5, 6) + [0, 0, 5, -5][register % 4], -5, 18);
    const f = note(n);
    const decay = clamp(1.5 * Math.pow(220 / f, 0.45), 0.3, 2.2);
    if (!claim(decay)) return;
    const when = ctx.currentTime + 0.025 + after;
    const hard = clamp(0.35 + speed / 4, 0.35, 1);
    const level = 0.2 * hard * (0.25 + 0.75 * near) * Math.pow(330 / Math.max(f, 330), 0.5);
    const dest = out(pan, level, 0.5 + 0.4 * (1 - near), 0.22);
    partial(dest, f, when, 1, decay);
    partial(dest, f * 2.004, when, 0.3 * hard, decay * 0.4);
    partial(dest, f * 5.4, when, 0.1 * hard, 0.06, 0.002);   // the tick of the tine
  }

  /** A ball landing: a drop of water, in tune. */
  function plip({ line, band, pan, near, after = 0 }) {
    if (!claim(0.5)) return;
    const f = note(clamp(10 + fold(line) + clamp(-band, -3, 3), 5, 18));
    const when = ctx.currentTime + 0.02 + after;
    const dest = out(pan, 0.09 * (0.3 + 0.7 * near), 0.8, 0.3);
    const o = partial(dest, f, when, 1, 0.28, 0.003);
    o.frequency.setValueAtTime(f * 0.62, when);
    o.frequency.exponentialRampToValueAtTime(f, when + 0.055);
  }

  /** A zero taking a ball. gamma: the zero's height on the line, which picks its bell. */
  function bell({ gamma, pan, near }) {
    if (!ctx || ctx.state !== 'running') return;
    const f = note(-4 + (Math.floor(Math.abs(gamma)) % 7));
    const when = ctx.currentTime + 0.03;
    const dest = out(pan * 0.6, 0.2 * (0.35 + 0.65 * near), 1.1, 0.15);
    partial(dest, f, when, 1, 6.5, 0.012);
    partial(dest, f * 2.0, when, 0.45, 4.5, 0.01);
    partial(dest, f * 2.76, when, 0.3, 3.2, 0.008);
    partial(dest, f * 5.4, when, 0.12, 1.4, 0.004);
    // And the last glint of the ball going, two octaves up.
    partial(dest, f * 8, when, 0.1, 0.9, 0.003);
  }

  function toggle() {
    muted = !muted;
    if (master) master.gain.setTargetAtTime(muted ? 0 : 0.9, ctx.currentTime, 0.15);
    return muted;
  }

  document.addEventListener('visibilitychange', () => {
    if (!ctx) return;
    if (document.hidden) ctx.suspend(); else ctx.resume();
  });

  return {
    start, place, pluck, plip, bell, toggle,
    get running() { return !!ctx && ctx.state === 'running'; },
    get voices() { return voices; },
    /** Peak of what is going out right now, 0..1, for checking levels. */
    peak() {
      if (!meter) return 0;
      const d = new Float32Array(meter.fftSize);
      meter.getFloatTimeDomainData(d);
      return d.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
    },
  };
}
