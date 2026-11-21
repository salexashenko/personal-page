// Riemann zeta in double precision on the CPU. The picture is drawn entirely
// on the GPU; this is the trusted reference its field is checked against.

// B_2k / (2k)! for k = 1..8
const BERN = [
  1 / 12,
  -1 / 720,
  1 / 30240,
  -1 / 1209600,
  1 / 47900160,
  -691 / 1307674368000,
  1 / 74724249600,
  -3617 / 10670622842880000,
];

/** Number of Dirichlet terms the Euler–Maclaurin sum uses at height t. */
export function termsFor(tAbs) {
  return Math.ceil(0.5 * Math.abs(tAbs) + 16);
}

/**
 * zeta(sigma + i t) and its derivative by Euler–Maclaurin summation.
 * Returns [re, im, dre, dim].
 */
export function zeta(sigma, t, N = termsFor(t) + 8) {
  if (Math.abs(t) < 1e-9) t = 1e-9;
  let sr = 0, si = 0, dr = 0, di = 0;
  for (let n = 1; n < N; n++) {
    const ln = Math.log(n);
    const m = Math.exp(-sigma * ln);
    const th = t * ln;
    const tr = m * Math.cos(th), ti = -m * Math.sin(th);
    sr += tr; si += ti;
    dr -= ln * tr; di -= ln * ti;
  }
  const lnN = Math.log(N);
  const mN = Math.exp(-sigma * lnN);
  const thN = t * lnN;
  const pr = mN * Math.cos(thN), pi = -mN * Math.sin(thN); // N^-s

  // T0 = N^(1-s) / (s-1)
  const ar = sigma - 1, ai = t;
  const ad = ar * ar + ai * ai;
  const t0r = N * (pr * ar + pi * ai) / ad;
  const t0i = N * (pi * ar - pr * ai) / ad;
  // T0 / (s-1)
  const qr = (t0r * ar + t0i * ai) / ad;
  const qi = (t0i * ar - t0r * ai) / ad;

  let zr = sr + t0r + 0.5 * pr;
  let zi = si + t0i + 0.5 * pi;
  let zpr = dr - lnN * t0r - qr - 0.5 * lnN * pr;
  let zpi = di - lnN * t0i - qi - 0.5 * lnN * pi;

  // P = s (s+1) ... (s+2k-2) N^(-s-2k+1),  D = sum 1/(s+j)
  let Pr = (sigma * pr - t * pi) / N;
  let Pi = (sigma * pi + t * pr) / N;
  const sd = sigma * sigma + t * t;
  let Dr = sigma / sd, Di = -t / sd;
  for (let k = 0; k < BERN.length; k++) {
    const c = BERN[k];
    zr += c * Pr; zi += c * Pi;
    const er = Dr - lnN, ei = Di;
    zpr += c * (Pr * er - Pi * ei);
    zpi += c * (Pr * ei + Pi * er);
    const a = sigma + 2 * k + 1, b = sigma + 2 * k + 2;
    // P *= (s+2k+1)(s+2k+2) / N^2
    const xr = Pr * a - Pi * t, xi = Pr * t + Pi * a;
    Pr = (xr * b - xi * t) / (N * N);
    Pi = (xr * t + xi * b) / (N * N);
    const da = a * a + t * t, db = b * b + t * t;
    Dr += a / da + b / db;
    Di -= t / da + t / db;
  }
  return [zr, zi, zpr, zpi];
}
