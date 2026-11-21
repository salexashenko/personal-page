// The Riemann–Siegel formula, which the field uses high up the line, where
// Euler–Maclaurin would need t/2 terms a point and this needs sqrt(t / 2 pi).
//
//   zeta(s) = R(s) + chi(s) conj(R(1 - conj(s)))
//
// and, with a = sqrt(t / 2 pi), N = floor(a), x = a - N,
//
//   R(s) = sum_{n <= N} n^-s  +  (-1)^N a^-s e^(i pi a^2) (c0 + c1 / a + c2 / a^2 + ...)
//
// The second part is Siegel's integral of z^-s e^(i pi z^2) / (e^(i pi z) - e^(-i pi z))
// down the diagonal through its saddle at a. Writing z = a + w and expanding
// z^-s in powers of w / a leaves, at each order, a polynomial in w and sigma
// against one fixed kernel, so every c_k is a combination of the moments
//
//   M_m(x) = integral of w^m e^(2 pi i w^2) / (2i sin(pi (x + w))) dw
//
//   c0 = M0
//   c1 = -sigma M1 - (2 pi i / 3) M3
//   c2 = sigma (sigma + 1) / 2 M2 + i pi (1/2 + 2 sigma / 3) M4 - (2 pi^2 / 9) M6
//
// The moments depend only on x, are smooth on [0, 1], and are worked out here
// once, in double precision, as Chebyshev series for the shader to evaluate.
// Stopping at c2 is good to a few parts in a million at t = 200 and far
// better above; the shader starts using it at T_RS.

export const T_RS = 400;          // above this height the field is summed by Riemann–Siegel
export const SIEGEL_TERMS = 16;   // Chebyshev terms kept per moment

const ORDERS = [0, 1, 2, 3, 4, 6];
// Each moment is stored already multiplied by the constant it carries in c1 and c2.
const WEIGHTS = [[1, 0], [1, 0], [1, 0], [0, -2 * Math.PI / 3], [0, Math.PI], [-2 * Math.PI * Math.PI / 9, 0]];

/** M_m(x) for each order, by the trapezium rule down the diagonal midway between the poles at -x and 1 - x. */
function moments(x) {
  const h = 0.02, reach = 4.5, r = Math.SQRT1_2;
  const sums = ORDERS.map(() => [0, 0]);
  for (let u = -reach; u <= reach + 1e-9; u += h) {
    const wr = 0.5 - x + r * u, wi = r * u;
    // e^(2 pi i w^2)
    const er = Math.exp(-4 * Math.PI * wr * wi), ea = 2 * Math.PI * (wr * wr - wi * wi);
    const nr = er * Math.cos(ea), ni = er * Math.sin(ea);
    // 2i sin(pi (x + w))
    const pr = Math.PI * (x + wr), pi = Math.PI * wi;
    const dr = -2 * Math.cos(pr) * Math.sinh(pi), di = 2 * Math.sin(pr) * Math.cosh(pi);
    const dd = dr * dr + di * di;
    let fr = (nr * dr + ni * di) / dd, fi = (ni * dr - nr * di) / dd;
    let order = 0;
    ORDERS.forEach((m, k) => {
      for (; order < m; order++) [fr, fi] = [fr * wr - fi * wi, fr * wi + fi * wr];
      sums[k][0] += fr; sums[k][1] += fi;
    });
  }
  // The path runs downward: dw = -e^(i pi / 4) du.
  return sums.map(([sr, si]) => [-h * r * (sr - si), -h * r * (sr + si)]);
}

/**
 * Chebyshev coefficients of the six weighted moments on x in [0, 1], as
 * [moment][term] = [re, im]. The series is c[0] + sum c[k] T_k(2x - 1).
 */
export function siegelTables(terms = SIEGEL_TERMS) {
  const n = 48;
  const samples = [];
  for (let j = 0; j < n; j++) samples.push(moments(0.5 + 0.5 * Math.cos((Math.PI * (j + 0.5)) / n)));
  return ORDERS.map((_, m) => {
    const [wr, wi] = WEIGHTS[m];
    const out = [];
    for (let k = 0; k < terms; k++) {
      let cr = 0, ci = 0;
      for (let j = 0; j < n; j++) {
        const c = Math.cos((Math.PI * k * (j + 0.5)) / n);
        cr += samples[j][m][0] * c; ci += samples[j][m][1] * c;
      }
      const norm = (k === 0 ? 1 : 2) / n;
      out.push([norm * (cr * wr - ci * wi), norm * (cr * wi + ci * wr)]);
    }
    return out;
  });
}
