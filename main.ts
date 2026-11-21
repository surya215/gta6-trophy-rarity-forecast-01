// GTA VI trophy earn-rate forecaster — Team A: numerator / denominator decomposition.
//
// Idea: an earn rate is a ratio, r = 100 * E / P, where E is how many tracked
// players have earned the trophy and P is how many players Exophase tracks.
// Exophase refreshes its counts on its own schedule (hours apart), so the
// history looks like a staircase: long flat stretches, then a jump in P that
// dilutes every rate at once, while E keeps growing as people play. Fitting
// the rate directly confuses those two motions. Instead this program:
//
//   1. reconstructs E_i(t) = rate_i(t) * P(t) / 100 and P(t) from the history;
//   2. fits each of them in log-log time, ln E and ln P against ln(1 + hours
//      since the first read), with the fitted slope shrunk toward a tier /
//      story prior when the signal is thin;
//   3. forecasts E_i and P separately to the target and divides;
//   4. puts quantiles on the logit scale with a spread that combines a
//      random-walk term, the fit residuals, the size of refresh jumps seen
//      in the history, and trend uncertainty that grows with the horizon.
//
// Everything is deterministic (no randomness, no clock, no network) and uses
// only the Node standard library.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

// Standard-normal quantiles at the scored levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95.
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

type Trophy = { id: number; name: string; description: string; tier: string };
type Read = { at: string; players_tracked: number; rates: Array<number | null> };
type CaseFile = {
  id: string;
  horizon_minutes: number;
  target: string;
  trophies: Trophy[];
  history: Read[];
};

const clamp = (x: number, lo: number, hi: number): number => (x < lo ? lo : x > hi ? hi : x);
const clampPct = (x: number): number => clamp(x, 0.01, 99.99);
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-clamp(l, -12, 12)));
const round4 = (x: number): number => Math.round(x * 10000) / 10000;

/** Ordinary least squares of y on x. Returns slope, intercept and residual sd. */
function ols(xs: number[], ys: number[]): { slope: number; intercept: number; residSd: number } {
  const n = xs.length;
  if (n === 0) return { slope: 0, intercept: 0, residSd: 0.5 };
  if (n === 1) return { slope: 0, intercept: ys[0], residSd: 0.5 };
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const slope = sxx > 1e-12 ? sxy / sxx : 0;
  const intercept = my - slope * mx;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (intercept + slope * xs[i]);
    ss += r * r;
  }
  const residSd = n > 2 ? Math.sqrt(ss / (n - 2)) : 0.25;
  return { slope, intercept, residSd };
}

/** Quadratic least squares y ~ a + b x + c x^2 (3x3 normal equations). */
function quadFit(xs: number[], ys: number[]): { a: number; b: number; c: number } | null {
  const n = xs.length;
  if (n < 5) return null;
  let s0 = n;
  let s1 = 0;
  let s2 = 0;
  let s3 = 0;
  let s4 = 0;
  let t0 = 0;
  let t1 = 0;
  let t2 = 0;
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    const x2 = x * x;
    s1 += x;
    s2 += x2;
    s3 += x2 * x;
    s4 += x2 * x2;
    t0 += ys[i];
    t1 += x * ys[i];
    t2 += x2 * ys[i];
  }
  // Solve [[s0,s1,s2],[s1,s2,s3],[s2,s3,s4]] [a,b,c] = [t0,t1,t2] by Gauss elimination.
  const m = [
    [s0, s1, s2, t0],
    [s1, s2, s3, t1],
    [s2, s3, s4, t2],
  ];
  for (let col = 0; col < 3; col++) {
    let piv = col;
    for (let r = col + 1; r < 3; r++) if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    if (Math.abs(m[piv][col]) < 1e-9) return null;
    const tmp = m[col];
    m[col] = m[piv];
    m[piv] = tmp;
    for (let r = col + 1; r < 3; r++) {
      const f = m[r][col] / m[col][col];
      for (let k = col; k < 4; k++) m[r][k] -= f * m[col][k];
    }
  }
  const c = m[2][3] / m[2][2];
  const b = (m[1][3] - m[1][2] * c) / m[1][1];
  const a = (m[0][3] - m[0][1] * b - m[0][2] * c) / m[0][0];
  if (![a, b, c].every(Number.isFinite)) return null;
  return { a, b, c };
}

/**
 * Prior log-log growth slope for a trophy's earner count, from what the
 * trophy is. Story chapters come early and are earned fast (earlier chapter
 * = faster); the platinum is earned by almost nobody in week one; gold and
 * silver sit in between. This is only a prior: real signal overrides it.
 */
function priorSlope(t: Trophy, lastRate: number): number {
  const chapter = /Chapter\s+(\d+)/i.exec(t.description ?? "");
  let base: number;
  if (chapter) {
    const n = parseInt(chapter[1], 10);
    base = clamp(1.05 - 0.09 * n, 0.25, 1.0);
  } else if (t.tier === "platinum") base = 0.3;
  else if (t.tier === "gold") base = 0.35;
  else if (t.tier === "silver") base = 0.5;
  else base = 0.65;
  // Near-saturated trophies cannot keep growing fast, whatever they are:
  // most of the pool already holds them, so their earner count tracks the
  // player count. (This cap applies to story chapters too.)
  if (lastRate > 40) base = Math.min(base, 0.35);
  else if (lastRate > 15) base = Math.min(base, 0.5);
  return base;
}

/** Prior median rate (percent) for a trophy never earned yet in the history. */
function priorRate(t: Trophy): number {
  if (t.tier === "platinum") return 0.05;
  if (t.tier === "gold") return 0.3;
  if (t.tier === "silver") return 0.8;
  return 1.5;
}

function forecastCase(c: CaseFile): Array<{ id: number; q: number[] }> {
  const hist = c.history;
  const last = hist[hist.length - 1];
  const t0 = Date.parse(hist[0].at);
  const hoursSinceFirst = (r: Read): number => Math.max(0, (Date.parse(r.at) - t0) / 3600000);
  const xOf = (r: Read): number => Math.log1p(hoursSinceFirst(r));
  const xLast = xOf(last);
  const xTarget = Math.log1p(Math.max(0, (Date.parse(c.target) - t0) / 3600000));
  // Fall back to horizon_minutes if target parsing ever fails.
  const xTgt = Number.isFinite(xTarget)
    ? xTarget
    : xLast + Math.log1p(c.horizon_minutes / 60);
  const horizonDays = c.horizon_minutes / 1440;
  const dx = Math.max(0, xTgt - xLast);

  // --- Collapse the staircase: keep only reads that change P or any rate. ---
  const distinct: Read[] = [];
  for (const r of hist) {
    const prev = distinct[distinct.length - 1];
    if (!prev) {
      distinct.push(r);
      continue;
    }
    const sameP = prev.players_tracked === r.players_tracked;
    const sameRates =
      prev.rates.length === r.rates.length &&
      prev.rates.every((v, i) => v === r.rates[i]);
    if (!sameP || !sameRates) distinct.push(r);
  }
  const nDistinct = distinct.length;

  // --- Denominator: fit ln P against ln(1 + hours). ---
  const px = distinct.map(xOf);
  const py = distinct.map((r) => Math.log(Math.max(1, r.players_tracked)));
  const pFit = ols(px, py);
  // Launch-week growth decelerates in log-time (fast, then slowing): the
  // quadratic fit's local slope at the last read is the honest
  // extrapolation, never the early steep slope.
  const pQuad = quadFit(px, py);
  const pLocalSlope = pQuad ? pQuad.b + 2 * pQuad.c * xLast : pFit.slope;
  // Blend the quadratic's local slope with the straight-line slope: the
  // quadratic alone can overreact to early curvature, the line alone
  // ignores deceleration. The local slope already carries the slowing;
  // no extra curvature term is added on top of it.
  const bPFit = clamp(0.5 * pLocalSlope + 0.5 * pFit.slope, 0, 1.3);
  // With almost no distinct reads the player-growth slope is unidentified:
  // shrink it toward a launch-week prior (0.7) instead of leaving P frozen
  // while E grows at its prior slope — that mismatch alone would explode
  // every rate on a long horizon from a single read.
  const wP = nDistinct >= 2 ? Math.min(1, (nDistinct - 1) / 3) : 0;
  const bP = wP * bPFit + (1 - wP) * 0.7;
  const pLast = Math.max(1, last.players_tracked);
  let pTgt = Math.exp(Math.log(pLast) + bP * dx);
  pTgt = clamp(pTgt, pLast, pLast * 40);

  // Biggest refresh jump seen in ln P between consecutive distinct reads.
  let maxStepLogP = 0;
  for (let i = 1; i < distinct.length; i++) {
    const step = Math.abs(
      Math.log(Math.max(1, distinct[i].players_tracked)) -
        Math.log(Math.max(1, distinct[i - 1].players_tracked)),
    );
    if (step > maxStepLogP) maxStepLogP = step;
  }

  // --- Spread building blocks (logit sigma, combined in quadrature). ---
  const sWalk = 0.2 * Math.sqrt(horizonDays);
  // A history that never moved at all means a refresh is overdue: the next
  // one can shift every rate at once, so carry a wide refresh allowance.
  const sRefresh =
    nDistinct <= 1
      ? 0.75
      : clamp(0.25 + 0.35 * maxStepLogP, 0.25, 0.6);
  const fitUncP = clamp(pFit.residSd, 0.08, 0.6);

  const rows: Array<{ id: number; q: number[] }> = [];
  c.trophies.forEach((t, i) => {
    // Last non-null rate for this trophy (walks back over late-appearing trophies).
    let lastRate: number | null = null;
    let lastIdx = -1;
    for (let k = hist.length - 1; k >= 0; k--) {
      const v = hist[k].rates[i];
      if (v !== null && v !== undefined) {
        lastRate = v;
        lastIdx = k;
        break;
      }
    }

    // Trophy at (essentially) 100%: everyone tracked holds it, and the one
    // refresh on record left it at 100%. Keep it near the top, tight spread.
    if (lastRate !== null && lastRate >= 99.99) {
      const m = 99.6;
      const s = 0.3;
      rows.push({ id: t.id, q: Z.map((z) => round4(clamp(invLogit(logit(m) + z * s), 0.01, 99.99))) });
      return;
    }

    // --- Numerator: fit ln(E + 0.5) against ln(1 + hours). ---
    // Only distinct reads carry signal; duplicates would overweight flats.
    const ex: number[] = [];
    const ey: number[] = [];
    const eTimes: number[] = [];
    let eLast = 0;
    for (const dr of distinct) {
      const v = dr.rates[i];
      if (v === null || v === undefined) continue;
      const e = (v / 100) * dr.players_tracked;
      ex.push(xOf(dr));
      ey.push(Math.log(e + 0.5));
      eTimes.push(Date.parse(dr.at));
    }
    for (let k = 0; k < hist.length; k++) {
      const v = hist[k].rates[i];
      if (v === null || v === undefined) continue;
      if (k === lastIdx) eLast = (v / 100) * hist[k].players_tracked;
    }
    const eFit = ols(ex, ey);
    // Damped trend: the quadratic's local slope at the last read captures
    // the slowing growth of a trophy nearing saturation, blended with the
    // straight-line slope so early curvature cannot dominate. Straight-line
    // fits look steep for early-story trophies and would otherwise explode.
    const eQuad = quadFit(ex, ey);
    const eLocalSlope = eQuad ? eQuad.b + 2 * eQuad.c * xLast : eFit.slope;
    const eFitSlope = 0.5 * eLocalSlope + 0.5 * eFit.slope;
    const prior = priorSlope(t, lastRate ?? priorRate(t));
    // Weight the fitted slope by how much distinct signal exists; thin
    // histories lean on the prior, rich ones on the fit.
    const w = ex.length >= 2 ? ex.length / (ex.length + 4) : 0;
    let bE = w * clamp(eFitSlope, -0.2, 1.5) + (1 - w) * prior;
    // Recent-window slope: distinct reads from the last 24 h of clock time.
    // Launch-week behaviour drifts (early story rush, then slowing), so the
    // recent slope gets a 35% say when at least 3 recent points exist.
    if (ex.length >= 3) {
      const tLastMs = Date.parse(last.at);
      const rx: number[] = [];
      const ry: number[] = [];
      for (let j = 0; j < ex.length; j++) {
        if (tLastMs - eTimes[j] <= 24 * 3600000) {
          rx.push(ex[j]);
          ry.push(ey[j]);
        }
      }
      if (rx.length >= 3) {
        const recent = ols(rx, ry);
        bE = 0.65 * bE + 0.35 * clamp(recent.slope, -0.2, 1.5);
      }
    }

    let eTgt: number;
    if (lastRate === null || eLast <= 0) {
      // Never earned (or never seen): start from a half-earner pseudo-count
      // and grow it at the prior slope, so the rate stays small but nonzero.
      const e0 = 0.5;
      eTgt = e0 * Math.exp(prior * dx);
    } else {
      eTgt = (eLast + 0.5) * Math.exp(bE * dx) - 0.5;
    }
    eTgt = Math.max(0, eTgt);
    // Earners can never exceed the players tracked.
    eTgt = Math.min(eTgt, 0.995 * pTgt);

    let m = (100 * eTgt) / pTgt;
    // Cross-check in rate space: a recent-window fit of logit(rate) against
    // log-time sees saturation directly (the rate flattening), while the
    // E/P fit can stay steep from the early rush. Blend the two medians in
    // logit space; the rate fit gets half the weight once the trophy is
    // common enough for its recent slope to be meaningful.
    if (lastRate !== null && lastRate < 99 && ex.length >= 3) {
      const tLastMs = Date.parse(last.at);
      const rx: number[] = [];
      const ry: number[] = [];
      for (const dr of distinct) {
        const v = dr.rates[i];
        if (v === null || v === undefined) continue;
        if (tLastMs - Date.parse(dr.at) <= 36 * 3600000) {
          rx.push(xOf(dr));
          ry.push(logit(v));
        }
      }
      if (rx.length >= 3) {
        const rf = ols(rx, ry);
        const slopeR = clamp(rf.slope, -0.3, 1.0);
        const m2 = invLogit(logit(lastRate) + 0.7 * slopeR * dx);
        const wr = lastRate >= 1 ? 0.5 : 0.25;
        m = invLogit((1 - wr) * logit(m) + wr * logit(m2));
      }
    }
    // Flat history + short horizon: the next refresh dilutes the rate before
    // earners catch up (new players have not earned anything yet). Pull the
    // median down by a modest dilution factor in exactly that regime; the
    // trend fit already handles dilution when the history actually moves.
    if (nDistinct <= 1 && horizonDays < 0.5 && lastRate !== null && lastRate < 99) {
      m = m * 0.55;
    }
    m = clampPct(m);

    // --- Spread for this trophy. ---
    const residE = ex.length >= 3 ? clamp(eFit.residSd, 0.08, 0.7) : 0.45;
    // Fit residuals on a staircase history overstate true noise (they are
    // mostly model misfit from saturation), so cap the combined fit term.
    const fitUnc = Math.min(0.4, Math.sqrt(residE * residE + fitUncP * fitUncP));
    const trendUnc = 0.15 * Math.abs(bE - bP) * dx;
    const horizonScale = 1 + 0.25 * (dx / Math.max(1, xLast + 0.5));
    const s = clamp(
      Math.sqrt(sWalk * sWalk + (fitUnc * horizonScale) ** 2 + sRefresh * sRefresh + trendUnc * trendUnc),
      0.3,
      1.05,
    );

    const base = logit(m);
    const q = Z.map((z) => round4(clamp(invLogit(base + z * s), 0.01, 99.99)));
    // Enforce non-decreasing after rounding.
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    rows.push({ id: t.id, q });
  });
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith(".in"))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -".in".length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), "utf8")) as CaseFile;
  if (!c.history || c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c as CaseFile);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
