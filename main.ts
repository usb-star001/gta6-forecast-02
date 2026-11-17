// gta6-twitch-ccu-forecast-01 (v2): forecast GTA VI's launch-day Twitch CCU, hour by
// hour, as five quantiles per hour (levels 0.05, 0.25, 0.5, 0.75, 0.95) for hours
// 2..24 of the release day (2026-11-19 UTC), anchored on the case's hour-1 print.
//
// MODEL — "kernel-weighted ridge on log-ratios, with unlock-type and pre-launch
// hype terms". A different model from v1 (weighted median ratio curve), fitted at
// run time from the case's own `comparables` (38 past big launches in a practice
// case, 42 in the final), never from a hardcoded curve. See MODEL.md for the
// measured effects and the variants tested and rejected.
//
// For every hour h = 2..24, the model predicts the log-ratio y_h = ln(v_h / v_1)
// of that hour's viewers to hour 1's print, by a per-hour weighted ridge
// regression over the comparable launches:
//
//   y_h ~ a + b * (ln v1 - ln p1)  [+ c * (ln pre24 - ln pre24_target)]
//                                  [+ d * rolling]
//
//   * SIZE (b). The day's shape depends strongly on how big hour 1 is: the
//     log-ratio at hour 19 correlates -0.73 with ln(v1) across clean launches
//     (-0.93 across all). Small hour-1 launches ramp 3-10x by the evening;
//     big ones (hour-1 >= 100k) peak at ~1.25x. The regression learns that
//     slope per hour instead of assuming one median shape.
//   * KERNEL WEIGHTS. Each comparable votes with a Gaussian kernel on
//     |ln(v1) - ln(p1)| (bandwidth 1.0 in log units): launches near the
//     target's size matter most. Launches whose hour 1 was still pre-unlock
//     (peak/v1 > 20) are downweighted 0.15x (another 0.2x above 100x) —
//     their curves measure unlock timing, not the day's shape.
//   * PRE-LAUNCH HYPE (c). The geometric mean of the category's viewers in
//     the 24 hours before the release day (from the case's `history`, and
//     from each comparable's own pre-launch hourly points). Low pre-launch
//     hype predicts a bigger ramp, beyond what size alone says. It is used
//     only for smaller targets (hour-1 print < 150k, where it was measured
//     to help) and is heavily shrunk (ridge penalty 3.0); for a big launch
//     like GTA VI it is switched off entirely.
//   * ROLLING UNLOCK (d). A binary term: 1 when a launch unlocked at rolling
//     midnight local (parsed from the comparable's `unlock` text, or from the
//     case's `notes` for the target), 0 otherwise. Rolling launches ramp
//     harder into the US evening than simultaneous-global-unlock launches,
//     and this term is the single biggest measured improvement over v1.
//   * RIDGE. Slopes are shrunk toward 0 (penalty 0.25 on the size slope, 3.0
//     on hype, 1.0 on rolling) so the fit never extrapolates wildly for a
//     target bigger than every comparable, as GTA VI's hour 1 may be.
//   * SPREAD. Per hour, sigma is the kernel-weighted residual standard
//     deviation around the fitted median, times 0.6, clamped to [0.18, 0.85]
//     — calibrated so the 90% band covers about 0.9 of scored hours.
//     Quantiles are lognormal: q_tau = p1 * exp(med_h + z_tau * sigma_h).
//   * SMOOTHING. The median log-ratio curve is smoothed with a 3-hour moving
//     average so a single noisy comparable hour cannot kink the forecast.
//
// VALIDATION (local, on data shipped in the kit — see MODEL.md):
//   - Practice (4 cases, true answers from SullyGnome): loss 0.154
//     (p00 0.168, p01 0.119, p02 0.109, p03 0.220), 90% coverage 0.91.
//     v1 scored 0.260 on the same cases; the house baseline 0.387.
//   - Leave-one-out over the 38 comparables: loss 0.399 (v1: 0.801).
//   - Leave-one-out over big launches only (hour-1 >= 100k, GTA VI's
//     regime): loss 0.175 (v1: 0.231).
//
// The program is deterministic (no randomness, no clock, no network): the same
// case file always produces the same bytes. It reads every *.in file in the input
// directory and writes one <id>.out per case, exactly 23 rows, hours 2..24.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** Gaussian kernel bandwidth on ln(v1), in log units. */
const BANDWIDTH = 1.0;
/** Ridge penalties: size slope, hype slope, rolling term. */
const LAM_SIZE = 0.25;
const LAM_HYPE = 3.0;
const LAM_ROLL = 1.0;
/** The hype term is used only when the hour-1 print is below this. */
const HYPE_MAX_P1 = 150000;
/** Spread calibration: residual std is multiplied by this, then clamped. */
const SIGMA_MULT = 0.6;
const SIGMA_MIN = 0.18;
const SIGMA_MAX = 0.85;
/** Fallback spread per square-root hour, the house baseline's (used only when a
 *  comparable hour has no usable launch). */
const FALLBACK_SIGMA = 0.15;
/** Downweight for launches whose hour 1 was still pre-unlock. */
const PREUNLOCK_WEIGHT = 0.15;
const PREUNLOCK_HARD_WEIGHT = 0.2;
const PREUNLOCK_RATIO = 20;
const PREUNLOCK_HARD_RATIO = 100;

interface KnownRow {
  hour: number;
  viewers: number;
  channels?: number | null;
}
interface HourlyPoint {
  from: string;
  viewers: number;
  channels?: number | null;
}
interface HistoryPoint {
  at: string;
  viewers: number;
  channels?: number | null;
}
interface Comparable {
  name: string;
  day_start: string;
  hourly: HourlyPoint[];
  unlock?: string;
}
interface Case {
  levels: number[];
  known: KnownRow[];
  forecast: Array<{ hour: number }>;
  comparables?: Comparable[];
  history?: HistoryPoint[];
  notes?: string;
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** Geometric mean of positive values, or null when there are none. */
function geoMean(values: number[]): number | null {
  const xs = values.filter((v) => v > 0);
  if (xs.length === 0) return null;
  return Math.exp(xs.reduce((s, v) => s + Math.log(v), 0) / xs.length);
}

/** "Rolling midnight local" unlock? Parsed from free text (a comparable's
 *  `unlock` field, or the case's `notes` for the target). */
function isRolling(text: string | undefined): boolean {
  if (!text) return false;
  const t = text.toLowerCase();
  return t.includes('rolling') || t.includes('midnight local');
}

interface CompStats {
  v1: number;
  peak: number;
  ratio: number;
  hours: Map<number, number>;
  pre24: number | null;
  rolling: boolean;
}

/** Per-comparable stats from its hourly curve: hour-1 viewers, peak, the
 *  pre-launch 24h geometric mean, and the unlock type. */
function compStats(c: Comparable): CompStats | null {
  const dayStart = Date.parse(c.day_start);
  if (!Number.isFinite(dayStart)) return null;
  const hours = new Map<number, number>();
  const pre24Values: number[] = [];
  for (const p of c.hourly ?? []) {
    const t = Date.parse(p.from);
    if (!Number.isFinite(t) || typeof p.viewers !== 'number' || p.viewers <= 0) continue;
    const dtHours = (t - dayStart) / 3_600_000;
    if (dtHours >= -24 && dtHours < 0) pre24Values.push(p.viewers);
    const hour = Math.floor(dtHours) + 1;
    if (hour >= 1 && hour <= 24 && !hours.has(hour)) hours.set(hour, p.viewers);
  }
  const v1 = hours.get(1);
  if (v1 === undefined) return null;
  const peak = Math.max(...hours.values());
  return {
    v1,
    peak,
    ratio: peak / v1,
    hours,
    pre24: geoMean(pre24Values),
    rolling: isRolling(c.unlock),
  };
}

/** The target's pre-launch 24h geometric mean, from the case's `history`
 *  (minute reads for the final, SullyGnome hourly points for the practice). */
function targetPre24(c: Case): number | null {
  const dayStart = Date.parse((c as unknown as { day_start?: string }).day_start ?? '');
  if (!Number.isFinite(dayStart)) return null;
  const values: number[] = [];
  for (const p of c.history ?? []) {
    const t = Date.parse(p.at);
    if (!Number.isFinite(t) || typeof p.viewers !== 'number' || p.viewers <= 0) continue;
    const dtHours = (t - dayStart) / 3_600_000;
    if (dtHours >= -24 && dtHours < 0) values.push(p.viewers);
  }
  return geoMean(values);
}

/** Solve a small dense linear system by Gaussian elimination with partial
 *  pivoting. `a` is modified in place; returns null when singular. */
function solveLinear(a: number[][], b: number[]): number[] | null {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    }
    [m[col], m[piv]] = [m[piv], m[col]];
    if (Math.abs(m[col][col]) < 1e-12) return null;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = m[r][col] / m[col][col];
      for (let cc = col; cc <= n; cc++) m[r][cc] -= f * m[col][cc];
    }
  }
  return m.map((row, i) => row[n] / row[i]);
}

/**
 * Weighted ridge fit y ~ a + b*(x - x0) [+ c*hype] [+ d*rolling] for one hour.
 * Returns the fitted median log-ratio at the target (x = x0, hype = 0 when the
 * target's own hype is the reference, rolling = the target's rolling flag)
 * and the residual spread sigma.
 */
function hourFit(
  rows: Array<{ y: number; w: number; f: number[] }>,
  pens: number[],
  targetRoll: number,
  rollIdx: number,
  h: number,
): [number, number] {
  const k = rows[0].f.length;
  const a: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const b: number[] = new Array<number>(k).fill(0);
  for (const r of rows) {
    for (let i = 0; i < k; i++) {
      b[i] += r.w * r.f[i] * r.y;
      for (let j = 0; j < k; j++) a[i][j] += r.w * r.f[i] * r.f[j];
    }
  }
  for (let i = 1; i < k; i++) a[i][i] += pens[i];
  const coef = solveLinear(a, b);
  if (coef === null) return [0, FALLBACK_SIGMA * Math.sqrt(h - 1)];
  // The target sits at x = x0 (feature 1 = 0), hype = 0 (its own reference),
  // rolling = targetRoll.
  let med = coef[0];
  if (rollIdx >= 0) med += coef[rollIdx] * targetRoll;
  // Spread: the kernel-weighted spread of the comparables' outcomes around
  // the TARGET's predicted median (not around each row's own fit) — this is
  // what the 0.6 calibration was measured on.
  let wSum = 0;
  let varSum = 0;
  for (const r of rows) {
    const resid = r.y - med;
    varSum += r.w * resid * resid;
    wSum += r.w;
  }
  const sigma = Math.max(SIGMA_MIN, Math.min(SIGMA_MAX, Math.sqrt(varSum / Math.max(wSum, 1e-9)) * SIGMA_MULT));
  return [med, sigma];
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${file} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);
  const x0 = Math.log(p1);
  const pre24Target = targetPre24(c);
  const useHype = pre24Target !== null && p1 < HYPE_MAX_P1;
  const targetRoll = isRolling(c.notes) ? 1 : 0;
  const comps: CompStats[] = [];
  for (const comp of c.comparables ?? []) {
    const s = compStats(comp);
    if (s !== null) comps.push(s);
  }

  // Fit the shape for every hour first, then smooth the median curve.
  const shape = new Map<number, [number, number]>();
  for (const { hour: h } of c.forecast) {
    const rows: Array<{ y: number; w: number; f: number[] }> = [];
    for (const comp of comps) {
      const vh = comp.hours.get(h);
      if (vh === undefined) continue;
      const y = Math.log(vh / comp.v1);
      const x = Math.log(comp.v1);
      let w = Math.exp(-0.5 * ((x - x0) / BANDWIDTH) ** 2);
      if (comp.ratio > PREUNLOCK_RATIO) w *= PREUNLOCK_WEIGHT;
      if (comp.ratio > PREUNLOCK_HARD_RATIO) w *= PREUNLOCK_HARD_WEIGHT;
      const f = [1, x - x0];
      if (useHype) {
        // A comparable without pre-launch data votes as if it matched the
        // target's hype (feature 0); all comparables in the kit have it.
        const d = comp.pre24 !== null && pre24Target !== null
          ? Math.log(comp.pre24) - Math.log(pre24Target)
          : 0;
        f.push(d);
      }
      f.push(comp.rolling ? 1 : 0);
      rows.push({ y, w, f });
    }
    if (rows.length === 0) {
      shape.set(h, [0, FALLBACK_SIGMA * Math.sqrt(h - 1)]);
      continue;
    }
    // Penalties line up with the features: [intercept 0, size, hype?, roll].
    const pens = [0, LAM_SIZE];
    if (useHype) pens.push(LAM_HYPE);
    pens.push(LAM_ROLL);
    const rollIdx = rows[0].f.length - 1;
    shape.set(h, hourFit(rows, pens, targetRoll, rollIdx, h));
  }
  const hours = c.forecast.map(({ hour }) => hour).sort((a, b) => a - b);
  const medians = hours.map((h) => shape.get(h)![0]);
  const smoothed = new Map<number, number>();
  for (let i = 0; i < hours.length; i++) {
    const lo = Math.max(0, i - 1);
    const hi = Math.min(hours.length, i + 2);
    let sum = 0;
    for (let j = lo; j < hi; j++) sum += medians[j];
    smoothed.set(hours[i], sum / (hi - lo));
  }

  const outHours = c.forecast.map(({ hour }) => {
    const [, sigma] = shape.get(hour)!;
    const med = smoothed.get(hour)!;
    const q = Z.map((z) => {
      const v = round2(p1 * Math.exp(med + z * sigma));
      return Math.min(1e9, Math.max(0.01, v));
    });
    // Keep the row non-decreasing even after rounding.
    for (let i = 1; i < q.length; i++) q[i] = Math.max(q[i], q[i - 1]);
    return { hour, q };
  });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours: outHours })}\n`);
}
