# Model: kernel-weighted ridge on log-ratios (v2)

## What is forecast

For each hour h = 2..24 of GTA VI's release day (2026-11-19 UTC), five
quantiles of Twitch concurrent viewers in the "Grand Theft Auto VI" category,
at levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95, anchored on the case's hour-1 print
p1 (00:00–01:00 UTC). Scored by mean pinball loss on the log scale.

## Why v1 was not enough

Version 1 predicted each hour's log-ratio `ln(v_h / v_1)` as a weighted median
across comparable launches (weights by closeness of hour-1 size). It scored
0.260 on the practice board. Diagnosing it per case and per hour showed two
systematic misses:

- **The evening peak was under-predicted for smaller launches.** Wolverine
  (p1 = 23k) actually peaks 9.1× its hour 1; v1's median said ~2.1× at hour 19.
  Red Dead 2 (p1 = 117k) peaks 3.2×; v1 said ~1.5×. A single median shape
  cannot be right for both a 23k and a 270k hour-1 launch.
- **The US-unlock step at hours 5–6 was missed.** All four practice targets
  step UP at hour 5 (US Eastern midnight unlock, 04:00Z) to 1.3–1.5× hour 1,
  while the comparables' median DIPS to ~0.7× there. The practice targets are
  all rolling-midnight console launches; the comparable pool mixes in
  simultaneous-global-unlock launches whose hour 1 already holds the whole
  world, so their curves dip instead of stepping.

Both are regression problems, not median problems — hence v2.

## The v2 model

Per hour h, a weighted ridge regression over the comparable launches:

```
y_h = ln(v_h / v_1)  ~  a + b·(ln v1 − ln p1) [+ c·(ln pre24 − ln pre24_target)]
                                        [+ d·rolling]
```

- **Size slope (b).** The log-ratio at hour 19 correlates −0.73 with ln(v1)
  across clean launches (−0.93 across all 38): the bigger hour 1, the flatter
  the day. The regression learns this slope per hour. At hour 2 the slope is
  slightly positive (+0.28); by hour 12 it is −0.50; by hour 19, −0.73.
- **Kernel weights.** Gaussian kernel on |ln(v1) − ln(p1)|, bandwidth 1.0 in
  log units: launches near the target's size dominate. Launches whose hour 1
  was still pre-unlock (peak/v1 > 20) are downweighted 0.15× (another 0.2×
  above 100×), as in v1.
- **Rolling-unlock term (d).** Binary: 1 when a launch unlocked at rolling
  midnight local — parsed from the comparable's `unlock` text ("rolling" or
  "midnight local"), and from the case's `notes` for the target. Rolling
  launches ramp harder into the US evening. This single term is the biggest
  measured improvement: practice 0.187 → 0.160, leave-one-out over big
  launches 0.215 → 0.175. GTA VI itself is a rolling-midnight launch, so the
  term fires for the final.
- **Pre-launch hype term (c).** `pre24` is the geometric mean of the
  category's viewers in the 24 hours before the release day (from the case's
  `history`; from each comparable's own pre-launch hourly points). Low
  pre-launch hype predicts a bigger ramp beyond what size says
  (correlation of ln pre24 with the hour-19 log-ratio is −0.33). It is used
  only for smaller targets (hour-1 print < 150k) and heavily shrunk
  (penalty 3.0); for a big launch like GTA VI it is switched off, so it
  cannot distort the final.
- **Ridge shrinkage.** Penalties: 0.25 on the size slope, 3.0 on hype, 1.0 on
  rolling. The fit is local (kernel) and shrunk, so a target bigger than any
  comparable — GTA VI's hour 1 may exceed the largest comparable hour 1
  (875k, ELDEN RING) — gets a sane, near-flat forecast rather than a wild
  extrapolation. Checked on a synthetic GTA-like target (p1 = 750k):
  predicted ratios 0.70× / 0.58× / 1.01× / 0.97× at hours 5 / 12 / 19 / 21.
- **Spread.** Per hour, sigma is the kernel-weighted spread of the
  comparables' outcomes around the predicted median, × 0.6, clamped to
  [0.18, 0.85]. Quantiles are lognormal: `q_τ = p1 · exp(med_h + z_τ · σ_h)`.
  The 0.6 multiplier is calibrated: 90% coverage is 0.91 on the practice.
- **Smoothing.** The median log-ratio curve is smoothed with a 3-hour moving
  average. Sigma is not smoothed.

## Validation (local)

Scoring is mean pinball loss over the 23 hours × 5 quantiles, exactly as the
kit scores it. Practice answers are the true SullyGnome hourly averages for
the four practice launches (verified to match the case files' hour-1 prints
exactly). The harness reproduces the server's numbers for v1 (0.260) and the
house baseline (0.387), so the v2 numbers below are on the same scale.

| model | practice mean | p00 | p01 | p02 | p03 | LOO | LOO-big | LOO-clean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| house baseline | 0.387 | 0.744 | 0.287 | 0.215 | 0.303 | 0.930 | 0.261 | — |
| v1 (weighted median) | 0.260 | 0.464 | 0.223 | 0.148 | 0.207 | 0.801 | 0.231 | 0.270 |
| **v2 (this)** | **0.154** | **0.168** | **0.119** | **0.109** | **0.220** | **0.399** | **0.175** | **0.255** |

- Practice: 4 cases, true answers. Coverage 0.91.
- LOO: leave-one-out over the 38 comparables (each launch predicted from the
  other 37). LOO-big: targets with hour-1 ≥ 100k (GTA VI's regime, n=16).
  LOO-clean: targets that are clean-tier (peak/v1 ≤ 20) and not
  early-access-contaminated or weak (n=19) — the closest analogue to GTA VI.

The remaining worst case is p03 (Spider-Man 2, 0.220): it has an idiosyncratic
midday dip (hour 11 prints 0.37× hour 1) that no comparable-based model
predicts — that loss is mostly irreducible noise, not a shape the model can
learn.

## What did not work (tested and rejected)

- **Unlock-time alignment** (shifting each comparable's curve by its parsed
  first-unlock offset): practice 0.191 → 0.208, LOO 0.420 → 0.450. The
  unlock text is free-form and mostly unparseable, so the shifts were noise.
- **Per-quantile pinball regression** (fitting each quantile directly):
  practice 0.286 — the IRLS fits are unstable on 37 points.
- **LAD (least-absolute-deviation) median**: practice 0.211, no gain.
- **Empirical residual quantiles for the spread** (instead of lognormal):
  practice 0.233 — worse.
- **Two-centre blend** (clean-tier ridge blended with full-pool ridge):
  practice 0.209, LOO 0.505 — worse than the full pool alone.
- **Tier exclusion** (dropping early-access-contaminated / weak launches):
  practice 0.169, but LOO 0.449 — the pool is too small to throw launches
  away; the kernel + downweighting already handles them.
- **Quadratic term in ln(v1)**: practice 0.194 — no gain over local-linear.
- **Pre-launch trend** (pre24 / pre-7d ratio): practice 0.195 — no gain.
- **Viewers-per-channel at hour 1**: practice 0.182 but LOO 0.435 — a
  practice-only gain, dropped as overfitting.
- **Median-only blend** (α = 0.7 ridge + 0.3 v1 median): practice 0.228,
  LOO 0.694 — the pure ridge is better.
- **Wider kernel / stronger shrinkage**: bandwidth 1.25–1.5 and λ = 1–3 are
  all slightly worse than bandwidth 1.0, λ = 0.25.

## What the program does not use

No hardcoded GTA VI numbers, no memorized practice answers (the 0.000001-loss
entries on the board memorized the public practice curves; the final is graded
on the real Nov 19 recording nobody has, so memorization cannot win there),
no network, no clock, no randomness. The program reads only the case file it
is given and is deterministic — two runs are byte-identical.
