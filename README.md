# GTA VI launch-day Twitch forecast — v2

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/usb-star001/gta6-forecast-02/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/usb-star001/gta6-forecast-02/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour
of its release day, 2026-11-19 (UTC), as five quantiles per hour (hours 2–24),
from everything recorded up to the end of hour 1. Built for the
[make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
($250 total, scored by mean pinball loss on the log scale).

This is version 2. Version 1 (a weighted-median ratio curve,
`usb-star001/gta6-forecast-01`) scored 0.260 on the practice board; this version
scores **0.154** on the same practice cases.

## The model

`main.ts` (Node 22, standard library only) fits a **kernel-weighted ridge
regression of log-ratios** at run time, per hour, from the case file's own
comparable launches:

1. Anchor on the case's hour-1 print `p1` (UK, central Europe and
   New Zealand/Australia are already unlocked in hour 1).
2. For each comparable launch and each hour, take `ln(viewers_h / viewers_1)`
   and regress it on `ln(viewers_1)` — the day's shape depends on size: small
   hour-1 launches ramp 3–10× by the evening, big ones peak at ~1.25×.
3. Add two terms the median curve missed: a **rolling-midnight-unlock**
   indicator (rolling launches ramp harder into the US evening than
   simultaneous-global launches) and, for smaller targets, a **pre-launch
   hype** term (the category's geometric-mean viewers in the 24 hours before
   the release day).
4. Weight each launch by a Gaussian kernel on how close its hour-1 size is to
   `p1`, downweight launches whose hour 1 was still pre-unlock, and shrink
   every slope with a ridge penalty so a target bigger than any comparable
   never extrapolates wildly.
5. Spread: the kernel-weighted spread of comparable outcomes around the
   predicted median, calibrated so the 90% band covers ~0.9 of scored hours.

Full details, measured effects and rejected variants: [`MODEL.md`](MODEL.md).
Sources: [`SOURCES.md`](SOURCES.md).

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` case in the input directory and writes one `<id>.out` per
case: 23 rows (hours 2–24) of five quantiles each. Deterministic — two runs
are byte-identical. No network, no clock, no randomness.

## Layout

- `main.ts` — the forecaster (the entrypoint; no other language file at root)
- `MODEL.md` — method, measured effects, validation, rejected variants
- `SOURCES.md` — grounding sources for the numbers the model is built on
- `LICENSE` — MIT
