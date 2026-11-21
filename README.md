# GTA VI Trophy Rarity Forecast — Team A

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/surya215/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/surya215/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI (PS5) trophy's earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy.

## Approach: decompose the rate into earners over players

An earn rate is a ratio, `r = 100 * E / P`, where `E` is how many tracked
players have earned the trophy and `P` is how many players Exophase tracks.
Exophase refreshes its counts on its own schedule (hours apart), so the
history is a staircase: long flat stretches, then a jump in `P` that dilutes
every rate at once, while `E` keeps growing as people play. Fitting the rate
directly confuses those two motions, so this entry models them separately:

1. **Reconstruct** `E_i(t) = rate_i(t) * P(t) / 100` and `P(t)` from the
   history, collapsing consecutive identical reads (the flat steps carry no
   signal and would overweight the fits).
2. **Fit in log-log time**, `ln E` and `ln P` against `ln(1 + hours since the
   first read)`, with a damped trend: the fitted slope is blended between a
   straight line and the local slope of a quadratic fit (which captures the
   slowing growth of trophies nearing saturation), plus a recent-24h window
   slope. Slopes are shrunk toward a tier/story prior (early story chapters
   fast, platinum very slow, near-saturated trophies capped) when the signal
   is thin, and the player-growth slope is shrunk toward a launch-week prior
   when there are almost no distinct reads.
3. **Forecast** `E_i` and `P` separately to the target and divide. The
   median is cross-checked against a recent-window fit of `logit(rate)` vs
   log-time (which sees saturation directly) and the two are blended in
   logit space.
4. **Quantiles on the logit scale** with a spread combining a random-walk
   term, fit residuals (capped — staircase residuals are mostly saturation
   misfit, not noise), the size of refresh jumps seen in the history, and
   trend uncertainty growing with the horizon. A completely flat history
   carries a wide refresh allowance, and its median is pulled down by a
   dilution factor, because the next refresh brings in new players who have
   not earned anything yet.

Edge cases: the 100% trophy stays near 100% with a tight spread; trophies
never earned use a half-earner pseudo-count grown at the prior slope;
null rates are skipped (the last non-null rate anchors the forecast);
trophies appearing late are handled the same way; a single-read history
falls back to priors with a wide spread.

## Run

Node 22, standard library only, no network, deterministic (no randomness,
no clock):

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` in the input directory and writes `/output/<id>.out`
(`{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}`).

## Local validation

- Practice cases p00/p01: output valid (55 rows, monotone quantiles in
  0–100), byte-identical across two runs, ~0.2 s per run.
- Synthetic cases with realistic dynamics (power-law player growth,
  saturating per-trophy earn curves, 5–8 h Exophase refresh staircase,
  plus noisy and growth-kink variants): mean pinball loss 0.28–0.30 vs
  0.31–0.35 for the persistence baseline in all 7 scenarios, 90%
  coverage 0.96.
- Edge cases (single-read history, 0% rates, all-null rates, late-appearing
  trophy, mid-history refresh jump): valid and deterministic.

## Known weaknesses

- The platinum (and any trophy whose earners only appear after a long lag)
  is hard for a log-log power law: its earner count can jump far faster
  than any fitted slope once players start finishing the game. The spread
  is the main defence there.
- The synthetic validation world is smoother than reality (no regime
  change in trophy curves, refresh schedule fixed), so the spread is kept
  deliberately wider than the synthetic optimum — coverage 0.96 rather
  than 0.90 — as insurance against a real refresh or growth surprise.
- The practice cases cannot validate the trend fit at all (their histories
  are perfectly flat); they only check format, determinism, and that the
  spread/dilution copes with the unseen first refresh.

---
vfpg0cj5
