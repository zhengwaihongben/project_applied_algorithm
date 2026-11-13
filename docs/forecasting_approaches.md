# Forecasting approaches

The trade: a PTP Obligation from source hub to sink hub, bought in the day-ahead auction at 10:00 on T−1.
Profit per MW per hour = `error = rt_gap − da_gap`, the real-time price gap minus the gap the auction priced in.
Every approach below answers one question: **for each pair and hour of day T, which way will the auction be wrong, and how confident are we?**

All approaches use the same trading rules, risk rules and backtest, so their results can be compared directly. Details are in the [spec](superpowers/specs/2026-11-13-ptp-strategy-design.md).

## 1. Foundation model forecasts the auction's error (main approach)

Chronos-2 (Amazon's time-series foundation model, zero-shot with no training) forecasts tomorrow's `error` per pair and hour, as quantiles q10…q90.
Inputs: the error history, the calendar, and ERCOT's wind forecast (West, Panhandle) and load forecast (by weather zone), always the version published before 10:00 on T−1.
Trade when the forecast distribution is mostly above +$5 (go long) or mostly below −$5 (go short).

Variants we compare:
- **with wind and load forecasts** vs **calendar only**: do the fundamentals add anything?
- **hour × month baseline**: average error by hour and month from 2024. Is the model better than a lookup table?

## 2. Simple rules: do we need a model at all?

| | Approach | Rule | Idea it tests |
|---|---|---|---|
| 2a | **Persistence** | Trade in the direction of the auction's error for the same hour on T−2, if it's above $5 | The auction repeats its recent mistake |
| 2b | **Calendar premium** | Always trade the typical direction for that hour and month | There's a steady premium to collect without forecasting |
| 2c | **Fundamentals rule** | West→North when the West wind forecast is high **and** the Dallas-area load forecast is high; reverse when wind is low | The deck's story as one transparent rule |

## 3. Forecast congestion itself, not the auction's error

Chronos-2 forecasts tomorrow's real-time gap `rt_gap` directly; we trade when it differs from the day-ahead gap by more than $5.
This is the more intuitive framing ("predict congestion"). We expect it to do worse, because the auction already prices in most of the predictable congestion. Worth running as a direct comparison.

## 4. A second foundation model: TabPFN-TS

Prior Labs' TabPFN-TS treats forecasting as tabular regression with known future inputs, which matches our wind and load forecasts.
Prior Labs claims it does best when future covariates are known (their own benchmark, not independently verified). Same inputs and outputs as Chronos-2, so it swaps in directly.
Check the licence and CPU speed first.

## 5. Use the market's own PTP bids

ERCOT publishes how many MW of PTP Obligations cleared at each hub in every auction.
- **Price-impact check:** is our size small compared with the market? (A risk check, not a signal.)
- **Crowding signal:** does heavy buying into a sink on T−1 predict the auction being wrong on T? First confirm the data is public before 10:00 on T−1.

## Ideas not in the spec yet

Only if the approaches above are done and time remains:
- **Gradient boosting / quantile regression** (for example, LightGBM) on the same inputs: the standard "classic ML" benchmark for foundation models.
- **Grid constraints and outages:** ERCOT publishes which transmission lines are binding and which are out of service. These directly cause hub spreads, but the data is large and harder to align in time.
- **Combining models:** average the quantiles of Chronos-2 and TabPFN-TS.

## Order of work

1 (main comparison) → 2a, 2b → 3, 2c → 4 → 5 → the ideas not in the spec yet.
