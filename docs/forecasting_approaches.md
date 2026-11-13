# Forecasting approaches

## What we forecast: the auction's "error"

**The trade.** At 10:00 on T−1 we buy a PTP Obligation for one hour of day T, for example source WEST → sink NORTH.
It settles in two legs:
- we **pay** the day-ahead price gap: `da_gap = DA price at NORTH − DA price at WEST`, fixed in the auction;
- we **receive** the real-time price gap: `rt_gap = RT price at NORTH − RT price at WEST`, from what actually happens on T (the hour's average of the four 15-minute prices).

**Profit per MW for that hour:**

```
error = rt_gap − da_gap
```

**Why we call it the "error".** The day-ahead gap is the auction's forecast of congestion between the two hubs. The real-time gap is the congestion that actually happened. `error` is how wrong the auction was. If the auction were always right, `error` would average zero and nobody could make money.

**Example (one hour, WEST → NORTH):**

| | NORTH | WEST | Gap (NORTH − WEST) |
|---|---|---|---|
| Day-ahead | $40 | $25 | `da_gap` = $15 (we pay) |
| Real-time | $60 | $20 | `rt_gap` = $40 (we receive) |

`error` = $40 − $15 = **+$25/MWh**. With 10 MW that's $250 for the hour, minus about $0.50/MWh in costs.

**Sign = direction.**
- `error > 0`: congestion was worse than the auction expected, so buying source → sink makes money.
- `error < 0`: congestion was milder than the auction expected (or reversed), so the opposite trade (NORTH → WEST) makes money.

Every approach below answers one question: **for each pair and hour of day T, will `error` be clearly positive or clearly negative, and how sure are we?**
We only trade when the forecast is confidently beyond ±$5. Below that, costs and noise eat the edge.

All approaches use the same trading rules, risk rules and backtest, so their results can be compared directly. Details are in the [spec](superpowers/specs/2026-11-13-ptp-strategy-design.md).

## 1. Foundation models forecast the auction's error (main approach)

A time-series foundation model, used zero-shot with no training, forecasts tomorrow's `error` per pair and hour, as quantiles q10…q90.
Inputs: the error history, the calendar, and ERCOT's wind forecast (West, Panhandle) and load forecast (by weather zone), always the version published before 10:00 on T−1.
Trade when the forecast distribution is mostly above +$5 (go long) or mostly below −$5 (go short).

Models (same inputs, same output, only the model changes):
- **Chronos-2** (Amazon): main model.
- **TabPFN-TS** (Prior Labs): treats forecasting as tabular regression with known future inputs, which fits our wind and load forecasts. Prior Labs claims it does best in that setting, but that's their own benchmark and hasn't been checked independently. Check its licence and CPU speed first.

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

## 4. Use the market's own PTP bids

ERCOT publishes how many MW of PTP Obligations cleared at each hub in every auction.
- **Price-impact check:** is our size small compared with the market? (A risk check, not a signal.)
- **Crowding signal:** does heavy buying into a sink on T−1 predict the auction being wrong on T? First confirm the data is public before 10:00 on T−1.

## Ideas not in the spec yet

Only if the approaches above are done and time remains:
- **Gradient boosting / quantile regression** (for example, LightGBM) on the same inputs: the standard "classic ML" benchmark for foundation models.
- **Grid constraints and outages:** ERCOT publishes which transmission lines are binding and which are out of service. These directly cause hub spreads, but the data is large and harder to align in time.
- **Combining models:** average the quantiles of Chronos-2 and TabPFN-TS.

## Order of work

1 with Chronos-2 → 2a, 2b → 1 with TabPFN-TS → 3, 2c → 4 → the ideas not in the spec yet.
