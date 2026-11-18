import numpy as np
import pandas as pd

from lgbm import features, train_cutoff


def _spreads(days=20):
    dates = pd.date_range("2024-06-01", periods=days)
    rows = [{"pair": "A->B", "delivery_date": d, "hour": h,
             "hour_ending": (d + pd.Timedelta(hours=h + 1)).tz_localize("America/Chicago").tz_convert("UTC"),
             "error": float(i), "da_gap": 100.0 + i, "rt_gap": 0.0}
            for i, d in enumerate(dates) for h in range(24)]
    return pd.DataFrame(rows)


def test_features_only_use_information_known_at_bid_time():
    f = features(_spreads()).set_index("delivery_date")
    day = f.loc["2024-06-15"]                      # error on this day = 14
    assert (day["err_lag2"] == 12).all()           # T-2: last settled day
    assert (day["err_lag7"] == 7).all()
    assert np.allclose(day["err_mean7"], np.mean(range(6, 13)))   # T-8 .. T-2
    assert (day["err_day2"] == 12).all()
    assert (day["da_lag1"] == 113).all()           # T-1's auction, public before our bid
    assert (day[["err_lag2", "err_lag3", "err_lag7", "err_mean7", "err_day2", "rt_lag2"]] < 14).all().all()


def test_training_never_sees_results_from_after_bid_time():
    assert train_cutoff("2024-07-01") == pd.Timestamp("2024-06-29")   # bid for Jul 1 on Jun 30 knows Jun 29


def test_extra_features_use_t2_actuals_t1_auction_and_t_outage_forecast():
    from lgbm import MISSES, OUTAGE, SHADOW
    s = _spreads()
    he = pd.Index(s["hour_ending"].unique())
    day = pd.Series(range(len(he)), index=he) // 24          # value = day number (0 = Jun 1)
    from compare_models import FUNDAMENTALS
    cov = pd.DataFrame({m: 0.0 for m in FUNDAMENTALS}, index=he)   # forecast 0, so miss = actual
    extra = pd.DataFrame({**{f"{m}_act": day for m in MISSES}, **{c: day for c in SHADOW + OUTAGE}}, index=he)
    f = features(s, cov, extra).set_index("delivery_date").loc["2024-06-15"]   # day 14
    assert (f["wind_west_miss_lag2"] == 12).all() and (f["load_sys_miss_day2"] == 12).all()
    assert (f["sp_westex_lag1"] == 13).all() and (f["sp_pnhndl_day1"] == 13).all()
    assert (f["out_west"] == 14).all()                       # forecast for T itself, published before 10:00 T-1
