"""Hourly hub prices -> per-pair DA gap, RT gap and the auction's error (= PnL per MW, source -> sink)."""
import pandas as pd


def build_spreads(hourly, pairs):
    wide = hourly.pivot_table(index=["hour_ending", "delivery_date", "hour"], columns="point", values=["dam", "rtm"])
    parts = []
    for source, sink in pairs:
        part = pd.DataFrame({
            "da_gap": wide["dam"][sink] - wide["dam"][source],
            "rt_gap": wide["rtm"][sink] - wide["rtm"][source],
        }).reset_index()
        part["pair"] = f"{source}->{sink}"
        parts.append(part)
    out = pd.concat(parts, ignore_index=True).dropna(subset=["da_gap", "rt_gap"])
    out["error"] = out["rt_gap"] - out["da_gap"]
    return out.reset_index(drop=True)
