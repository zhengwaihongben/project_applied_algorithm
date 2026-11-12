"""Download ERCOT DAM and RTM settlement point prices for all hubs, one CSV per month.

Credentials: ERCOT_USERNAME, ERCOT_PASSWORD, ERCOT_SUBSCRIPTION_KEY. Re-running skips finished months.
"""
import os

import ercot as er
import pandas as pd

from config import DOWNLOAD_START, END, HUBS, RAW_DAM, RAW_RTM


def month_ranges(start, end):
    """[(first_day, last_day), ...] per calendar month, clipped to [start, end], as YYYY-MM-DD."""
    out = []
    for m in pd.period_range(start, end, freq="M"):
        a = max(m.start_time, pd.Timestamp(start))
        b = min(m.end_time.normalize(), pd.Timestamp(end))
        out.append((a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")))
    return out


def download_all():
    # save=False: don't write credentials to ~/.ercot/credentials.json in plain text
    er.configure(os.environ["ERCOT_USERNAME"], os.environ["ERCOT_PASSWORD"],
                 os.environ["ERCOT_SUBSCRIPTION_KEY"], save=False)
    RAW_DAM.mkdir(parents=True, exist_ok=True)
    RAW_RTM.mkdir(parents=True, exist_ok=True)
    for first, last in month_ranges(DOWNLOAD_START, END):
        for market, fetch, folder in (("DAM", er.dam, RAW_DAM), ("RTM", er.rtm, RAW_RTM)):
            path = folder / f"{market}_hubs_{first[:7]}.csv"
            if path.exists():
                continue
            fetch(HUBS, first, last, format="long").to_csv(path, index=False)
            print(f"saved {path.name}")


if __name__ == "__main__":
    download_all()
