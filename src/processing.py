"""Raw monthly ERCOT price CSVs -> one row per (hour_ending, hub) with DAM and hourly RTM price."""
import pandas as pd

from config import PROCESSED, RAW_DAM, RAW_RTM

HOURLY_FILE = PROCESSED / "hourly_prices.csv"


def local_day_and_hour(hour_ending):
    """ERCOT delivery date and hour (0-23) from a UTC hour-ending series; HE24 belongs to the same day."""
    start = hour_ending.dt.tz_convert("America/Chicago") - pd.Timedelta(hours=1)
    return start.dt.tz_localize(None).dt.normalize(), start.dt.hour


def _read(files):
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["interval_ending"] = pd.to_datetime(df["interval_ending"], utc=True)
    return df


def rtm_hourly(rtm):
    """Average the four 15-minute RT prices into the hour they end in."""
    rtm = rtm.assign(hour_ending=rtm["interval_ending"].dt.ceil("h"))
    out = rtm.groupby(["hour_ending", "settlement_point"], as_index=False)["price"].mean()
    return out.rename(columns={"settlement_point": "point", "price": "rtm"})


def build_hourly(dam, rtm):
    dam = dam.rename(columns={"interval_ending": "hour_ending", "settlement_point": "point", "price": "dam"})
    df = dam[["hour_ending", "point", "dam"]].merge(rtm_hourly(rtm), on=["hour_ending", "point"], how="inner")
    df["delivery_date"], df["hour"] = local_day_and_hour(df["hour_ending"])
    return df.sort_values(["hour_ending", "point"], ignore_index=True)


def load_hourly():
    df = pd.read_csv(HOURLY_FILE, parse_dates=["delivery_date"])
    df["hour_ending"] = pd.to_datetime(df["hour_ending"], utc=True)
    return df


def main():
    hourly = build_hourly(_read(sorted(RAW_DAM.glob("DAM_hubs_*.csv"))), _read(sorted(RAW_RTM.glob("RTM_hubs_*.csv"))))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(HOURLY_FILE, index=False)
    print(hourly.groupby("point").size())
    print(hourly[["dam", "rtm"]].describe())


if __name__ == "__main__":
    main()
