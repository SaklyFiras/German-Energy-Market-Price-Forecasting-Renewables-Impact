# ingestion/fetch_smard.py
from __future__ import annotations
from pathlib import Path
from typing import Literal, Dict, Optional
import argparse
import pandas as pd
import requests
import yaml

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
BASE = "https://www.smard.de/app/chart_data"
Resolution = Literal["quarterhour", "hour", "day", "week", "month", "year"]


def to_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    else:
        return t.tz_convert("UTC")


def load_cfg(path: Path = Path("configs/smard.yaml")) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Create it with series IDs.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def url_index(filter_id: int, region: str, resolution: Resolution) -> str:
    return f"{BASE}/{filter_id}/{region}/index_{resolution}.json"


def url_payload(filter_id: int, region: str, resolution: Resolution, ts: int) -> str:
    # filename: {filter}_{region}_{resolution}_{timestamp}.json
    return f"{BASE}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{ts}.json"


def fetch_index(filter_id: int, region: str, resolution: Resolution) -> list[int]:
    u = url_index(filter_id, region, resolution)
    r = requests.get(u, timeout=60)
    r.raise_for_status()
    payload = r.json()
    ts_list = payload.get("timestamps") or payload.get("availableTimestamps")
    if not ts_list:
        raise SystemExit(
            f"Unexpected index format at {u}: keys={list(payload.keys())}")
    return sorted(ts_list)


def fetch_one_file(filter_id: int, region: str, resolution: Resolution, ts: int) -> pd.DataFrame:
    u = url_payload(filter_id, region, resolution, ts)
    r = requests.get(u, timeout=60)
    r.raise_for_status()
    payload = r.json()
    data = payload.get("series") or payload.get("data")
    if not isinstance(data, list):
        raise SystemExit(
            f"Unexpected payload at {u}: keys={list(payload.keys())}")
    df = pd.DataFrame(data, columns=["ts_ms", "value"])
    df["ts_utc"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df[["ts_utc", "value"]]


def fetch_series(filter_id: int, region: str, resolution: Resolution,
                 start=None, end=None) -> pd.DataFrame:
    timestamps = fetch_index(filter_id, region, resolution)
    # index timestamps are epoch ms; filter by range if provided
    if start is not None:
        start_ms = int(to_utc(start).timestamp() * 1000)   # <-- changed
        timestamps = [t for t in timestamps if t >= start_ms]
    if end is not None:
        end_ms = int(to_utc(end).timestamp() * 1000)       # <-- changed
        timestamps = [t for t in timestamps if t <= end_ms]

    parts = []
    for ts in timestamps:
        parts.append(fetch_one_file(filter_id, region, resolution, ts))
    if not parts:
        return pd.DataFrame(columns=["ts_utc", "value"])
    out = pd.concat(parts, ignore_index=True).sort_values(
        "ts_utc").drop_duplicates("ts_utc")
    return out


def aggregate_to_hourly(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    if df.empty:
        return df
    g = (df.set_index("ts_utc")
           .sort_index()
           .resample("1h")
           .mean())
    g = g.rename(columns={"value": colname}).reset_index()
    return g


def main():
    cfg = load_cfg()
    region = cfg.get("region", "DE")
    resolution_cfg = cfg.get("resolution", "hour")
    ids: Dict[str, int] = {k: int(v) for k, v in cfg["series"].items()}

    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", choices=["hour", "quarterhour"], default=resolution_cfg,
                    help="Override resolution from config")
    ap.add_argument("--start", help="UTC start YYYY-MM-DD (optional)")
    ap.add_argument("--end", help="UTC end YYYY-MM-DD (optional)")
    ap.add_argument("--limit-years", type=int, default=None,
                    help="If set, only fetch roughly last N years")
    ap.add_argument("--save-qh", action="store_true",
                    help="When using quarterhour, also save raw quarter-hour Parquets")
    args = ap.parse_args()

    # Resolve time window
    start_ts = pd.Timestamp(args.start).tz_localize(
        "UTC") if args.start else None
    end_ts = pd.Timestamp(args.end).tz_localize("UTC") if args.end else None

    if args.limit_years is not None:
        # If end not provided, use now; if start not provided, compute end - N years
        end_ts = end_ts or pd.Timestamp.utcnow().tz_localize("UTC")
        approx_start = end_ts - pd.DateOffset(years=args.limit_years)
        start_ts = start_ts or approx_start

    res: Resolution = "quarterhour" if args.resolution == "quarterhour" else "hour"
    use_qh = (res == "quarterhour")

    # -------- Load (total consumption)
    load_raw = fetch_series(ids["load_actual"], region, res, start_ts, end_ts)
    if use_qh:
        if args.save_qh:
            load_raw.rename(columns={"value": "load_mw"}).to_parquet(
                RAW / "smard_load_qh.parquet", index=False)
        load_df = aggregate_to_hourly(load_raw, "load_mw")
    else:
        load_df = load_raw.rename(columns={"value": "load_mw"})
    load_df.to_parquet(RAW / "smard_load.parquet", index=False)
    print("Saved:", RAW / "smard_load.parquet", "rows:", len(load_df))

    # -------- Wind = onshore + offshore
    wind_on_raw = fetch_series(
        ids["wind_onshore"],  region, res, start_ts, end_ts)
    wind_off_raw = fetch_series(
        ids["wind_offshore"], region, res, start_ts, end_ts)
    if use_qh and args.save_qh:
        wind_on_raw.rename(columns={"value": "wind_on_mw"}).to_parquet(
            RAW / "smard_wind_on_qh.parquet", index=False)
        wind_off_raw.rename(columns={"value": "wind_off_mw"}).to_parquet(
            RAW / "smard_wind_off_qh.parquet", index=False)
    wind_on_h = aggregate_to_hourly(wind_on_raw,  "wind_on_mw") if use_qh else wind_on_raw.rename(
        columns={"value": "wind_on_mw"})
    wind_off_h = aggregate_to_hourly(wind_off_raw, "wind_off_mw") if use_qh else wind_off_raw.rename(
        columns={"value": "wind_off_mw"})
    wind = pd.merge(wind_on_h, wind_off_h, on="ts_utc",
                    how="outer").sort_values("ts_utc")
    wind["wind_mw"] = wind.get("wind_on_mw", 0).fillna(
        0) + wind.get("wind_off_mw", 0).fillna(0)
    wind = wind[["ts_utc", "wind_mw"]]
    wind.to_parquet(RAW / "smard_gen_wind.parquet", index=False)
    print("Saved:", RAW / "smard_gen_wind.parquet", "rows:", len(wind))

    # -------- Solar PV
    solar_raw = fetch_series(ids["solar_pv"], region, res, start_ts, end_ts)
    if use_qh and args.save_qh:
        solar_raw.rename(columns={"value": "solar_mw"}).to_parquet(
            RAW / "smard_gen_solar_qh.parquet", index=False)
    solar_df = aggregate_to_hourly(solar_raw, "solar_mw") if use_qh else solar_raw.rename(
        columns={"value": "solar_mw"})
    solar_df.to_parquet(RAW / "smard_gen_solar.parquet", index=False)
    print("Saved:", RAW / "smard_gen_solar.parquet", "rows:", len(solar_df))


if __name__ == "__main__":
    main()
