# features/build_features.py
from pathlib import Path
import pandas as pd
import numpy as np
import holidays

RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)
FEA = Path("data/features"); FEA.mkdir(parents=True, exist_ok=True)

def read_parquet(path: Path, cols, ts="ts_utc"):
    """Read parquet if exists; normalize timestamp column and return [ts]+cols."""
    if not path.exists():
        return pd.DataFrame(columns=[ts] + cols)
    df = pd.read_parquet(path)
    if ts not in df.columns:
        for c in ["timestamp", "time", "datetime", "ts"]:
            if c in df.columns:
                df = df.rename(columns={c: ts})
                break
    df[ts] = pd.to_datetime(df[ts], utc=True)
    return df[[ts] + [c for c in cols if c in df.columns]]

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure tz-aware index (UTC)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    idx_local = df.index.tz_convert("Europe/Berlin")
    df["hour"] = idx_local.hour
    df["dow"] = idx_local.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6])

    years = pd.Index(idx_local.year).unique().tolist()
    de_holidays = holidays.country_holidays("DE", years=years)
    holiday_dates = pd.to_datetime(list(de_holidays.keys()))
    local_norm = idx_local.tz_localize(None).normalize()
    df["is_holiday_de"] = local_norm.isin(holiday_dates)
    return df

def add_lags_rollings(df: pd.DataFrame, col: str) -> pd.DataFrame:
    # Lags
    for h in [1, 24, 48, 168]:
        df[f"{col}_lag{h}"] = df[col].shift(h)
    # Rollings
    df[f"{col}_roll24_mean"] = df[col].rolling(24).mean()
    df[f"{col}_roll168_mean"] = df[col].rolling(168).mean()
    return df

def main():
    # ---------- TARGET: price (ENTSOE primary, fallback to OPSD) ----------
    entsoe = read_parquet(RAW / "entsoe_day_ahead.parquet", ["price_eur_mwh"])
    opsd   = read_parquet(RAW / "opsd_bootstrap.parquet",
                          ["price_eur_mwh", "load_mw", "wind_mw", "solar_mw"])

    # Merge price (ENTSO-E takes precedence if overlapping)
    price = opsd[["ts_utc", "price_eur_mwh"]].copy()
    if not entsoe.empty:
        price = pd.concat([price, entsoe]).sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")

    # ---------- DRIVERS: prefer SMARD over OPSD ----------
    smard_load = read_parquet(RAW / "smard_load.parquet", ["load_mw"])
    smard_wind = read_parquet(RAW / "smard_gen_wind.parquet", ["wind_mw"])
    smard_solar = read_parquet(RAW / "smard_gen_solar.parquet", ["solar_mw"])

    # Start with OPSD, then overwrite with SMARD where available (by timestamp)
    load  = opsd[["ts_utc", "load_mw"]].copy()
    wind  = opsd[["ts_utc", "wind_mw"]].copy()
    solar = opsd[["ts_utc", "solar_mw"]].copy()

    if not smard_load.empty:
        load = pd.concat([load, smard_load]).sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")
    if not smard_wind.empty:
        wind = pd.concat([wind, smard_wind]).sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")
    if not smard_solar.empty:
        solar = pd.concat([solar, smard_solar]).sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")

    # ---------- ALIGN + FEATURE ENGINEERING ----------
    df = price.merge(load, on="ts_utc", how="outer") \
              .merge(wind, on="ts_utc", how="outer") \
              .merge(solar, on="ts_utc", how="outer")

    # Resample to hourly grid
    df = df.sort_values("ts_utc").set_index("ts_utc").resample("1h").mean()

    # Renewables share: (wind+solar)/load (proxy). Guard for <=0 or NaNs.
    total_ren = df.get("wind_mw", 0).fillna(0) + df.get("solar_mw", 0).fillna(0)
    denom = df.get("load_mw", np.nan)
    df["renewables_share"] = np.where((denom > 0) & np.isfinite(denom), total_ren / denom, np.nan)

    # Calendar + lags/rollings
    df = add_calendar(df)
    for c in ["price_eur_mwh", "load_mw"]:
        if c in df.columns:
            df = add_lags_rollings(df, c)

    # Final clean: drop rows needed for lags/rollings; ensure numeric dtypes
    df = df.dropna().copy()

    out = FEA / "hourly.parquet"
    df.to_parquet(out)
    print("Saved:", out, "rows:", len(df))
    # Optional: quick peek
    print(df.tail(3))

if __name__ == "__main__":
    main()
