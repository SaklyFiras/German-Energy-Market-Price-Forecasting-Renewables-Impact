# ingestion/fetch_opsd.py
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
URL = "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"

# Select the first column that exists from a preferred list


def pick(cols, preferred_list):
    for name in preferred_list:
        if name in cols:
            return name
    return None


def main():
    print(
        f"Downloading OPSD time series from:\n  {URL}\n(This may take a moment...)")
    df = pd.read_csv(URL, parse_dates=["utc_timestamp"])
    df = df.rename(columns={"utc_timestamp": "ts_utc"})
    cols = set(df.columns)

    # ---- Preferred names in your file (from your output) ----
    price_col = pick(cols, [
        "DE_LU_price_day_ahead",                  # preferred
        "DE_day_ahead_price_EUR_per_MWh",         # older naming (fallback)
    ])
    load_col = pick(cols, [
        "DE_LU_load_actual_entsoe_transparency",  # preferred
        "DE_load_actual_entsoe_transparency",     # fallback
    ])
    wind_col = pick(cols, [
        "DE_LU_wind_generation_actual",           # preferred
        "DE_wind_generation_actual",              # fallback
    ])
    solar_col = pick(cols, [
        "DE_LU_solar_generation_actual",          # preferred
        "DE_solar_generation_actual",             # fallback
    ])

    print("Detected columns:")
    print("  price:", price_col)
    print("  load :", load_col)
    print("  wind :", wind_col)
    print("  solar:", solar_col)

    missing = [name for name, col in [
        ("price", price_col), ("load", load_col), ("wind",
                                                   wind_col), ("solar", solar_col)
    ] if col is None]
    if missing:
        raise SystemExit("Missing required columns: " + ", ".join(missing))

    out = df[["ts_utc", price_col, load_col, wind_col, solar_col]].copy()
    out.columns = ["ts_utc", "price_eur_mwh", "load_mw", "wind_mw", "solar_mw"]

    # Coerce numeric in case of strings
    for c in ["price_eur_mwh", "load_mw", "wind_mw", "solar_mw"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values("ts_utc")
    path = RAW / "opsd_bootstrap.parquet"
    out.to_parquet(path, index=False)
    print("Saved:", path, "rows:", len(out))


if __name__ == "__main__":
    main()
