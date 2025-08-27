# models/predict_fan.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import holidays

ART = Path("models/artifacts")
ART.mkdir(parents=True, exist_ok=True)
FEA = Path("data/features/hourly.parquet")
TARGET = "price_eur_mwh"

OUT_CSV = ART / "predictions_fan.csv"
OUT_PNG = ART / "predictions_fan.png"


def load_recent(days=180):
    df = pd.read_parquet(FEA)
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        df = df.set_index("ts_utc").sort_index()
    for c in df.select_dtypes(include=["bool"]).columns:
        df[c] = df[c].astype(int)
    df = df.select_dtypes(include=["number"])
    return df.loc[df.index >= (df.index.max() - pd.Timedelta(days=days))]


def calendar_frame(start_utc, periods=24):
    idx = pd.date_range(start_utc, periods=periods, freq="1h", tz="UTC")
    idx_local = idx.tz_convert("Europe/Berlin")
    hols = holidays.country_holidays("DE", years=idx_local.year.unique())
    df = pd.DataFrame(index=idx)
    df["hour"] = idx_local.hour
    df["dow"] = idx_local.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_holiday_de"] = idx_local.tz_localize(None).normalize().isin(
        pd.to_datetime(list(hols.keys()))
    ).astype(int)
    return df


def main():
    hist = load_recent()
    features = [c for c in hist.columns if c != TARGET]

    last_ts = hist.index.max()
    future_idx = pd.date_range(
        last_ts + pd.Timedelta(hours=1), periods=24, freq="1h", tz="UTC")
    cal = calendar_frame(future_idx[0], periods=24)

    fut = pd.DataFrame(index=future_idx)
    for col in ["hour", "dow", "is_weekend", "is_holiday_de"]:
        fut[col] = cal[col]
    for col in ["load_mw", "wind_mw", "solar_mw", "renewables_share"]:
        fut[col] = hist[col].tail(24).mean()

    work = pd.concat([hist, fut], axis=0).sort_index()

    preds = {}
    for q in range(5, 100, 5):  # q05 … q95
        fname = ART / f"xgb_q{q}.json"
        if not fname.exists():
            raise SystemExit(
                f"Missing {fname}, run train_quantiles_full.py first.")
        model = xgb.XGBRegressor()
        model.load_model(fname.as_posix())
        X_fut = work.loc[future_idx, features].astype(float)
        preds[q] = pd.Series(model.predict(
            X_fut), index=future_idx, name=f"q{q}")

    pred_df = pd.concat(preds.values(), axis=1)
    pred_df.to_csv(OUT_CSV, index_label="ts_utc")
    print("Saved CSV:", OUT_CSV)

    plt.figure(figsize=(12, 6))
    hist_tail = hist[[TARGET]].loc[hist.index >=
                                   (last_ts - pd.Timedelta(hours=48))]
    plt.plot(hist_tail.index, hist_tail[TARGET], label="Actual (last 48h)")
    # Fan plot
    for q in range(5, 50, 5):
        lower = pred_df[f"q{q}"]
        upper = pred_df[f"q{100-q}"]
        plt.fill_between(pred_df.index, lower, upper, alpha=0.1,
                         label=f"{q}-{100-q}%" if q in [5, 10, 25] else None)
    plt.plot(pred_df.index, pred_df["q50"],
             "--o", color="black", label="Median (q50)")
    plt.title("Day-ahead probabilistic forecast fan (next 24h)")
    plt.ylabel("EUR/MWh")
    plt.xlabel("UTC Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    print("Saved plot:", OUT_PNG)


if __name__ == "__main__":
    main()
