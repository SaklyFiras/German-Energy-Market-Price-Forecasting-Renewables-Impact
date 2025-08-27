# models/predict_next_24h.py
from pathlib import Path
import numpy as np
import pandas as pd
import holidays
import xgboost as xgb
import matplotlib.pyplot as plt

ART = Path("models/artifacts")
ART.mkdir(parents=True, exist_ok=True)
FEA = Path("data/features/hourly.parquet")

OUT_CSV = ART / "predictions_next24h_quantile.csv"
OUT_PNG = ART / "predictions_next24h_quantile.png"

TARGET = "price_eur_mwh"


def read_recent_features(days=180) -> pd.DataFrame:
    if not FEA.exists():
        raise SystemExit(
            "Missing features parquet. Run `make build-features` first.")
    base = pd.read_parquet(FEA)
    if "ts_utc" in base.columns:
        base["ts_utc"] = pd.to_datetime(base["ts_utc"], utc=True)
        base = base.set_index("ts_utc").sort_index()
    for c in base.select_dtypes(include=["bool"]).columns:
        base[c] = base[c].astype(int)
    base = base.select_dtypes(include=["number"])
    return base.loc[base.index >= (base.index.max() - pd.Timedelta(days=days))]


def calendar_frame(start_utc: pd.Timestamp, periods=24) -> pd.DataFrame:
    idx = pd.date_range(start_utc, periods=periods, freq="1h", tz="UTC")
    idx_local = idx.tz_convert("Europe/Berlin")
    years = pd.Index(idx_local.year).unique().tolist()
    de_hols = holidays.country_holidays("DE", years=years)
    hol_dates = pd.to_datetime(list(de_hols.keys()))
    df = pd.DataFrame(index=idx)
    df["hour"] = idx_local.hour
    df["dow"] = idx_local.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_holiday_de"] = idx_local.tz_localize(
        None).normalize().isin(hol_dates).astype(int)
    return df


def prepare_matrix(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    X = df.reindex(columns=feature_names).copy()
    for c in X.select_dtypes(include=["bool"]).columns:
        X[c] = X[c].astype(int)
    return X.astype(float)


def main():
    # Load recent features & determine feature set
    hist = read_recent_features(180)
    if TARGET not in hist.columns:
        raise SystemExit(f"Target '{TARGET}' missing from features parquet.")
    features = [c for c in hist.columns if c != TARGET]

    # Load quantile models
    models = {}
    for q in (10, 50, 90):
        p = ART / f"xgb_q{q}.json"
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run `make train-quantile`.")
        m = xgb.XGBRegressor()
        m.load_model(p.as_posix())
        models[q] = m

    last_ts = hist.index.max()
    start_future = last_ts + pd.Timedelta(hours=1)
    future_idx = pd.date_range(start_future, periods=24, freq="1h", tz="UTC")

    # Calendar + exogenous proxies (use recent hour profile means)
    cal = calendar_frame(start_future, 24)
    fut = pd.DataFrame(index=future_idx)
    for col in ["hour", "dow", "is_weekend", "is_holiday_de"]:
        if col in hist.columns:
            fut[col] = cal[col]
    # simple exogenous proxies (mean of last 28 days per hour)
    last28 = hist.loc[hist.index >= hist.index.max() - pd.Timedelta(days=28)]
    prof = last28.copy()
    prof.index = prof.index.tz_convert("Europe/Berlin")
    hour_mean = prof.groupby(prof.index.hour)[
        ["load_mw", "wind_mw", "solar_mw"]].mean()
    for ts in future_idx:
        h = ts.tz_convert("Europe/Berlin").hour
        for c in ["load_mw", "wind_mw", "solar_mw"]:
            if c in hist.columns and h in hour_mean.index:
                fut.loc[ts, c] = hour_mean.loc[h, c]
    if "renewables_share" in hist.columns:
        den = fut.get("load_mw", pd.Series(index=fut.index)).replace(0, np.nan)
        fut["renewables_share"] = (fut.get("wind_mw", 0).fillna(
            0) + fut.get("solar_mw", 0).fillna(0)) / den

    # Working frame with history (has lags/rollings) + future placeholders
    work = hist.copy()
    work = pd.concat([work, fut], axis=0).sort_index()

    def set_lags_rollings(ts):
        # price lags/rollings
        if "price_eur_mwh_lag1" in work.columns:
            work.loc[ts, "price_eur_mwh_lag1"] = work[TARGET].get(
                ts - pd.Timedelta(hours=1), np.nan)
        if "price_eur_mwh_lag24" in work.columns:
            work.loc[ts, "price_eur_mwh_lag24"] = work[TARGET].get(
                ts - pd.Timedelta(hours=24), np.nan)
        if "price_eur_mwh_lag48" in work.columns:
            work.loc[ts, "price_eur_mwh_lag48"] = work[TARGET].get(
                ts - pd.Timedelta(hours=48), np.nan)
        if "price_eur_mwh_lag168" in work.columns:
            work.loc[ts, "price_eur_mwh_lag168"] = work[TARGET].get(
                ts - pd.Timedelta(hours=168), np.nan)
        if "price_eur_mwh_roll24_mean" in work.columns:
            last24 = work[TARGET].loc[:ts - pd.Timedelta(hours=1)].tail(24)
            work.loc[ts, "price_eur_mwh_roll24_mean"] = last24.mean() if len(
                last24) > 0 else np.nan
        if "price_eur_mwh_roll168_mean" in work.columns:
            last168 = work[TARGET].loc[:ts - pd.Timedelta(hours=1)].tail(168)
            work.loc[ts, "price_eur_mwh_roll168_mean"] = last168.mean() if len(
                last168) > 0 else np.nan
        # load lags/rollings
        if "load_mw_lag1" in work.columns:
            work.loc[ts, "load_mw_lag1"] = work["load_mw"].get(
                ts - pd.Timedelta(hours=1), np.nan)
        if "load_mw_lag24" in work.columns:
            work.loc[ts, "load_mw_lag24"] = work["load_mw"].get(
                ts - pd.Timedelta(hours=24), np.nan)
        if "load_mw_lag48" in work.columns:
            work.loc[ts, "load_mw_lag48"] = work["load_mw"].get(
                ts - pd.Timedelta(hours=48), np.nan)
        if "load_mw_lag168" in work.columns:
            work.loc[ts, "load_mw_lag168"] = work["load_mw"].get(
                ts - pd.Timedelta(hours=168), np.nan)
        if "load_mw_roll24_mean" in work.columns:
            last24 = work["load_mw"].loc[:ts - pd.Timedelta(hours=1)].tail(24)
            work.loc[ts, "load_mw_roll24_mean"] = last24.mean() if len(
                last24) > 0 else np.nan
        if "load_mw_roll168_mean" in work.columns:
            last168 = work["load_mw"].loc[:ts -
                                          pd.Timedelta(hours=1)].tail(168)
            work.loc[ts, "load_mw_roll168_mean"] = last168.mean() if len(
                last168) > 0 else np.nan

    rows = []
    for ts in future_idx:
        set_lags_rollings(ts)
        # simple backfill for any remaining NaNs in exogenous/calendar
        for c in ["hour", "dow", "is_weekend", "is_holiday_de", "renewables_share", "load_mw", "wind_mw", "solar_mw"]:
            if c in work.columns and pd.isna(work.loc[ts, c]):
                prev = work[c].loc[:ts].dropna()
                work.loc[ts, c] = prev.iloc[-1] if not prev.empty else 0.0

        X_row = prepare_matrix(work.loc[[ts]], features)
        y_p10 = float(models[10].predict(X_row)[0])
        y_p50 = float(models[50].predict(X_row)[0])
        y_p90 = float(models[90].predict(X_row)[0])
        # use median to roll forward lags
        work.loc[ts, TARGET] = y_p50
        rows.append({"ts_utc": ts, "q10": y_p10, "q50": y_p50, "q90": y_p90})

    pred = pd.DataFrame(rows).set_index("ts_utc")
    pred.to_csv(OUT_CSV,  index_label="ts_utc")
    print("Saved CSV:", OUT_CSV)

    # plot: last 48h actual + ribbon
    plt.figure(figsize=(11, 4))
    hist_tail = hist[[TARGET]].loc[hist.index >=
                                   (last_ts - pd.Timedelta(hours=48))]
    if not hist_tail.empty:
        plt.plot(hist_tail.index, hist_tail[TARGET], label="Actual (last 48h)")
    plt.plot(pred.index, pred["q50"], "--o", label="Median (p50)")
    plt.fill_between(
        pred.index, pred["q10"], pred["q90"], alpha=0.25, label="p10–p90 interval")
    plt.title("Day-ahead probabilistic forecast (next 24h)")
    plt.xlabel("UTC time")
    plt.ylabel("EUR/MWh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    print("Saved plot:", OUT_PNG)


if __name__ == "__main__":
    main()
