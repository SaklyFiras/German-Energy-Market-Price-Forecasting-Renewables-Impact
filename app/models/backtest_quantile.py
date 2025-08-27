# models/backtest_quantile.py
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import json

ART = Path("models/artifacts"); ART.mkdir(parents=True, exist_ok=True)
FEA = Path("data/features/hourly.parquet")
TARGET = "price_eur_mwh"
OUT = ART / "backtest_quantile.json"

def load_data():
    df = pd.read_parquet(FEA)
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        df = df.set_index("ts_utc").sort_index()
    for c in df.select_dtypes(include=["bool"]).columns:
        df[c] = df[c].astype(int)
    df = df.select_dtypes(include=["number"]).dropna()
    return df

def main():
    df = load_data()
    features = [c for c in df.columns if c != TARGET]
    # train/test split: last 30 days as test
    cutoff = df.index.max() - pd.Timedelta(days=30)
    train = df.loc[df.index <= cutoff]
    test = df.loc[df.index > cutoff]

    models = {}
    for q in (10,50,90):
        fname = ART / f"xgb_q{q}.json"
        if not fname.exists():
            raise SystemExit(f"Missing {fname}. Run `make train-quantile`.")
        m = xgb.XGBRegressor()
        m.load_model(fname.as_posix())
        models[q] = m

    X_test = test[features].astype(float)
    y_test = test[TARGET]

    preds = {}
    for q, m in models.items():
        preds[q] = pd.Series(m.predict(X_test), index=X_test.index, name=f"q{q}")

    pred_df = pd.concat(preds.values(), axis=1)

    # Coverage: % of actuals inside [q10, q90]
    inside = (y_test >= pred_df["q10"]) & (y_test <= pred_df["q90"])
    coverage = inside.mean()

    # Sharpness: average interval width
    width = (pred_df["q90"] - pred_df["q10"]).mean()

    # Median error
    mae_p50 = mean_absolute_error(y_test, pred_df["q50"])

    results = {
        "days_tested": int((test.index.max() - cutoff).days),
        "coverage": float(coverage),
        "expected_coverage": 0.8,
        "avg_interval_width": float(width),
        "mae_p50": float(mae_p50)
    }

    OUT.write_text(json.dumps(results, indent=2))
    print("Saved backtest:", OUT)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
