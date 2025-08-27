# models/calibration_quantile.py
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

ART = Path("models/artifacts"); ART.mkdir(parents=True, exist_ok=True)
FEA = Path("data/features/hourly.parquet")
TARGET = "price_eur_mwh"

OUT_JSON = ART / "calibration_quantile.json"
OUT_PNG = ART / "calibration_quantile.png"

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
    cutoff = df.index.max() - pd.Timedelta(days=30)
    test = df.loc[df.index > cutoff]

    models = {}
    quantiles = [0.1, 0.5, 0.9]
    for q in quantiles:
        fname = ART / f"xgb_q{int(q*100)}.json"
        if not fname.exists():
            raise SystemExit(f"Missing {fname}. Run `make train-quantile`.")
        m = xgb.XGBRegressor(); m.load_model(fname.as_posix())
        models[q] = m

    X_test = test[features].astype(float)
    y_test = test[TARGET]

    results = {}
    for q, m in models.items():
        yhat = m.predict(X_test)
        emp = np.mean(y_test <= yhat)   # empirical coverage
        results[f"q{int(q*100)}"] = {"nominal": q, "empirical": float(emp)}

    # Save JSON
    pd.DataFrame(results).T.to_json(OUT_JSON, indent=2)
    print("Saved:", OUT_JSON)

    # Plot
    qs = [r["nominal"] for r in results.values()]
    obs = [r["empirical"] for r in results.values()]
    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1],"k--", label="Perfect calibration")
    plt.plot(qs, obs, "o-", label="Model calibration")
    plt.xlabel("Nominal quantile")
    plt.ylabel("Observed frequency")
    plt.title("Quantile calibration (last 30 days)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    print("Saved:", OUT_PNG)

if __name__ == "__main__":
    main()
