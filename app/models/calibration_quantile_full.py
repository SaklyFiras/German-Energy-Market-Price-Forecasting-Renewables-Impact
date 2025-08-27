# models/calibration_quantile_full.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

ART = Path("models/artifacts"); ART.mkdir(parents=True, exist_ok=True)
FEA = Path("data/features/hourly.parquet")
TARGET = "price_eur_mwh"

OUT_JSON = ART / "calibration_quantile_full.json"
OUT_PNG = ART / "calibration_quantile_full.png"

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

    X_test, y_test = test[features].astype(float), test[TARGET]
    results = {}

    qs = [q/100 for q in range(5, 100, 5)]
    emp = []
    for q in qs:
        fname = ART / f"xgb_q{int(q*100)}.json"
        if not fname.exists():
            raise SystemExit(f"Missing {fname}, run train_quantiles_full.py.")
        m = xgb.XGBRegressor(); m.load_model(fname.as_posix())
        yhat = m.predict(X_test)
        emp.append(np.mean(y_test <= yhat))
    results = {"nominal": qs, "empirical": emp}
    pd.DataFrame(results).to_json(OUT_JSON, indent=2)

    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1],"k--", label="Perfect calibration")
    plt.plot(qs, emp, "o-", label="Model calibration")
    plt.xlabel("Nominal quantile")
    plt.ylabel("Observed frequency")
    plt.title("Quantile calibration curve (last 30 days)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    print("Saved:", OUT_PNG)

if __name__ == "__main__":
    main()
