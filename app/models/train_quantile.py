# models/train_quantile.py
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

ART = Path("models/artifacts"); ART.mkdir(parents=True, exist_ok=True)
FEA = Path("data/features/hourly.parquet")
TARGET = "price_eur_mwh"

def load_data():
    df = pd.read_parquet(FEA)
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        df = df.set_index("ts_utc").sort_index()
    for c in df.select_dtypes(include=["bool"]).columns:
        df[c] = df[c].astype(int)
    df = df.select_dtypes(include=["number"]).dropna()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y

def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def train_quantile(X, y, q: float, n_splits=5):
    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": q,
        "eval_metric": "mae",
        "n_estimators": 400,
        "learning_rate": 0.06,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []
    for tr, te in tscv.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        maes.append(mean_absolute_error(y_te, y_hat))
        rmses.append(_rmse(y_te, y_hat))
    # fit on all data
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model, {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses))}

def main():
    X, y = load_data()
    quantiles = [0.1, 0.5, 0.9]
    metrics = {}
    for q in quantiles:
        model, m = train_quantile(X, y, q)
        fname = ART / f"xgb_q{int(q*100)}.json"
        model.save_model(fname.as_posix())
        metrics[f"q{int(q*100)}"] = m
        print(f"Trained q={q:.1f} → {fname.name}, MAE={m['mae']:.3f}, RMSE={m['rmse']:.3f}")
    (ART / "metrics_quantile.json").write_text(json.dumps(metrics, indent=2))
    print("Saved metrics:", ART / "metrics_quantile.json")

if __name__ == "__main__":
    main()
