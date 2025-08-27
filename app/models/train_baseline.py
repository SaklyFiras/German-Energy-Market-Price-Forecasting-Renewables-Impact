# models/train_baseline.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

FEA = Path("data/features/hourly.parquet")
ART = Path("models/artifacts"); ART.mkdir(parents=True, exist_ok=True)
TARGET = "price_eur_mwh"

def make_dataset(df: pd.DataFrame):
    y = df[TARGET].astype(float)
    features = [c for c in df.columns if c not in [TARGET]]
    X = df[features].astype(float)
    return X, y, features

def walk_forward_eval(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, models = [], [], []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=4,
            random_state=42
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        y_hat = model.predict(X_te)

        mae  = mean_absolute_error(y_te, y_hat)
        mse  = mean_squared_error(y_te, y_hat)         # <-- keep plain MSE
        rmse = float(np.sqrt(mse))                      # <-- compute RMSE manually
        maes.append(float(mae)); rmses.append(rmse); models.append(model)
        print(f"Fold {fold}: MAE={mae:.3f}, RMSE={rmse:.3f}")
    return models[-1], float(np.mean(maes)), float(np.mean(rmses))

def main():
    if not FEA.exists():
        raise SystemExit("Missing features parquet. Run features/build_features.py")

    df = pd.read_parquet(FEA)
    # Keep numeric columns only (convert bools to ints)
    for c in df.select_dtypes(include=["bool"]).columns:
        df[c] = df[c].astype(int)
    num_cols = df.select_dtypes(include=["number"]).columns
    df = df[num_cols].copy()
    if TARGET not in df.columns:
        raise SystemExit(f"Target '{TARGET}' missing from features.")

    # Drop any remaining NaNs just in case
    df = df.dropna().copy()

    X, y, feats = make_dataset(df)
    model, mae, rmse = walk_forward_eval(X, y, n_splits=5)

    ART.mkdir(parents=True, exist_ok=True)
    model.save_model((ART / "xgb_baseline.json").as_posix())
    (ART / "metrics.json").write_text(json.dumps(
        {"mae": mae, "rmse": rmse, "n_features": len(feats)}, indent=2
    ))
    print("Saved model:", ART / "xgb_baseline.json")
    print("Metrics:", {"mae": mae, "rmse": rmse})

if __name__ == "__main__":
    main()
