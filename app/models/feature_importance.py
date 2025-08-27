# models/feature_importance.py
from pathlib import Path
import json, matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd

ART = Path("models/artifacts")
MODEL = ART / "xgb_baseline.json"
FIG = ART / "feature_importance.png"

def main():
    # Rebuild the exact feature list used in training
    df = pd.read_parquet("data/features/hourly.parquet")
    for c in df.select_dtypes(include=["bool"]).columns:
        df[c] = df[c].astype(int)
    df = df.select_dtypes(include=["number"])
    target = "price_eur_mwh"
    X = df[[c for c in df.columns if c != target]]

    model = xgb.XGBRegressor()
    model.load_model(MODEL.as_posix())

    # Get gain-based importance if available; fallback to weight
    booster = model.get_booster()
    fmap = {f"f{i}": name for i, name in enumerate(X.columns)}
    score = booster.get_score(importance_type="gain") or booster.get_score(importance_type="weight")
    # Map back to real names
    items = [(fmap.get(k, k), v) for k, v in score.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    names, vals = zip(*items) if items else ([], [])

    plt.figure(figsize=(8, 6))
    plt.barh(names[::-1], vals[::-1])
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(FIG, dpi=160)
    print("Saved:", FIG)

if __name__ == "__main__":
    main()
