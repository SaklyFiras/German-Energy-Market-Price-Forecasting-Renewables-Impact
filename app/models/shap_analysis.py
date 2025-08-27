# models/shap_analysis.py
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

ART = Path("models/artifacts"); ART.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ART / "xgb_baseline.json"
FEA_PATH = Path("data/features/hourly.parquet")
TARGET = "price_eur_mwh"

def load_data():
    df = pd.read_parquet(FEA_PATH)
    # make sure index is ts_utc
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        df = df.set_index("ts_utc").sort_index()
    for c in df.select_dtypes(include=["bool"]).columns:
        df[c] = df[c].astype(int)
    df = df.select_dtypes(include=["number"]).dropna()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return df, X, y

def main():
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Train first (`make train-baseline`).")
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH.as_posix())

    df, X, y = load_data()

    # Recent 7 days via .loc mask (no FutureWarning)
    end = df.index.max()
    start = end - pd.Timedelta(days=7)
    recent = X.loc[start:end]
    if recent.empty:
        recent = X.tail(1000)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(recent)

    # 1) Summary bar (mean |SHAP|)
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values, recent, show=False, plot_type="bar", max_display=20)
    plt.tight_layout()
    bar_path = ART / "shap_summary_bar.png"
    plt.savefig(bar_path, dpi=160)
    print("Saved:", bar_path)

    # 2) Beeswarm
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values, recent, show=False, max_display=20)
    plt.tight_layout()
    swarm_path = ART / "shap_beeswarm.png"
    plt.savefig(swarm_path, dpi=160)
    print("Saved:", swarm_path)

    # 3) Waterfall for most recent row
    latest_x = recent.tail(1)
    latest_sv = explainer.shap_values(latest_x)

    # expected_value can be array-like; latest_sv is (1, n_features) → flatten to 1D
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.array(base).reshape(-1)[0])
    latest_sv_1d = np.array(latest_sv).reshape(-1)

    shap.plots._waterfall.waterfall_legacy(
        base, latest_sv_1d,
        feature_names=list(latest_x.columns),
        max_display=20, show=False
    )
    plt.tight_layout()
    wf_path = ART / "shap_waterfall_latest.png"
    plt.savefig(wf_path, dpi=160)
    print("Saved:", wf_path)

if __name__ == "__main__":
    main()
