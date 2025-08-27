# web/app_streamlit.py
import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

ART = Path("models/artifacts")

st.set_page_config(page_title="German Energy Market Forecast", layout="wide")

st.title("⚡ German Energy Market Forecasting")
st.markdown("Probabilistic forecasts with XGBoost quantile regression.")

# Sidebar
page = st.sidebar.radio(
    "Navigation", ["Latest Data", "Forecast (Fan)", "Explainability", "Calibration"])

if page == "Latest Data":
    st.header("Recent features & prices")
    fea = Path("data/features/hourly.parquet")
    if fea.exists():
        df = pd.read_parquet(fea)

        # ensure datetime index
        if "ts_utc" in df.columns:
            df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
            df = df.set_index("ts_utc")
        else:
            # assume index already is datetime
            df.index = pd.to_datetime(df.index, utc=True)

        df = df.sort_index().last("7D")
        st.line_chart(df[["price_eur_mwh", "load_mw", "wind_mw", "solar_mw"]])
    else:
        st.error("Features parquet not found. Run `make build-features`.")

elif page == "Forecast (Fan)":
    st.header("Next 24h Forecast (Fan Chart)")
    fan_csv = ART / "predictions_fan.csv"
    fan_png = ART / "predictions_fan.png"
    if fan_csv.exists():
        df = pd.read_csv(fan_csv, parse_dates=["ts_utc"], index_col="ts_utc")
        st.line_chart(df[["q50"]])
        st.dataframe(df)
    if fan_png.exists():
        st.image(fan_png, caption="Fan forecast (p05–p95 intervals)")

elif page == "Explainability":
    st.header("Feature Importance & SHAP")
    imp = ART / "feature_importance.png"
    shap_bar = ART / "shap_summary_bar.png"
    shap_bee = ART / "shap_beeswarm.png"
    shap_wf = ART / "shap_waterfall_latest.png"
    for p in [imp, shap_bar, shap_bee, shap_wf]:
        if p.exists():
            st.image(p, caption=p.name)

elif page == "Calibration":
    st.header("Calibration Curve")
    cal = ART / "calibration_quantile_full.png"
    if cal.exists():
        st.image(cal, caption="Quantile calibration (last 30d)")
    else:
        st.error("Calibration plot missing. Run `make calibrate-quantiles-full`.")
