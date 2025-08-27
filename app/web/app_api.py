# web/app_api.py
from fastapi import FastAPI
import pandas as pd
import xgboost as xgb
from pathlib import Path
import os, psycopg2

app = FastAPI(title="Energy Forecast API")

ART = Path("models/artifacts")
FEA = Path("data/features/hourly.parquet")
TARGET = "price_eur_mwh"

def load_recent(days=7):
    df = pd.read_parquet(FEA)

    # ensure datetime index
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        df = df.set_index("ts_utc")
    else:
        # assume index already is datetime
        df.index = pd.to_datetime(df.index, utc=True)

    df = df.sort_index()
    return df.last(f"{days}D")

@app.get("/predict")
def predict_next24h():
    df = load_recent(180)
    features = [c for c in df.columns if c != TARGET]

    preds = {}
    for q in range(5, 100, 5):
        fname = ART / f"xgb_q{q}.json"
        if not fname.exists(): continue
        m = xgb.XGBRegressor(); m.load_model(fname.as_posix())
        X_last = df[features].iloc[[-1]].astype(float)
        preds[f"q{q}"] = float(m.predict(X_last)[0])

    return {"latest_timestamp": str(df.index[-1]), "forecast_next_hour": preds}

DB = dict(
    host=os.getenv("POSTGRES_HOST","epfd-postgres"),
    dbname=os.getenv("POSTGRES_DB","epfd"),
    user=os.getenv("POSTGRES_USER","epfd"),
    password=os.getenv("POSTGRES_PASSWORD","epfd")
)

@app.get("/predictions/next24h")
def predictions_next24h():
    sql = """
      SELECT ts_utc, y_p10, y_p50, y_p90
      FROM energy.predictions_hourly
      WHERE ts_utc >= now() AT TIME ZONE 'UTC'
      ORDER BY ts_utc ASC
      LIMIT 24;
    """
    with psycopg2.connect(**DB) as conn:
        df = pd.read_sql(sql, conn, parse_dates=["ts_utc"])
    return {"rows": df.to_dict(orient="records")}
