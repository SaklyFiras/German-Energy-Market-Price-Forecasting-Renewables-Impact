# db/save_predictions.py
import os, psycopg2
from psycopg2.extras import execute_values
import pandas as pd

DB = dict(
    host=os.getenv("POSTGRES_HOST","epfd-postgres"),
    dbname=os.getenv("POSTGRES_DB","epfd"),
    user=os.getenv("POSTGRES_USER","epfd"),
    password=os.getenv("POSTGRES_PASSWORD","epfd")
)

def save_fan(csv_path="models/artifacts/predictions_fan.csv"):
    df = pd.read_csv(csv_path, parse_dates=["ts_utc"])
    rows = [(r.ts_utc, r.y_p50, r.get("y_p10"), r.get("y_p90")) for r in df.to_dict("records")]
    sql = """
    INSERT INTO energy.predictions_hourly (ts_utc, y_p50, y_p10, y_p90)
    VALUES %s
    ON CONFLICT (ts_utc) DO UPDATE
      SET y_p50=EXCLUDED.y_p50, y_p10=EXCLUDED.y_p10, y_p90=EXCLUDED.y_p90, created_at=now();
    """
    with psycopg2.connect(**DB) as conn, conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)
    print(f"Saved {len(rows)} rows to energy.predictions_hourly")

if __name__ == "__main__":
    save_fan()
