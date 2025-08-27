# db/load_features_to_pg.py
import os
from pathlib import Path
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Config from env (docker-compose .env)
PG_USER = os.getenv("POSTGRES_USER", "epfd")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "epfd")
PG_DB   = os.getenv("POSTGRES_DB", "epfd")
PG_HOST = os.getenv("POSTGRES_HOST", "postgres")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

PARQUET_PATH = Path("data/features/hourly.parquet")

def main():
    if not PARQUET_PATH.exists():
        raise SystemExit(f"Missing {PARQUET_PATH}. Run features/build_features.py first.")

    df = pd.read_parquet(PARQUET_PATH)
    required = [
        "price_eur_mwh","load_mw","wind_mw",
        "solar_mw","renewables_share","hour","dow","is_weekend"
    ]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"Expected column '{c}' not found in features parquet.")

    df = df.reset_index().rename(columns={"ts_utc":"ts_utc"})
    if "ts_utc" not in df.columns:
        raise SystemExit("Could not find 'ts_utc' column in features data.")

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    rows = list(df[
        ["ts_utc","price_eur_mwh","load_mw","wind_mw",
         "solar_mw","renewables_share","hour","dow","is_weekend"]
    ].itertuples(index=False, name=None))

    sql = '''
    INSERT INTO energy.features_hourly
    (ts_utc, price_eur_mwh, load_mw, wind_mw, solar_mw,
     renewables_share, hour, dow, is_weekend)
    VALUES %s
    ON CONFLICT (ts_utc) DO UPDATE SET
      price_eur_mwh = EXCLUDED.price_eur_mwh,
      load_mw = EXCLUDED.load_mw,
      wind_mw = EXCLUDED.wind_mw,
      solar_mw = EXCLUDED.solar_mw,
      renewables_share = EXCLUDED.renewables_share,
      hour = EXCLUDED.hour,
      dow = EXCLUDED.dow,
      is_weekend = EXCLUDED.is_weekend;
    '''

    conn = psycopg2.connect(
        user=PG_USER, password=PG_PASS,
        dbname=PG_DB, host=PG_HOST, port=PG_PORT
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=10000)
    conn.close()
    print(f"Upserted {len(rows)} rows into energy.features_hourly.")

if __name__ == "__main__":
    main()
