# db/migrations.py
import os
import psycopg2

DB_HOST = os.getenv("POSTGRES_HOST", "epfd-postgres")
DB_NAME = os.getenv("POSTGRES_DB", "epfd")
DB_USER = os.getenv("POSTGRES_USER", "epfd")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "epfd")

DDL = """
CREATE SCHEMA IF NOT EXISTS energy;

CREATE TABLE IF NOT EXISTS energy.predictions_hourly (
  ts_utc       TIMESTAMPTZ PRIMARY KEY,
  y_p50        DOUBLE PRECISION NOT NULL,
  y_p10        DOUBLE PRECISION,
  y_p90        DOUBLE PRECISION,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS energy.features_hourly (
    ts_utc           TIMESTAMPTZ PRIMARY KEY,
    price_eur_mwh    DOUBLE PRECISION,
    load_mw          DOUBLE PRECISION,
    wind_mw          DOUBLE PRECISION,
    solar_mw         DOUBLE PRECISION,
    renewables_share DOUBLE PRECISION
);

-- New columns required by loader
ALTER TABLE energy.features_hourly
    ADD COLUMN IF NOT EXISTS hour       SMALLINT,
    ADD COLUMN IF NOT EXISTS dow        SMALLINT,
    ADD COLUMN IF NOT EXISTS is_weekend BOOLEAN;

-- Helpful index (safe to re-run)
CREATE INDEX IF NOT EXISTS idx_features_hourly_ts ON energy.features_hourly (ts_utc);
"""


def main():
    conn = psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )
    conn.autocommit = True
    with conn, conn.cursor() as cur:
        cur.execute(DDL)
    conn.close()
    print("✅ Migrations applied: schema 'energy' and table 'features_hourly' ensured.")


if __name__ == "__main__":
    main()
