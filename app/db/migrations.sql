-- db/migrations.sql
-- Postgres schemas and tables for German Energy Market Forecasting (DE-LU)

-- Create schema
CREATE SCHEMA IF NOT EXISTS energy;

-- =========================
-- Raw tables (as-landed)
-- =========================

-- ENTSOE day-ahead prices (DE-LU)
CREATE TABLE IF NOT EXISTS energy.raw_entsoe_day_ahead_price (
    ts_utc          TIMESTAMP WITH TIME ZONE NOT NULL,
    price_eur_mwh   NUMERIC,
    currency        TEXT,
    bidding_zone    TEXT DEFAULT 'DE-LU',
    raw_payload     JSONB,
    ingest_ts       TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (ts_utc, bidding_zone)
);

-- SMARD load (total consumption)
CREATE TABLE IF NOT EXISTS energy.raw_smard_load (
    ts_utc        TIMESTAMP WITH TIME ZONE NOT NULL,
    load_mw       NUMERIC,
    region        TEXT DEFAULT 'DE',
    raw_payload   JSONB,
    ingest_ts     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (ts_utc, region)
);

-- SMARD generation by source (wind)
CREATE TABLE IF NOT EXISTS energy.raw_smard_gen_wind (
    ts_utc        TIMESTAMP WITH TIME ZONE NOT NULL,
    wind_mw       NUMERIC,
    region        TEXT DEFAULT 'DE',
    raw_payload   JSONB,
    ingest_ts     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (ts_utc, region)
);

-- SMARD generation by source (solar)
CREATE TABLE IF NOT EXISTS energy.raw_smard_gen_solar (
    ts_utc        TIMESTAMP WITH TIME ZONE NOT NULL,
    solar_mw      NUMERIC,
    region        TEXT DEFAULT 'DE',
    raw_payload   JSONB,
    ingest_ts     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (ts_utc, region)
);

-- =========================
-- Feature store (hourly, aligned to CET/CEST index)
-- =========================
CREATE TABLE IF NOT EXISTS energy.features_hourly (
    ts_utc              TIMESTAMP WITH TIME ZONE PRIMARY KEY,
    ts_cet              TIMESTAMP WITHOUT TIME ZONE,
    price_eur_mwh       NUMERIC,
    load_mw             NUMERIC,
    wind_mw             NUMERIC,
    solar_mw            NUMERIC,
    renewables_share    NUMERIC, -- (wind + solar) / total_gen (when available)
    hour                SMALLINT,
    dow                 SMALLINT,
    is_weekend          BOOLEAN,
    is_holiday_de       BOOLEAN,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =========================
-- Metadata / model registry (lightweight)
-- =========================
CREATE TABLE IF NOT EXISTS energy.model_registry (
    model_id        TEXT PRIMARY KEY,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    algo            TEXT,
    params          JSONB,
    train_start     TIMESTAMP WITH TIME ZONE,
    train_end       TIMESTAMP WITH TIME ZONE,
    mae             NUMERIC,
    rmse            NUMERIC,
    p90_abs_err     NUMERIC,
    notes           TEXT
);

-- =========================
-- Predictions (served or batch)
-- =========================
CREATE TABLE IF NOT EXISTS energy.predictions (
    ts_utc              TIMESTAMP WITH TIME ZONE NOT NULL,
    yhat_eur_mwh        NUMERIC,
    model_id            TEXT REFERENCES energy.model_registry(model_id),
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (ts_utc, model_id)
);

-- Helpful index for time-range queries
CREATE INDEX IF NOT EXISTS idx_features_hourly_ts ON energy.features_hourly(ts_utc);
CREATE INDEX IF NOT EXISTS idx_raw_entsoe_ts ON energy.raw_entsoe_day_ahead_price(ts_utc);
CREATE INDEX IF NOT EXISTS idx_raw_smard_load_ts ON energy.raw_smard_load(ts_utc);
CREATE INDEX IF NOT EXISTS idx_raw_smard_wind_ts ON energy.raw_smard_gen_wind(ts_utc);
CREATE INDEX IF NOT EXISTS idx_raw_smard_solar_ts ON energy.raw_smard_gen_solar(ts_utc);
