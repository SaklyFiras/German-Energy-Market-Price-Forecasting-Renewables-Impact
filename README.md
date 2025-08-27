⚡ German Energy Market Price Forecasting & Renewables Impact

Forecasting day-ahead electricity prices in Germany 🇩🇪, with a focus on the impact of renewable generation (wind & solar) and demand dynamics.
This project combines data pipelines, machine learning, and automation into a production-style workflow.

📖 Overview

Predicts hourly electricity prices using demand, wind, solar, and calendar features.

Supports probabilistic forecasts (quantile regression with XGBoost) to capture uncertainty.

Provides interpretability with feature importance and SHAP values.

Automates pipelines with Apache Airflow.

Interactive Streamlit dashboard and FastAPI API for serving forecasts.

⚡ Features

✅ Data ingestion from OPSD, ENTSO-E, SMARD
✅ Feature engineering (lags, rolling means, calendar effects, holidays)
✅ Baseline model: XGBoost with MAE ≈ 3.2 €/MWh
✅ Probabilistic forecasts (10%, 50%, 90%)
✅ SHAP-based interpretability per prediction
✅ Airflow DAGs for daily automation
✅ Streamlit dashboard + REST API

🛠️ Tech Stack

Python: pandas, scikit-learn, xgboost, shap, psycopg2

Databases: PostgreSQL

Deployment: Docker + Docker Compose

Visualization: Streamlit, matplotlib

Automation: Apache Airflow

API: FastAPI

🚀 Getting Started
1. Clone Repository
git clone https://github.com/YOURNAME/German-Energy-Forecasting.git
cd German-Energy-Forecasting/app

2. Configure Environment

Copy .env.example → .env and add your credentials:

ENTSOE_TOKEN=your_entsoe_api_key
POSTGRES_USER=epfd
POSTGRES_PASSWORD=epfd
POSTGRES_DB=epfd

3. Start Services
docker compose -f docker/docker-compose.yml up -d --build

4. Run Pipelines
# Apply database migrations
make migrate

# Fetch sample OPSD data
make fetch-opsd

# Build & load features
make build-features
make load-features

# Train baseline model
make train-baseline
make show-metrics

5. Interactive Apps

Streamlit dashboard → http://localhost:8501

FastAPI endpoint → http://localhost:8000/docs

Airflow UI → http://localhost:8080

📊 Outputs
Forecasts

Next 24h predictions → predictions_next24h.csv + plot

SHAP Analysis

Feature importance

Local explanations per timestamp

Quantile Fan Forecasts

10% – 50% – 90% intervals

📅 Automation with Airflow

Airflow DAG: energy_forecast_dag.py

Tasks:

Fetch new data (OPSD / ENTSO-E / SMARD)

Build & load features into Postgres

Train & evaluate models

Forecast next 24h

Run data quality checks (row counts, duplicates)

Run:

cd app/airflow
docker compose up -d


Airflow UI → http://localhost:8080

📈 Results

Baseline XGBoost: MAE ≈ 3.2 €/MWh, RMSE ≈ 5.1 €/MWh

Renewable share strongly impacts price volatility.

SHAP values show:

High load → price ↑

High renewables share → price ↓

🔑 Why This Matters

Energy prices are highly volatile in renewable-heavy grids.

Probabilistic forecasts provide uncertainty bounds (crucial for trading, grid balancing, and risk management).

This project demonstrates how to productionize ML for energy systems — from raw data → features → models → API → automated pipelines.

📄 License

MIT License © 2025 Firas

✨ If you like this project, ⭐ star the repo and share it!