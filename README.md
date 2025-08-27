# ⚡ German Energy Market Price Forecasting

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![Docker](https://img.shields.io/badge/docker-ready-blue)]()
[![Airflow](https://img.shields.io/badge/airflow-2.9-orange)]()
[![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red)]()

Forecasting **day-ahead electricity prices** in Germany with the impact of renewables (wind & solar).  
The project combines **data ingestion (OPSD, ENTSO-E, SMARD)**, **feature engineering**, **machine learning (XGBoost, quantile regression)**,  
and **automation with Airflow**, exposing both a **FastAPI service** and a **Streamlit dashboard**.

---

## 📌 Features

- 🔄 **Data ingestion** from OPSD, ENTSO-E, and SMARD APIs  
- 🛠️ **Feature engineering** with lagged values, rolling means, and calendar effects  
- 📈 **Baseline & Quantile models** (XGBoost) for point + probabilistic forecasts  
- 🔍 **Model explainability** via SHAP feature importances  
- 📊 **Visualization**: prediction plots, fan charts, calibration curves  
- ⚙️ **Automation**: Airflow DAGs for daily data fetch, feature building, training, validation  
- 🚀 **APIs & Dashboard**: FastAPI for predictions + Streamlit dashboard for exploration  

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Ingestion
        OPSD --> Raw[Postgres Raw Data]
        ENTSOE --> Raw
        SMARD --> Raw
    end

    subgraph Features
        Raw --> FE[Feature Engineering]
        FE --> PG[Postgres Features]
    end

    subgraph Modeling
        PG --> Train[Train Models]
        Train --> Metrics[Metrics + Artifacts]
        PG --> Predict[Predict Next 24h]
    end

    subgraph Serving
        Predict --> API[FastAPI]
        Predict --> Streamlit
    end

    subgraph Automation
        Airflow --> Ingestion
        Airflow --> Features
        Airflow --> Modeling
    end
```

---

## 🚀 Quickstart

```bash
# 1. Clone repo
git clone https://github.com/<your-username>/german-energy-forecasting.git
cd german-energy-forecasting/app

# 2. Copy env file and add secrets
cp .env.example .env
# Edit .env to add ENTSOE_TOKEN and DB settings

# 3. Start services
docker compose up -d

# 4. Run migrations
make migrate

# 5. Build features
make build-features

# 6. Train baseline model
make train-baseline

# 7. Predict next 24h
make forecast
```

---

## 📊 Example Results

| Metric | Value |
|--------|-------|
| MAE    | ~3.17 €/MWh |
| RMSE   | ~5.11 €/MWh |

## 📊 Results & Visuals

Here are some example outputs from the project:

| 24h Forecast | Probabilistic Fan Chart |
|--------------|--------------------------|
| ![Next24h](app/models/artifacts/predictions_next24h.png) | ![FanChart](app/models/artifacts/predictions_fan.png) |

| Feature Importance | SHAP Beeswarm |
|--------------------|----------------|
| ![Importance](app/models/artifacts/feature_importance.png) | ![SHAP](app/models/artifacts/shap_beeswarm.png) |


### Airflow DAG (ETL & Forecasting Pipeline)
![Airflow DAG](images/Airflow_DAG_Success_Run.png)

### FastAPI Endpoint (Serving Forecasts)
![FastAPI Endpoint](images/FastAPI_Endpoint.jpeg)

### Streamlit Dashboard (Visualization & Explainability)
![Streamlit Dashboard](images/Streamlit_Dashboard.png)


---

## 📅 Automation with Airflow

Airflow orchestrates the pipeline:

- **Fetch data** from OPSD, ENTSO-E, SMARD  
- **Build features** and load into Postgres  
- **Train models** and validate metrics  
- **Run data quality checks**  

Access UI: [http://localhost:8080](http://localhost:8080)  

---

## 📦 Tech Stack

- **Python** (pandas, scikit-learn, xgboost, shap)  
- **Docker** (multi-service setup)  
- **Postgres** for feature store  
- **Airflow** for orchestration  
- **FastAPI** (REST API for predictions)  
- **Streamlit** (interactive dashboard)  

---

## 🔑 Environment Variables

Create a `.env` file with:

```ini
POSTGRES_USER=epfd
POSTGRES_PASSWORD=epfd
POSTGRES_DB=epfd
POSTGRES_HOST=postgres

ENTSOE_TOKEN=your_api_key_here
```

---

## 📌 To Do / Future Work

- [ ] CRPS scoring for probabilistic forecasts  
- [ ] CI/CD pipeline (GitHub Actions)  
- [ ] Model registry integration (MLflow or similar)  
- [ ] Deploy dashboard online  

---

## 📜 License

MIT License © 2025 Firas Sakli




