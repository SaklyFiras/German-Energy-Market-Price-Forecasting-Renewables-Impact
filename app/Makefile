# ---------- Paths ----------
COMPOSE = docker compose -f docker/docker-compose.yml

# ---------- One-time setup ----------
.PHONY: deps
deps:  ## Install Python deps inside 'py' container
	$(COMPOSE) exec py pip install -r requirements.txt

.PHONY: pg-driver
pg-driver: ## Install Postgres driver for the loader
	$(COMPOSE) exec py pip install psycopg2-binary

# ---------- Data bootstrap ----------
.PHONY: fetch-opsd
fetch-opsd: ## Download & save OPSD bootstrap parquet
	$(COMPOSE) exec py python ingestion/fetch_opsd.py

.PHONY: build-features
build-features: ## Build hourly features parquet
	$(COMPOSE) exec py python features/build_features.py

# ---------- Database ----------
.PHONY: migrate
migrate: ## Apply SQL schema inside Postgres
	MSYS2_ARG_CONV_EXCL=/repo/** $(COMPOSE) exec -T postgres sh -lc 'psql -U $${POSTGRES_USER:-epfd} -d $${POSTGRES_DB:-epfd} -f /repo/db/migrations.sql'

.PHONY: load-features
load-features: ## Load parquet features into Postgres
	$(COMPOSE) exec py python db/load_features_to_pg.py

.PHONY: verify-features
verify-features: ## Quick check: show 5 latest feature rows
	docker exec -it epfd-postgres psql -U $${POSTGRES_USER:-epfd} -d $${POSTGRES_DB:-epfd} -c "SELECT ts_utc, price_eur_mwh, renewables_share FROM energy.features_hourly ORDER BY ts_utc DESC LIMIT 5;"

# ---------- Quality of life ----------
.PHONY: up down logs ps jupyter psql
up:       ## Start all containers
	$(COMPOSE) up -d
down:     ## Stop containers (keep volumes)
	$(COMPOSE) down
logs:     ## Tail docker logs
	$(COMPOSE) logs -f --tail=200
ps:       ## Show running services
	$(COMPOSE) ps
jupyter:  ## Open Jupyter URL hint
	@echo "Jupyter Lab → http://localhost:8888  (token=$$JUPYTER_TOKEN or 'dev')"
psql:     ## Open interactive psql
	docker exec -it epfd-postgres psql -U $${POSTGRES_USER:-epfd} -d $${POSTGRES_DB:-epfd}

# ---------- ENTSO-E (requires ENTSOE_TOKEN in .env) ----------
.PHONY: fetch-entsoe
fetch-entsoe:
	$(COMPOSE) exec py python ingestion/fetch_entsoe.py

.PHONY: backfill-month
backfill-month: ## Backfill ENTSO-E DA prices for a month (YEAR=2025 MONTH=08)
	@if [ -z "$(YEAR)" ] || [ -z "$(MONTH)" ]; then echo "Usage: make backfill-month YEAR=2025 MONTH=08"; exit 1; fi
	$(COMPOSE) exec py python ingestion/fetch_entsoe.py --start $(YEAR)-$(MONTH)-01 --end $$(date -u -d "$(YEAR)-$(MONTH)-01 +1 month" +"%Y-%m-%d")
	$(MAKE) build-features
	$(MAKE) load-features
	$(MAKE) verify-features

# ---------- SMARD ----------
.PHONY: fetch-smard
fetch-smard: ## Fetch SMARD load/wind/solar to Parquet
	$(COMPOSE) exec py python ingestion/fetch_smard.py

.PHONY: fetch-smard-qh
fetch-smard-qh: ## Fetch SMARD quarter-hour → aggregate hourly
	$(COMPOSE) exec py python ingestion/fetch_smard.py --resolution quarterhour --save-qh

.PHONY: fetch-smard-last2y
fetch-smard-last2y: ## SMARD last 2 years (hourly)
	$(COMPOSE) exec py python ingestion/fetch_smard.py --limit-years 2

.PHONY: fetch-smard-range
fetch-smard-range: ## SMARD custom date range (START=YYYY-MM-DD END=YYYY-MM-DD)
	@if [ -z "$(START)" ] || [ -z "$(END)" ]; then echo "Usage: make fetch-smard-range START=2024-01-01 END=2024-12-31"; exit 1; fi
	$(COMPOSE) exec py python ingestion/fetch_smard.py --start $(START) --end $(END)


.PHONY: refresh-smard
refresh-smard: ## SMARD -> build features -> load -> verify
	$(MAKE) fetch-smard
	$(MAKE) build-features
	$(MAKE) load-features
	$(MAKE) verify-features



# ---------- End-to-end refresh ----------
.PHONY: refresh
refresh: ## Fetch latest (ENTSO-E) -> build features -> load to Postgres
	$(MAKE) fetch-entsoe
	$(MAKE) build-features
	$(MAKE) load-features
	$(MAKE) verify-features


.PHONY: train-baseline show-metrics
train-baseline: ## Train XGBoost baseline with walk-forward CV
	$(COMPOSE) exec py python models/train_baseline.py

.PHONY: show-metrics show-model
show-metrics: ## Print saved metrics (works on Windows/Linux/macOS)
	$(COMPOSE) exec py bash -lc 'cat /app/models/artifacts/metrics.json'

show-model: ## List saved model file
	$(COMPOSE) exec py bash -lc 'ls -lh /app/models/artifacts/xgb_baseline.json'

.PHONY: importance
importance: ## Save XGBoost feature importance plot
	$(COMPOSE) exec py python models/feature_importance.py


.PHONY: forecast
forecast: ## Predict next 24h (CSV + PNG)
	$(COMPOSE) exec py python models/predict_next_24h.py

.PHONY: shap-global
shap-global: ## SHAP: summary bar + beeswarm + latest waterfall
	$(COMPOSE) exec py python models/shap_analysis.py

.PHONY: train-quantile
train-quantile:
	$(COMPOSE) exec py python models/train_quantile.py

.PHONY: forecast-quantile
forecast-quantile:
	$(COMPOSE) exec py python models/predict_next_24h.py

.PHONY: backtest-quantile
backtest-quantile:
	$(COMPOSE) exec py python models/backtest_quantile.py

.PHONY: calibrate-quantile
calibrate-quantile:
	$(COMPOSE) exec py python models/calibration_quantile.py

.PHONY: train-quantiles-full
train-quantiles-full:
	$(COMPOSE) exec py python models/train_quantiles_full.py

.PHONY: forecast-fan
forecast-fan:
	$(COMPOSE) exec py python models/predict_fan.py

.PHONY: calibrate-quantiles-full
calibrate-quantiles-full:
	$(COMPOSE) exec py python models/calibration_quantile_full.py




.PHONY: web-ui
web-ui:
	$(COMPOSE) exec py streamlit run web/app_streamlit.py --server.address=0.0.0.0 --server.port=8501

.PHONY: api
api:
	$(COMPOSE) exec py uvicorn web.app_api:app --host 0.0.0.0 --port 8000


# -------------------
# Airflow management
# -------------------

AIRFLOW_COMPOSE = docker compose -f airflow/docker-compose.yml

.PHONY: airflow-init
airflow-init:
	$(AIRFLOW_COMPOSE) up airflow-init

.PHONY: airflow-up
airflow-up:
	$(AIRFLOW_COMPOSE) up -d

.PHONY: airflow-down
airflow-down:
	$(AIRFLOW_COMPOSE) down

.PHONY: airflow-logs
airflow-logs:
	$(AIRFLOW_COMPOSE) logs -f

.PHONY: airflow-test
airflow-test:
	$(AIRFLOW_COMPOSE) exec airflow-webserver airflow dags trigger -d energy_forecast



