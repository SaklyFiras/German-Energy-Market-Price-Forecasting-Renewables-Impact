from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

default_args = {
    "owner": "firas",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="energy_forecast",
    default_args=default_args,
    description="Daily German energy market forecast",
    schedule_interval="0 6 * * *",   # 06:00 UTC daily
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
) as dag:

    migrate = BashOperator(
        task_id="migrate",
        # or `psql -f db/migrations.sql` depending on your setup
        bash_command="cd /app && python db/migrations.py"
    )

    fetch_smard = BashOperator(
        task_id="fetch_smard",
        bash_command="cd /app && python ingestion/fetch_smard.py --start {{ ds }} --end {{ next_ds }}"
    )

    # add ENTSO-E fetch if you like (token comes from env)
    # fetch_entsoe = BashOperator(
    #     task_id="fetch_entsoe",
    #     bash_command="cd /app && python ingestion/fetch_entsoe.py"
    # )

    build_features = BashOperator(
        task_id="build_features",
        bash_command="cd /app && python features/build_features.py"
    )

    load_features = BashOperator(
        task_id="load_features",
        bash_command="cd /app && python db/load_features_to_pg.py"
    )

    dq_count = SQLExecuteQueryOperator(
        task_id="dq_rowcount",
        # create an Airflow Connection for your DB, or switch to env string

        conn_id="epfd_postgres",
        sql="SELECT COUNT(*) FROM energy.features_hourly WHERE ts_utc >= now() - interval '2 days';",
    )
    dq_no_dupes = SQLExecuteQueryOperator(
        task_id="dq_no_dupes",
        conn_id="epfd_postgres",
        sql="""
        WITH d AS (
        SELECT ts_utc, COUNT(*) c
        FROM energy.features_hourly
        GROUP BY 1 HAVING COUNT(*) > 1
        )
        SELECT CASE 
                WHEN COUNT(*) > 0 THEN NULL 
                ELSE 1 
            END
        FROM d;
        """
    )

    train_quantiles = BashOperator(
        task_id="train_quantiles",
        bash_command="cd /app && python models/train_quantiles_full.py"
    )

    forecast_fan = BashOperator(
        task_id="forecast_fan",
        bash_command="cd /app && python models/predict_fan.py"
    )
    save_predictions = BashOperator(
        task_id="save_predictions",
        bash_command="cd /app && python db/save_predictions.py"
    )

fetch_smard >> build_features >> migrate >> load_features >> [
    dq_count, dq_no_dupes] >> train_quantiles >> forecast_fan >> save_predictions
