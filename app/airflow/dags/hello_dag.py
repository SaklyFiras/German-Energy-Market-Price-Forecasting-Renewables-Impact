from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "firas",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="hello_dag",
    default_args=default_args,
    description="Simple test DAG to confirm Airflow is working",
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    hello = BashOperator(
        task_id="say_hello",
        bash_command="echo 'Hello from Airflow!'"
    )
