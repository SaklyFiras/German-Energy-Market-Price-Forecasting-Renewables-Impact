# # airflow/dags/_alerts.py
# import os, requests
# def on_fail(context):
#     url = os.environ.get("SLACK_WEBHOOK")
#     if not url: return
#     ti = context["ti"]
#     msg = f":rotating_light: Task *{ti.task_id}* failed in DAG *{ti.dag_id}* (run_id={ti.run_id})."
#     requests.post(url, json={"text": msg})
