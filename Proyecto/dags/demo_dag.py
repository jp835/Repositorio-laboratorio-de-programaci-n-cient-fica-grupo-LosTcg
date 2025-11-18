from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow

def simple_mlflow_task():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("demo")
    with mlflow.start_run():
        mlflow.log_param("param1", 10)
        mlflow.log_metric("accuracy", 0.95)

with DAG(
    "demo_mlflow_dag",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    t1 = PythonOperator(
        task_id="log_to_mlflow",
        python_callable=simple_mlflow_task
    )
