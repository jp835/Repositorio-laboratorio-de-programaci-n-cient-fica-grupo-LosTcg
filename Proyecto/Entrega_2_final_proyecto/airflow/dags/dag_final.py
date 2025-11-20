from scripts.prepare_data import build_datasets_callable
from scripts.train_model_final import train_model_completo
from scripts.generate_predictions import generate_predictions_callable
from scripts.split_data import split_and_save_train_test
from scripts.create_folder import create_folders

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

from datetime import datetime

with DAG(
    dag_id='Pipeline_proyecto',
    description='Pipeline para prediccion de compra',
    start_date=datetime(2025, 10, 1),
    schedule_interval= '@daily',          
    catchup=False
) as dag:

    inicio = EmptyOperator(task_id='iniciar_pipeline')

    crear_carpetas = PythonOperator(
        task_id='crear_carpetas',
        python_callable=create_folders,
        provide_context=True
    )

    construir_data_set = PythonOperator(
        task_id='construir_data_set',
        python_callable=build_datasets_callable,
        op_kwargs={'execution_date': '{{ ds }}'}
    )


    separar_data = PythonOperator(
        task_id='separar_data',
        python_callable=split_and_save_train_test,
        op_kwargs={'execution_date': '{{ ds }}'}
    )

    entrenamiento = PythonOperator(
        task_id='preprocess_and_train',
        python_callable=train_model_completo,
        op_kwargs={'execution_date': '{{ ds }}'}
    )

    prediccion = PythonOperator(
        task_id='predicciones_semana_siguiente',
        python_callable=generate_predictions_callable,
        op_kwargs={'execution_date': '{{ ds }}'}
    )


inicio  >> crear_carpetas >> construir_data_set >> separar_data >> entrenamiento >> prediccion
    
