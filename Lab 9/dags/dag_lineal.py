from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface


with DAG(
    dag_id='hiring_lineal',
    description='Pipeline lineal de contratación con descarga de datos y Gradio',
    start_date=datetime(2024, 10, 1),
    schedule_interval=None,          
    catchup=False
) as dag:

    inicio = EmptyOperator(task_id='iniciar_pipeline')

    crear_carpetas = PythonOperator(
        task_id='crear_carpetas',
        python_callable=create_folders,
        provide_context=True
    )

    
    descargar_data = BashOperator(
        task_id='descargar_data',
        bash_command=(
            "curl -o /opt/airflow/{{ ds }}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        )
    )

    
    separar_data = PythonOperator(
        task_id='separar_data',
        python_callable=split_data,
        op_kwargs={'execution_date': '{{ ds }}'}
    )

    
    entrenamiento = PythonOperator(
        task_id='preprocess_and_train',
        python_callable=preprocess_and_train,
        op_kwargs={'execution_date': '{{ ds }}'}
    )

    
    gradio = PythonOperator(
        task_id='gradio_interface',
        python_callable=gradio_interface,
        op_kwargs={'execution_date': '{{ ds }}'}
    )

    
    inicio >> crear_carpetas >> descargar_data >> separar_data >> entrenamiento >> gradio
