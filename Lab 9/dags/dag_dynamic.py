from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from hiring_dynamic_functions import create_folders,load_and_merge,split_data,train_model,evaluate_models

with DAG(
    dag_id='dynamic_hiring',
    description='Pipeline dinamico de contratación con descarga de datos',
    start_date=datetime(2024, 10, 1),
    schedule_interval='0 15 5 * *',          
    catchup=True
) as dag:
    
    inicio = EmptyOperator(task_id='iniciar_pipeline')

    crear_carpetas = PythonOperator(
        task_id='crear_carpetas',
        python_callable=create_folders,
        provide_context=True
    )

    def branching(**kwargs):
        execution_date = kwargs['ds']
        if datetime.strptime(execution_date, "%Y-%m-%d") < datetime(2024, 11, 1):
            return 'descargar_data_1'
        else:
            return 'descargar_data_1_y_2'
        
    
    branch_task = BranchPythonOperator(
        task_id='branch_task',
        python_callable=branching,
        provide_context=True,
        dag=dag
    )

    descargar_data_1 = BashOperator(
        task_id='descargar_data_1',
        bash_command=(
            "curl -o /opt/airflow/{{ ds }}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        )
    )

    descargar_data_1_y_2 = BashOperator(
        task_id='descargar_data_1_y_2',
        bash_command=(
            "curl -o /opt/airflow/{{ ds }}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv && "
            "curl -o /opt/airflow/{{ ds }}/raw/data_2.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"
        )
    )

    load_and_merge_task = PythonOperator(
        task_id='cargar_y_concatenar',
        python_callable=load_and_merge,
        op_kwargs={'execution_date': '{{ ds }}'},
        trigger_rule='one_success'
    )

    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        op_kwargs={'execution_date': '{{ ds }}'}
    )

    train_rf = PythonOperator(
        task_id='train_random_forest',
        python_callable=train_model,
        op_kwargs={
            'modelo': RandomForestClassifier(n_estimators=100, random_state=42),
            'execution_date': '{{ ds }}'
        }
    )

    train_lr = PythonOperator(
        task_id='train_logistic_regression',
        python_callable=train_model,
        op_kwargs={
            'modelo': LogisticRegression(max_iter=1000),
            'execution_date': '{{ ds }}'
        }
    )

    train_dt = PythonOperator(
        task_id='train_decision_tree',
        python_callable=train_model,
        op_kwargs={
            'modelo': DecisionTreeClassifier(random_state=42),
            'execution_date': '{{ ds }}'
        }
    )

    evaluar_modelos = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_models,
        op_kwargs={'execution_date': '{{ ds }}'},
        trigger_rule='all_success'  # Se ejecuta solo si los 3 modelos fueron entrenados
    )

    inicio >> crear_carpetas >> branch_task
    branch_task >> [descargar_data_1, descargar_data_1_y_2] >> load_and_merge_task
    load_and_merge_task >> split_data_task
    split_data_task >> [train_rf, train_lr, train_dt] >> evaluar_modelos


