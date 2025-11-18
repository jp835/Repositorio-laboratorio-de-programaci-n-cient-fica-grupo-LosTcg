# airflow/dags/sodai_drinks_pipeline_safe.py
"""
DAG de Airflow para el pipeline de MLOps de SodAI Drinks.
Etapas de Preparación y Preprocesamiento (Tareas 1 y 2).
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib 
import pendulum 

from airflow.decorators import dag, task
from sklearn.compose import ColumnTransformer 

# --- Rutas y Constantes ---
BASE_DIR = "/opt/airflow" 
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models") 

sys.path.append(SCRIPTS_DIR)

# Imports seguros
try:
    from prepare_data import build_datasets_callable
    from preprocess import get_engineering_pipeline, get_preprocessor
except ImportError as e:
    print(f"[WARN] Scripts personalizados no encontrados: {e}")
    build_datasets_callable = lambda **kwargs: print("[WARN] build_datasets_callable no disponible")
    get_engineering_pipeline = lambda: print("[WARN] get_engineering_pipeline no disponible")
    get_preprocessor = lambda n, c: print("[WARN] get_preprocessor no disponible")

N_SAMPLE_TRAIN = 2_000_000

# Función auxiliar segura
def _transform_and_convert_dtypes(preprocessor: ColumnTransformer, df_eng: pd.DataFrame, feature_names: list, num_feat: list, cat_feat: list):
    try:
        data_np = preprocessor.transform(df_eng)
        df_processed = pd.DataFrame(data_np, columns=feature_names, index=df_eng.index)
        for col in num_feat:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype('float32')
        for col in cat_feat:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(str).astype('category')
        return df_processed
    except Exception as e:
        print(f"[ERROR] Fallo en transformación de datos: {e}")
        return pd.DataFrame()

# DAG
@dag(
    dag_id='sodai_drinks_prep_and_preprocess_v2',
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=['prep', 'preprocess', 'mds7202'],
)
def sodai_drinks_prep_dag():

    @task(task_id='preparar_datos')
    def task_prepare_data():
        print("Iniciando Tarea 1: Preparar datos")
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        try:
            build_datasets_callable(
                clientes_path=os.path.join(DATA_DIR, "clientes.parquet"),
                productos_path=os.path.join(DATA_DIR, "productos.parquet"),
                transacciones_path=os.path.join(DATA_DIR, "transacciones.parquet"), 
                train_output_path=os.path.join(PROCESSED_DIR, "df_train.parquet"),
                predict_output_path=os.path.join(PROCESSED_DIR, "df_predict.parquet")
            )
        except Exception as e:
            print(f"[ERROR] Error ejecutando build_datasets_callable: {e}")

    @task(task_id='preprocesar_datos')
    def task_preprocess_data():
        print(f"Iniciando Tarea 2: Preprocesamiento (muestreo {N_SAMPLE_TRAIN} filas)")
        try:
            dataset = pd.read_parquet(os.path.join(PROCESSED_DIR, "df_train.parquet"))
        except FileNotFoundError:
            print(f"[WARN] df_train.parquet no encontrado en {PROCESSED_DIR}")
            return

        os.makedirs(MODELS_DIR, exist_ok=True)
        if len(dataset) > N_SAMPLE_TRAIN:
            dataset = dataset.sample(n=N_SAMPLE_TRAIN, random_state=42)
        train_df = dataset.copy()
        y_train = train_df.get('compró')
        if y_train is None:
            print("[WARN] Columna 'compró' no encontrada")
            return

        # Ingeniería de Features
        eng_pipeline = get_engineering_pipeline()
        if callable(getattr(eng_pipeline, "fit_transform", None)):
            X_train_eng = eng_pipeline.fit_transform(train_df, y_train)
        else:
            X_train_eng = train_df.copy()

        # Features
        numeric_features = getattr(eng_pipeline.named_steps['detectar_tipos_finales'], 'get_numeric_features', lambda: [])()
        categorical_features = getattr(eng_pipeline.named_steps['detectar_tipos_finales'], 'get_categorical_features', lambda: [])()

        preprocessor = get_preprocessor(numeric_features, categorical_features)
        if callable(getattr(preprocessor, "fit", None)):
            preprocessor.fit(X_train_eng, y_train)

        # Guardar pipelines
        try:
            joblib.dump(eng_pipeline, os.path.join(MODELS_DIR, 'engineering_pipeline_V2.joblib'))
            joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor_V2.joblib'))
        except Exception as e:
            print(f"[ERROR] Error guardando pipelines: {e}")

        # Transformar y guardar
        try:
            feature_names_out = getattr(preprocessor, "get_feature_names_out", lambda: X_train_eng.columns)()
            X_train_final = _transform_and_convert_dtypes(preprocessor, X_train_eng, feature_names_out, numeric_features, categorical_features)
            X_train_final.to_parquet(os.path.join(PROCESSED_DIR, 'X_train_V2.parquet'), index=False, compression='gzip')
            print(f"X_train_V2.parquet creado con {len(X_train_final.columns)} columnas")
        except Exception as e:
            print(f"[ERROR] Error transformando y guardando datos: {e}")

    t1 = task_prepare_data()
    t2 = task_preprocess_data()
    t1 >> t2

sodai_drinks_prep_dag()
