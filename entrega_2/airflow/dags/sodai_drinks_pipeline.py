# airflow/dags/sodai_drinks_pipeline.py
"""
DAG de Airflow para el pipeline de MLOps de SodAI Drinks.
Incluye Preparación, Preprocesamiento, Entrenamiento Condicional y Predicción.
"""

import sys
import os
import pendulum 
import pandas as pd
import joblib
import numpy as np
import warnings

from airflow.decorators import dag, task
from airflow.operators.python import BranchPythonOperator
from airflow.models.baseoperator import chain

# --- Rutas y Constantes ---
BASE_DIR = "/opt/airflow" 
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models") 

sys.path.append(SCRIPTS_DIR)

# Imports de scripts auxiliares
try:
    from prepare_data import build_datasets_callable
    from preprocess import get_engineering_pipeline, get_preprocessor
    from train_model import train_model_callable
    from generate_predictions import generate_predictions_callable
    from drift_detection import check_for_drift
    IMPORT_SUCCESS = True
    print("✅ Todos los scripts importados correctamente")
except ImportError as e:
    print(f"[ERROR] Faltan scripts esenciales: {e}")
    IMPORT_SUCCESS = False
    raise e  # Esto hará que el DAG falle claramente si faltan imports

N_SAMPLE_TRAIN = 2_000_000

# DAG
@dag(
    dag_id='sodai_drinks_mlops_pipeline',
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="@monthly", 
    catchup=False,
    tags=['mlops', 'sodai_drinks'],
    default_args={
        'owner': 'airflow',
        'retries': 1,
    }
)
def sodai_drinks_mlops_pipeline():

    # --- TAREA 1: PREPARACIÓN DE DATOS ---
    @task(task_id='preparar_datos')
    def task_prepare_data():
        if not IMPORT_SUCCESS:
            raise Exception("Scripts no disponibles - verificar importaciones")
            
        print("Iniciando Tarea 1: Preparar datos (Crear andamiajes train/predict)")
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        build_datasets_callable(
            clientes_path=os.path.join(DATA_DIR, "clientes.parquet"),
            productos_path=os.path.join(DATA_DIR, "productos.parquet"),
            transacciones_path=os.path.join(DATA_DIR, "transacciones.parquet"), 
            train_output_path=os.path.join(PROCESSED_DIR, "df_train.parquet"),
            predict_output_path=os.path.join(PROCESSED_DIR, "df_predict.parquet")
        )

    # --- TAREA 2: PREPROCESAMIENTO ---
    @task(task_id='preprocesar_datos')
    def task_preprocess_data():
        warnings.filterwarnings("ignore")
        
        print(f"Iniciando Tarea 2: Preprocesamiento (muestreo {N_SAMPLE_TRAIN} filas)")
        try:
            # df_train.parquet tiene todas las features, incl. 'compró' e 'items'
            dataset = pd.read_parquet(os.path.join(PROCESSED_DIR, "df_train.parquet"))
        except FileNotFoundError:
            raise Exception(f"df_train.parquet no encontrado. ¿Falló la Tarea 1?")

        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Muestreo si el dataset es muy grande
        if len(dataset) > N_SAMPLE_TRAIN:
            print(f"Dataset original {len(dataset)} filas. Muestreando a {N_SAMPLE_TRAIN}...")
            dataset = dataset.sample(n=N_SAMPLE_TRAIN, random_state=42)
        
        train_df = dataset.copy()
        y_train = train_df.get('compró').astype(int)
        
        if y_train is None:
            raise ValueError("Columna 'compró' (target) no encontrada en df_train.parquet")

        # 1. Pipeline de Ingeniería de Features (fit_transform)
        eng_pipeline = get_engineering_pipeline()
        X_train_eng = eng_pipeline.fit_transform(train_df, y_train)

        # 2. Obtener listas de features finales
        numeric_features = eng_pipeline.named_steps['detectar_tipos_finales'].get_numeric_features()
        categorical_features = eng_pipeline.named_steps['detectar_tipos_finales'].get_categorical_features()
        print(f"Features numéricas detectadas: {numeric_features}")
        print(f"Features categóricas detectadas: {categorical_features}")

        # 3. Pipeline de Preprocesamiento (fit)
        preprocessor = get_preprocessor(numeric_features, categorical_features)
        preprocessor.fit(X_train_eng)

        # 4. Guardar pipelines (clave para reentrenamiento y predicción)
        joblib.dump(eng_pipeline, os.path.join(MODELS_DIR, 'engineering_pipeline_V2.joblib'))
        joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor_V2.joblib'))
        print("Pipelines de Ingeniería y Preprocesamiento guardados.")

        # 5. Transformar y guardar X_train (para la tarea de entrenamiento)
        feature_names_out = preprocessor.get_feature_names_out()
        X_train_final_np = preprocessor.transform(X_train_eng)
        
        # Convertir a DataFrame (OHE lo hace sparse si no se especifica sparse_output=False)
        if hasattr(X_train_final_np, 'toarray'):
            X_train_final_np = X_train_final_np.toarray()

        X_train_final = pd.DataFrame(X_train_final_np, columns=feature_names_out, index=X_train_eng.index)
        
        # Optimizar dtypes
        for col in X_train_final.columns:
            if col in numeric_features:
                X_train_final[col] = X_train_final[col].astype('float32')
            else:
                # Las OHE son float64 (0.0 o 1.0), podemos pasarlas a int8
                X_train_final[col] = X_train_final[col].astype('int8')
        
        X_train_final.to_parquet(os.path.join(PROCESSED_DIR, 'X_train_V2.parquet'), index=False, compression='gzip')
        print(f"X_train_V2.parquet creado con {len(X_train_final.columns)} columnas")

    # --- TAREA 3 (CONDICIONAL): DETECCIÓN DE DRIFT ---
    drift_check = BranchPythonOperator(
        task_id='detectar_drift_o_reentrenar',
        python_callable=check_for_drift,
        op_kwargs={'feature_name': 'num_deliver_per_week'}, 
    )
    
    # --- TAREA 4A: REENTRENAMIENTO (si hay drift o periodicidad) ---
    @task(task_id='reentrenar_modelo')
    def task_reentrenar_modelo():
        print("INICIANDO REENTRENAMIENTO FORZADO por Drift o periodicidad.")
        # Usamos 10 trials para la demo, pero puedes subirlo
        train_model_callable(trial_count=10, use_gpu=False) 

    # --- TAREA 4B: SALTAR ENTRENAMIENTO ---
    @task(task_id='no_reentrenar_skip_training')
    def task_skip_training():
        print("No se requiere reentrenamiento. Saltando la tarea de entrenamiento.")
        pass

    # --- TAREA 5: GENERACIÓN DE PREDICCIONES ---
    @task(task_id='generar_predicciones', trigger_rule='none_failed_min_one_success')
    def task_generate_predictions():
        print("Iniciando Tarea 5: Generación de predicciones")
        generate_predictions_callable()

    # --- FLUJO DEL DAG ---
    t1 = task_prepare_data()
    t2 = task_preprocess_data()
    t4a = task_reentrenar_modelo()
    t4b = task_skip_training()
    t5 = task_generate_predictions()

    # 1. Preparar Datos -> 2. Preprocesar Datos
    t1 >> t2 

    # 2. Preprocesar Datos -> 3. Decisión de Drift
    t2 >> drift_check 

    # 3. Decisión de Drift: 
    #    Si True (Drift) -> 4A (Reentrenar) -> 5 (Predecir)
    #    Si False (No Drift) -> 4B (Skipear) -> 5 (Predecir)
    chain(drift_check, t4a, t5)
    chain(drift_check, t4b, t5)

# Llamada final para que Airflow registre el DAG
sodai_drinks_mlops_pipeline()
