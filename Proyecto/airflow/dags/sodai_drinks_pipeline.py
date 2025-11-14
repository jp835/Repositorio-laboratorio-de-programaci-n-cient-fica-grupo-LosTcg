# En airflow/dags/sodai_drinks_pipeline.py

from airflow.decorators import dag, task
import pendulum
import sys
import os
import pandas as pd
import joblib 
import gc # <-- Importamos el Garbage Collector

# --- 1. Definición de Rutas ---
BASE_DIR = "/opt/airflow" 
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models") 

sys.path.append(SCRIPTS_DIR)

# --- 2. Importar las funciones de nuestros scripts ---
try:
    from prepare_data import build_datasets_callable
    from preprocess import get_engineering_pipeline, get_preprocessor
except ImportError as e:
    print(f"Error importando scripts: {e}")


# --- 3. Argumentos por Defecto del DAG ---
default_args = {
    'owner': 'Arturo_y_Juan',
    'start_date': pendulum.today('UTC'),
    'retries': 0, 
}

# --- 4. Definición del DAG ---
@dag(
    dag_id='sodai_drinks_pipeline',
    default_args=default_args,
    description='Pipeline de MLOps para SodAI Drinks (Entrega 2)',
    schedule=None,
    catchup=False,
    tags=['mlops', 'sodai', 'mds7202'],
)
def sodai_drinks_dag():
    """
    ### Pipeline de MLOps de SodAI Drinks
    Orquesta el pipeline de ML:
    1.  **Preparación de Datos**: Crea dataset de entrenamiento y andamiaje de predicción.
    2.  **Preprocesamiento**: Aplica ingeniería de features y preprocesa los datos.
    """
    
    # --- TAREA 1 ---
    @task(task_id='preparar_datos_y_andamiaje')
    def task_prepare_data():
        """
        Ejecuta el script prepare_data.py para construir los datasets.
        """
        print("Iniciando Tarea 1: build_datasets_callable")
        
        build_datasets_callable(
            clientes_path=os.path.join(DATA_DIR, "clientes.parquet"),
            productos_path=os.path.join(DATA_DIR, "productos.parquet"),
            transacciones_path=os.path.join(DATA_DIR, "transacciones.parquet"),
            train_output_path=os.path.join(PROCESSED_DIR, "df_train.parquet"),
            predict_output_path=os.path.join(PROCESSED_DIR, "df_predict.parquet")
        )
        print("Tarea 1 finalizada exitosamente.")

    # --- TAREA 2 (Versión Optimizada para Memoria) ---
    @task(task_id='preprocesar_datos')
    def task_preprocess_data():
        """
        Tarea 2: Carga los datos, los divide, y aplica los pipelines
        DE FORMA SECUENCIAL para ahorrar RAM.
        """
        print("Iniciando Tarea 2: Preprocesamiento de datos (Optimizado)")
        
        # --- 1. Cargar datos de la Tarea 1 ---
        TRAIN_PATH = os.path.join(PROCESSED_DIR, "df_train.parquet")
        PREDICT_PATH = os.path.join(PROCESSED_DIR, "df_predict.parquet")
        
        print("Cargando dataset de entrenamiento completo (9M filas)...")
        dataset = pd.read_parquet(TRAIN_PATH)
        
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

        # --- 2. Definir el Split Temporal (Holdout) de E1 ---
        print("Definiendo split temporal 70/15/15...")
        dataset = dataset.sort_values('week')
        weeks = dataset['week'].unique()
        n_weeks = len(weeks)

        train_weeks = weeks[:int(0.7 * n_weeks)]
        val_weeks = weeks[int(0.7 * n_weeks):int(0.85 * n_weeks)]
        test_weeks = weeks[int(0.85 * n_weeks):]

        # --- 3. Instanciar Pipelines ---
        # (No los entrenamos todavía)
        eng_pipeline = get_engineering_pipeline()
        
        # Necesitamos saber las columnas *después* de la ing. de features
        # para inicializar el preprocesador
        print("Pre-fiteando eng_pipeline en una muestra para obtener features...")
        sample_df = dataset.sample(n=50000, random_state=42)
        X_sample_eng = eng_pipeline.fit_transform(sample_df, sample_df['compró'])
        numeric_features = eng_pipeline.named_steps['detect_types'].get_numeric_features()
        categorical_features = eng_pipeline.named_steps['detect_types'].get_categorical_features()
        
        print(f"Features numéricas detectadas: {numeric_features}")
        print(f"Features categóricas detectadas: {categorical_features}")
        
        preprocessor = get_preprocessor(numeric_features, categorical_features)
        
        # --- 4. Procesar y Guardar DATOS DE ENTRENAMIENTO ---
        print("Procesando datos de ENTRENAMIENTO (Train)...")
        train_df = dataset[dataset['week'].isin(train_weeks)].copy()
        y_train = train_df['compró']
        
        X_train_eng = eng_pipeline.fit_transform(train_df, y_train)
        X_train_final = preprocessor.fit_transform(X_train_eng)
        
        feature_names = preprocessor.get_feature_names_out()
        pd.DataFrame(X_train_final, columns=feature_names).to_parquet(os.path.join(PROCESSED_DIR, 'X_train_final.parquet'), index=False)
        y_train.to_frame().to_parquet(os.path.join(PROCESSED_DIR, 'y_train.parquet'), index=False)
        
        print("Guardando pipelines entrenados...")
        joblib.dump(eng_pipeline, os.path.join(MODELS_DIR, 'engineering_pipeline.joblib'))
        joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor.joblib'))
        
        # Liberar memoria
        del train_df, y_train, X_train_eng, X_train_final
        gc.collect()

        # --- 5. Procesar y Guardar DATOS DE VALIDACIÓN ---
        print("Procesando datos de VALIDACIÓN (Validation)...")
        val_df = dataset[dataset['week'].isin(val_weeks)].copy()
        y_val = val_df['compró']
        
        X_val_eng = eng_pipeline.transform(val_df)
        X_val_final = preprocessor.transform(X_val_eng)
        
        pd.DataFrame(X_val_final, columns=feature_names).to_parquet(os.path.join(PROCESSED_DIR, 'X_val_final.parquet'), index=False)
        y_val.to_frame().to_parquet(os.path.join(PROCESSED_DIR, 'y_val.parquet'), index=False)

        del val_df, y_val, X_val_eng, X_val_final
        gc.collect()

        # --- 6. Procesar y Guardar DATOS DE PRUEBA ---
        print("Procesando datos de PRUEBA (Test)...")
        test_df = dataset[dataset['week'].isin(test_weeks)].copy()
        y_test = test_df['compró']
        
        X_test_eng = eng_pipeline.transform(test_df)
        X_test_final = preprocessor.transform(X_test_eng)
        
        pd.DataFrame(X_test_final, columns=feature_names).to_parquet(os.path.join(PROCESSED_DIR, 'X_test_final.parquet'), index=False)
        y_test.to_frame().to_parquet(os.path.join(PROCESSED_DIR, 'y_test.parquet'), index=False)

        del test_df, y_test, X_test_eng, X_test_final, dataset
        gc.collect()

        # --- 7. Procesar y Guardar DATOS DE PREDICCIÓN ---
        print("Procesando datos de PREDICCIÓN (Future)...")
        df_predict = pd.read_parquet(PREDICT_PATH)
        
        X_predict_eng = eng_pipeline.transform(df_predict)
        X_predict_final = preprocessor.transform(X_predict_eng)
        
        pd.DataFrame(X_predict_final, columns=feature_names).to_parquet(os.path.join(PROCESSED_DIR, 'X_predict_final.parquet'), index=False)
        
        print("Tarea 2 finalizada exitosamente.")


    # --- 6. Definir la secuencia del DAG ---
    prepare_task = task_prepare_data()
    preprocess_task = task_preprocess_data()
    
    prepare_task >> preprocess_task

# --- 7. Instanciar el DAG ---
sodai_drinks_dag()