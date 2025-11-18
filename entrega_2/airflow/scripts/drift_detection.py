# scripts/drift_detection.py  <-- (ESTA ES LA RUTA CORRECTA)
"""
Script para la tarea de detección de drift.
Implementa una lógica simple basada en el cambio de la media de una feature clave.
"""
import pandas as pd
import os
import joblib

# --- Rutas y Constantes ---
BASE_DIR = "/opt/airflow"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DRIFT_THRESHOLD = 0.1 # Umbral de cambio del 10% en la media de una feature

def check_for_drift(feature_name: str = 'num_deliver_per_week'):
    """
    Compara la media de una feature clave entre el set de entrenamiento y el set de predicción.
    Retorna el task_id de la siguiente tarea: 'reentrenar_modelo' si hay drift, 
    o 'no_reentrenar_skip_training' si no hay drift.
    """
    print(f"Iniciando Tarea Condicional: Detección de Drift en '{feature_name}'")

    # 1. Cargar Data
    try:
        df_train_eng = pd.read_parquet(os.path.join(PROCESSED_DIR, "df_train.parquet"))
        df_predict_eng = pd.read_parquet(os.path.join(PROCESSED_DIR, "df_predict.parquet"))
    except FileNotFoundError as e:
        print(f"[ERROR] Archivo no encontrado: {e}")
        return 'no_reentrenar_skip_training' # Fallar o no reentrenar si no hay data
    
    # 2. Aplicar Pipeline de Ingeniería al set de Entrenamiento (referencia)
    try:
        eng_pipeline = joblib.load(os.path.join(MODELS_DIR, 'engineering_pipeline_V2.joblib'))
    except FileNotFoundError:
        print("[ERROR] Pipeline de ingeniería no encontrado. No se puede calcular drift.")
        return 'no_reentrenar_skip_training'

    # Aplicar SÓLO la transformación de ingeniería (no el preprocesamiento final)
    df_train_transformed = eng_pipeline.transform(df_train_eng)
    df_predict_transformed = eng_pipeline.transform(df_predict_eng)

    # 3. Extraer la feature clave
    if feature_name not in df_train_transformed.columns or feature_name not in df_predict_transformed.columns:
        print(f"[WARN] Feature '{feature_name}' no encontrada en el set transformado.")
        # Usamos una feature simple como fallback si la clave falla
        feature_name = 'num_deliver_per_week'
        
    try:
        mean_train = df_train_transformed[feature_name].mean()
        mean_predict = df_predict_transformed[feature_name].mean()
    except KeyError:
        print(f"[ERROR] Fallo al extraer la media de '{feature_name}'.")
        return 'no_reentrenar_skip_training'
        
    # 4. Calcular Drift
    if mean_train == 0:
        drift = 1.0 # Si el promedio base es 0 y el nuevo no, es drift.
    else:
        drift = abs(mean_predict - mean_train) / mean_train

    print(f"Media de '{feature_name}' (Train Ref): {mean_train:.4f}")
    print(f"Media de '{feature_name}' (Predict Set): {mean_predict:.4f}")
    print(f"Drift porcentual: {drift:.4f}")

    # 5. Decidir reentrenamiento
    if drift > DRIFT_THRESHOLD:
        print(f"[DRIFT DETECTED] Drift ({drift:.4f}) > Threshold ({DRIFT_THRESHOLD}). REENTRENAMIENTO REQUERIDO.")
        return 'reentrenar_modelo' # ID de la tarea a ejecutar
    else:
        print(f"[NO DRIFT] Drift ({drift:.4f}) <= Threshold ({DRIFT_THRESHOLD}). NO SE REQUIERE REENTRENAMIENTO.")
        return 'no_reentrenar_skip_training' # ID de la tarea a ejecutar

if __name__ == "__main__":
    # Simulación de prueba local
    # check_for_drift()
    pass