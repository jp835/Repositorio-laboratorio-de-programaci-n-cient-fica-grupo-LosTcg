# airflow/scripts/generate_predictions.py
"""
Script para la tarea de generación de predicciones.
Carga el modelo/pipeline completo desde MLflow y lo aplica al set de predicción.
"""
import pandas as pd
import os
import mlflow
import joblib
import numpy as np  # ✅ Import necesario

# --- Rutas y Constantes ---
BASE_DIR = "/opt/airflow"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "sodai_drinks_lgbm_model"
MODEL_STAGE = "Staging"

def generate_predictions_callable():
    """
    Carga el modelo final desde MLflow y genera predicciones para la próxima semana.
    """
    print(f"Iniciando generación de predicciones")

    # 1. Cargar Data de Predicción
    predict_data_path = os.path.join(PROCESSED_DIR, "df_predict.parquet")
    try:
        df_predict_eng = pd.read_parquet(predict_data_path)
        print(f"✅ Datos de predicción cargados: {len(df_predict_eng)} filas")
    except FileNotFoundError:
        print(f"[ERROR] Archivo de predicción no encontrado en {predict_data_path}")
        return

    # 2. Cargar Modelo desde MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    print(f"Cargando modelo desde: {model_uri}")
    
    try:
        full_pipeline = mlflow.sklearn.load_model(model_uri=model_uri)
        print("✅ Modelo cargado exitosamente desde MLflow")
    except Exception as e:
        print(f"[ERROR] Fallo al cargar el modelo de MLflow: {e}")
        return

    # 3. Generar Predicciones
    try:
        predictions_proba = full_pipeline.predict_proba(df_predict_eng)[:, 1]
        print(f"✅ Predicciones generadas: {len(predictions_proba)} probabilidades")
    except Exception as e:
        print(f"[ERROR] Fallo al generar predicciones: {e}")
        return

    # 4. Guardar Resultados
    df_results = df_predict_eng[['customer_id', 'product_id', 'week']].copy()
    df_results['prob_compra'] = predictions_proba
    df_results.sort_values(by=['customer_id', 'prob_compra'], ascending=[True, False], inplace=True)
    
    predictions_output_path = os.path.join(PROCESSED_DIR, "predictions_V2.parquet")
    df_results.to_parquet(predictions_output_path, index=False, compression='gzip')
    
    print(f"✅ Predicciones guardadas en: {predictions_output_path}")
    print("Generación de predicciones completada exitosamente.")

if __name__ == "__main__":
    # Para pruebas
    generate_predictions_callable()