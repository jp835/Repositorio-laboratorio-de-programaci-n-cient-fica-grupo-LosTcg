# app/backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import joblib 

# --- Configuración del Modelo MLflow ---
MODEL_NAME = "sodai_drinks_lgbm_model"
MODEL_STAGE = "Staging"  # Cargar el modelo en etapa de Staging/Production
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# --- Inicialización ---
app = FastAPI(
    title="SodAI Drinks Prediction API",
    description="API para predecir la probabilidad de compra de productos por cliente.",
    version="1.0.0"
)

# Variable global para almacenar el pipeline (se cargará al inicio)
full_pipeline = None

# Definición del esquema de datos para FastAPI (debe coincidir con las features de entrada)
class PredictionInput(BaseModel):
    customer_id: str
    product_id: str
    week: str # e.g., '2025-01'
    # Features de Clientes (Se asume que estas son necesarias antes de FE)
    region_id: str
    zone_id: str
    customer_type: str
    Y: float
    X: float
    num_deliver_per_week: int
    num_visit_per_week: int
    # Features de Productos
    brand: str
    category: str
    sub_category: str
    segment: str
    package: str
    size: float
    
    # Placeholders para FE/Target, que el pipeline de predicción completará/eliminará
    compró: int = 0
    items: float = 0.0
    año: int = 2025
    semana: int = 1


@app.on_event("startup")
async def load_model():
    """Carga el modelo/pipeline de MLflow al iniciar la aplicación."""
    global full_pipeline
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    
    try:
        # Intenta cargar el pipeline completo registrado en MLflow
        full_pipeline = mlflow.sklearn.load_model(model_uri=model_uri)
        print(f"Modelo cargado exitosamente desde MLflow: {model_uri}")
    except Exception as e:
        print(f"ERROR: Fallo al cargar el modelo de MLflow ({model_uri}). {e}")
        # En un entorno de producción real, esto debería ser un error fatal.
        # Aquí, se podría intentar cargar un modelo de fallback local si MLflow no está disponible.
        raise RuntimeError(f"Fallo al cargar el modelo/pipeline desde MLflow: {e}")

# --- Endpoint de Salud ---
@app.get("/health")
def health_check():
    """Verifica el estado del servicio y si el modelo está cargado."""
    return {"status": "ok", "model_loaded": full_pipeline is not None}

# --- Endpoint de Predicción ---
@app.post("/predict")
def predict(input_data: PredictionInput):
    """Genera la probabilidad de compra (clase 1) para una combinación cliente-producto-semana."""
    
    if full_pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Intente más tarde.")

    # 1. Convertir Pydantic a DataFrame (Necesario para el pipeline de sklearn)
    input_dict = input_data.dict()
    
    # Asegurar que 'año' y 'semana' se calculen de 'week'
    try:
        year, week_num = input_dict['week'].split('-')
        input_dict['año'] = int(year)
        input_dict['semana'] = int(week_num)
    except:
        pass # Se asumen los defaults si falla el split
    
    # Crear un DataFrame con una sola fila
    df_input = pd.DataFrame([input_dict])

    # 2. Generar Predicción
    try:
        # predict_proba retorna un array de [prob_clase_0, prob_clase_1]
        prediction_proba = full_pipeline.predict_proba(df_input)[:, 1][0]
        
        # Opcional: SHAP para interpretabilidad de la predicción
        # Esto sería muy lento si no se usa un explainer precalculado o un microservicio. 
        # Se omite por rendimiento en el demo.
        
    except Exception as e:
        print(f"ERROR durante la predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno durante la predicción: {e}")

    # 3. Devolver Resultado
    return {
        "customer_id": input_data.customer_id,
        "product_id": input_data.product_id,
        "prob_compra": float(prediction_proba),
        "decision": "COMPRA ALTA" if prediction_proba > 0.5 else "COMPRA BAJA"
    }

if __name__ == "__main__":
    # Comando para correr localmente: uvicorn main:app --reload --port 8000
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)