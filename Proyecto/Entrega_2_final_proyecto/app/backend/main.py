# app/backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import joblib 
from datetime import datetime
from pathlib import Path
import pickle
from fastapi.responses import JSONResponse
from prepare_row import build_single_prediction_row

# --- Configuración del Modelo MLflow ---
# MODEL_NAME = "sodai_drinks_lgbm_model"
# MODEL_STAGE = "Staging"  # Cargar el modelo en etapa de Staging/Production
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Modelos" / "best_lgb_pipeline.pkl"
FE_Path = BASE_DIR / "Modelos" / "pipeline_f_e.pkl"


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
# --- Inicialización ---

with open(FE_Path, "rb") as f:
    fe_pipeline = pickle.load(f)

app = FastAPI(
    title="SodAI Drinks Prediction API",
    description="API para predecir la probabilidad de compra de productos por cliente.",
    version="1.0.0"
)



# Definición del esquema de datos para FastAPI (debe coincidir con las features de entrada)
class PredictionInput(BaseModel):
    customer_id: int
    product_id: int
    purchase_date : datetime


    



@app.post("/predict")
async def predict(input_data: PredictionInput):
        """Genera la probabilidad de compra (clase 1) para una combinación cliente-producto-semana."""
        X = pd.DataFrame([{
        "customer_id": input_data.customer_id,
        "product_id": input_data.product_id,
        "purchase_date": pd.to_datetime(input_data.purchase_date)  # asegura datetime64[ns]
        }])
        X = X = build_single_prediction_row(
        X["customer_id"].iloc[0],
        X["product_id"].iloc[0],
        X["purchase_date"].iloc[0]
         )

        X = fe_pipeline.transform(X)

        # Hacemos la predicción
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1] 
        return JSONResponse(content={
        "Compro": int(pred),
        "Probabilidad": float(prob)
        })


if __name__ == "__main__":
    # Comando para correr localmente: uvicorn main:app --reload --port 8000
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)