"""
Script para la tarea de entrenamiento.
Versión corregida - alineación de datos X e y.
"""

import pandas as pd
import joblib
import os
import sys
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
import mlflow
import mlflow.lightgbm
import numpy as np

BASE_DIR = "/opt/airflow"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "sodai_drinks_prod"

def train_model_corregido():
    print("Iniciando entrenamiento del modelo (LightGBM - Corregido)")
    
    try:
        # Cargar datos
        X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train_V2.parquet"))
        df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "df_train.parquet"))
        
        print(f"X_train shape: {X_train.shape}")
        print(f"df_train shape: {df_train.shape}")
        
        # VERIFICAR Y CORREGIR ALINEACIÓN
        # Asegurar que X_train y df_train tengan los mismos índices
        if not X_train.index.equals(df_train.index):
            print("⚠️  Los índices no coinciden. Realizando alineación...")
            # Tomar solo las filas que existen en ambos datasets
            common_indices = X_train.index.intersection(df_train.index)
            X_train = X_train.loc[common_indices]
            df_train = df_train.loc[common_indices]
            print(f"✅ Datos alineados. Nuevo shape: {X_train.shape}")
        
        # TRANSFORMAR TARGET A BINARIO
        y_train_original = df_train["compró"]
        y_train = (y_train_original > 0).astype(int)
        
        print(f"✅ Datos finales - X: {X_train.shape}, y: {y_train.shape}")
        print(f"✅ Target binario - Valores únicos: {y_train.unique()}")
        print(f"✅ Distribución binaria:")
        print(y_train.value_counts())
        
        # Calcular scale_pos_weight
        counts = y_train.value_counts()
        neg = counts.get(0, 1)
        pos = counts.get(1, 1)
        scale = neg / pos if pos > 0 else 1.0
        print(f"✅ Relación Neg/Pos: neg={neg}, pos={pos}, scale={scale:.2f}")
        
        # Configurar MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        with mlflow.start_run() as run:
            params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "max_depth": 6,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "scale_pos_weight": scale,
                "random_state": 42,
                "verbose": -1,
                "objective": "binary"
            }
            
            print("🚀 Entrenando modelo...")
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
            # Métricas
            y_pred = model.predict(X_train)
            f1 = f1_score(y_train, y_pred)
            accuracy = (y_pred == y_train).mean()
            
            # Log en MLflow
            mlflow.log_params(params)
            mlflow.log_metric("train_f1_score", f1)
            mlflow.log_metric("train_accuracy", accuracy)
            mlflow.log_metric("scale_pos_weight", scale)
            mlflow.log_metric("train_size", len(X_train))
            mlflow.log_metric("positive_class_ratio", y_train.mean())
            
            mlflow.lightgbm.log_model(model, "model")
            
            print("🎉 Modelo entrenado exitosamente")
            print(f"📊 F1 Score: {f1:.4f}")
            print(f"📊 Accuracy: {accuracy:.4f}")
            print(f"🔗 Run ID: {run.info.run_id}")
            
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_model_corregido()
