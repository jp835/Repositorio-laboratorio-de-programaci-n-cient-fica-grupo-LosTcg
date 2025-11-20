# airflow/scripts/generate_predictions.py
"""
Script para la tarea de generación de predicciones.
Carga el modelo/pipeline completo desde MLflow y lo aplica al set de predicción.
"""
import pandas as pd
import os
import mlflow
import joblib
import numpy as np  
import pickle



def generate_predictions_callable(execution_date = None):
    
    """
    Carga el modelo final desde MLflow y genera predicciones para la próxima semana.
    """
    print(f"Iniciando generación de predicciones")
    base_path = os.path.join(os.getcwd(), execution_date)
    preparada_path = os.path.join(base_path, "preparada")
    modelos_path = os.path.join(os.getcwd(), "Modelos")
    predicciones_path = os.path.join(os.getcwd(), "Predicciones")

    
    predict_data_path = os.path.join(preparada_path, "df_predict.parquet")

    
    MODEL_PATH = os.path.join(modelos_path, "best_lgb_pipeline.pkl")
    FE_Path = os.path.join(modelos_path, "pipeline_f_e.pkl")   


    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(FE_Path, "rb") as f:
        fe_pipeline = pickle.load(f)
    
    
    df_predict_base = pd.read_parquet(predict_data_path)
    df_predict_base = df_predict_base.drop("week", axis = 1)
    df_predict_eng = fe_pipeline.transform(df_predict_base)

    predicciones = model.predict(df_predict_eng)
    


    df_results = df_predict_base[['customer_id', 'product_id']].copy()
    df_results['prediccion_compra'] = predicciones
    df_results.sort_values(by=['customer_id', 'prediccion_compra'], ascending=[True, False], inplace=True)
    
    predictions_output_path = os.path.join(predicciones_path, "predictions_V2.parquet")
    df_results.to_parquet(predictions_output_path, index=False, compression='gzip')
    
    print(f"Predicciones guardadas en: {predictions_output_path}")
    print("Generación de predicciones completada exitosamente.")
    

if __name__ == "__main__":
    # Para pruebas
    generate_predictions_callable()