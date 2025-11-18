import pandas as pd
import joblib
import os
import sys
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report
import mlflow
import mlflow.lightgbm
import numpy as np

BASE_DIR = '/opt/airflow'
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
EXPERIMENT_NAME = 'sodai_drinks_prod'

def train_model_completo():
    print('🎯 ENTRENAMIENTO COMPLETO - LightGBM')
    
    try:
        # Cargar datos
        X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, 'X_train_V2.parquet'))
        df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, 'df_train.parquet'))
        
        print('📊 Datos cargados - X:', X_train.shape, 'df:', df_train.shape)
        
        # Alineación
        if not X_train.index.equals(df_train.index):
            common_indices = X_train.index.intersection(df_train.index)
            X_train = X_train.loc[common_indices]
            df_train = df_train.loc[common_indices]
            print('✅ Datos alineados:', X_train.shape)
        
        # Target binario
        y_train = (df_train['compró'] > 0).astype(int)
        
        print('🎯 Target - Clases:', y_train.unique())
        print('📈 Distribución:')
        print(y_train.value_counts())
        
        # Calcular scale
        counts = y_train.value_counts()
        scale = counts.get(0, 1) / counts.get(1, 1) if counts.get(1, 1) > 0 else 1.0
        print('⚖️  Scale pos weight:', round(scale, 2))
        
        # Configurar MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        with mlflow.start_run() as run:
            # Parámetros optimizados
            params = {
                'n_estimators': 150,
                'learning_rate': 0.1,
                'num_leaves': 63,
                'max_depth': 7,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'scale_pos_weight': scale,
                'random_state': 42,
                'verbose': -1,
                'objective': 'binary'
            }
            
            print('🚀 Entrenando modelo LightGBM...')
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
            # Predicciones y métricas
            y_pred = model.predict(X_train)
            y_pred_proba = model.predict_proba(X_train)[:, 1]
            
            f1 = f1_score(y_train, y_pred)
            accuracy = (y_pred == y_train).mean()
            
            print('')
            print('📊 RESULTADOS DEL ENTRENAMIENTO:')
            print('   F1 Score:', round(f1, 4))
            print('   Accuracy:', round(accuracy, 4))
            print('   Positive Ratio:', round(y_train.mean(), 4))
            
            # Reporte de clasificación
            print('')
            print('📋 Classification Report:')
            print(classification_report(y_train, y_pred))
            
            # Log en MLflow
            mlflow.log_params(params)
            mlflow.log_metric('train_f1_score', f1)
            mlflow.log_metric('train_accuracy', accuracy)
            mlflow.log_metric('scale_pos_weight', scale)
            mlflow.log_metric('train_size', len(X_train))
            mlflow.log_metric('positive_class_ratio', y_train.mean())
            
            print('✅ Métricas registradas en MLflow')
            
            # GUARDAR MODELO LOCALMENTE
            model_path = os.path.join(MODELS_DIR, 'lightgbm_model_final.joblib')
            joblib.dump(model, model_path)
            print('💾 Modelo guardado en:', model_path)
            
            # También guardar pipeline completo
            pipeline_path = os.path.join(MODELS_DIR, 'full_pipeline.joblib')
            try:
                # Cargar pipelines de preprocesamiento
                eng_pipeline = joblib.load(os.path.join(MODELS_DIR, 'engineering_pipeline_V2.joblib'))
                preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor_V2.joblib'))
                
                # Crear pipeline completo
                from sklearn.pipeline import Pipeline
                full_pipeline = Pipeline([
                    ('engineering', eng_pipeline),
                    ('preprocessing', preprocessor),
                    ('classifier', model)
                ])
                
                joblib.dump(full_pipeline, pipeline_path)
                print('💾 Pipeline completo guardado en:', pipeline_path)
            except Exception as e:
                print('⚠️  No se pudo guardar pipeline completo:', e)
            
            print('🔗 Run ID:', run.info.run_id)
            print('🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE')
            
    except Exception as e:
        print('❌ Error:', e)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    train_model_completo()
