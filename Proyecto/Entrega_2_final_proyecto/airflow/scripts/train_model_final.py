import pandas as pd
import joblib
import os
import sys
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report
import mlflow
import mlflow.lightgbm
import numpy as np
from preprocess import get_engineering_pipeline
from preprocess import get_preprocessor
from sklearn.pipeline import Pipeline
import optuna
import pickle



def train_model_completo(execution_date = None):
    print('ENTRENAMIENTO Modelo - LightGBM')
    base_path = os.path.join(os.getcwd(), execution_date)
    preparada_path = os.path.join(base_path, "preparada")
    MLFLOW_TRACKING_URI = os.path.join(base_path, "mlflow")
    modelos_path = os.path.join(os.getcwd(), "Modelos")
    os.makedirs(modelos_path, exist_ok=True)

    # Cargar datos
    splits_path = os.path.join(base_path, "splits")
    df_train = pd.read_parquet(os.path.join(splits_path, "train.parquet"))
    df_test  = pd.read_parquet(os.path.join(splits_path, "test.parquet"))

    df_train = df_train.drop("week", axis = 1)
    df_test = df_test.drop("week", axis = 1)

    
    feature_pipeline = get_engineering_pipeline()
    feature_pipeline.fit(df_train)
    X_train = feature_pipeline.transform(df_train)
    y_train = df_train["compró"]

    X_test = feature_pipeline.transform(df_test)
    y_test = df_test["compró"]
    print("Columnas de X_train:", X_train.columns.tolist())
    print("Columnas de X_test:", X_test.columns.tolist())
    print("Cantidad de columnas X_train:", len(X_train.columns))
    print("Cantidad de columnas X_test:", len(X_test.columns))
    print("Valores únicos en y_train:", np.unique(y_train))
    print("Cantidad por clase:", pd.Series(y_train).value_counts())

    numeric_features = feature_pipeline.named_steps["detectar_tipos_finales"].get_numeric_features()
    categorical_features = feature_pipeline.named_steps["detectar_tipos_finales"].get_categorical_features()

    

    feature_engineer_file = os.path.join(modelos_path, "pipeline_f_e.pkl")
    with open(feature_engineer_file, "wb") as f:
        pickle.dump(feature_pipeline, f)




    
    # Calcular scale
    counts = y_train.value_counts()
    scale = counts.get(0, 1) / counts.get(1, 1) if counts.get(1, 1) > 0 else 1.0
    print('Scale pos weight:', round(scale, 2))
    
    def objective(trial):
        
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 60),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 250),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
            "random_state": 42,
            "device": "cpu",  
            "objective": "binary",
            "boosting_type": "gbdt",
            "scale_pos_weight": scale,
            "verbose": -1
        }

        
        run_name = f"LGBMClassifier_lr_{params['learning_rate']:.3f}"
        with mlflow.start_run(run_name=run_name, nested=True):
            
            preprocessor = get_preprocessor(numeric_features,categorical_features)
            preprocessor.set_output(transform="pandas")
            
            model = LGBMClassifier(**params)
            pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("classifier", model)
            ])
            pipeline.fit(X_train, y_train)
            y_val_pred = pipeline.predict(X_test)
            f1 = f1_score(y_test, y_val_pred)
            mlflow.log_params(params)
            mlflow.log_metric("valid_f1", f1)
        return f1
    
    final_preprocessor = get_preprocessor(numeric_features, categorical_features)
    final_preprocessor.set_output(transform="pandas")
            

    # Configurar MLflow
    EXPERIMENT_NAME = 'sodai_drinks_prod'
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    study = optuna.create_study(direction="maximize", study_name="lgb_sodai")
    study.optimize(objective, n_trials=5)

    best_model = LGBMClassifier(**study.best_params)
    
    
    

    best_pipeline = Pipeline([
        ("preprocessing", final_preprocessor),
        ("classifier", best_model)
    ])
    best_pipeline.fit(X_train, y_train)
    pipeline_file = os.path.join(modelos_path, "best_lgb_pipeline.pkl")
    with open(pipeline_file, "wb") as f:
        pickle.dump(best_pipeline, f)

    with mlflow.start_run(run_name="best_model_final"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1", study.best_value)
        mlflow.log_artifact(pipeline_file, "Modelos")
        mlflow.log_artifact(feature_engineer_file,"Modelos")

    print("Optimization complete")
    print("Best F1:", study.best_value)
    print("Best params:", study.best_params)
    return None

    
    
        

if __name__ == '__main__':
    train_model_completo()
