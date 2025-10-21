import os
from pathlib import Path
import pickle

import pandas as pd
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import optuna.visualization as vis
import pkg_resources


BASE_DIR = Path(__file__).resolve().parent
MLRUNS_PATH = BASE_DIR / "mlruns"
PLOTS_PATH = BASE_DIR / "plots"
MODELS_PATH = BASE_DIR / "models"

#Si no existen las carpetas, las crea
for p in [MLRUNS_PATH, PLOTS_PATH, MODELS_PATH]:
    p.mkdir(exist_ok=True)

mlflow.set_tracking_uri(MLRUNS_PATH.as_uri())
EXPERIMENT_NAME = "Water_Optuna_XGBoost"
mlflow.set_experiment(EXPERIMENT_NAME)


df = pd.read_csv(BASE_DIR / "water_potability.csv")
X = df.drop(columns=["Potability"]).fillna(df.median())
y = df["Potability"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "random_state": 42,
        "n_jobs": -1
    }

    run_name = f"XGBoost_lr_{params['learning_rate']:.3f}"
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(params)
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        f1 = f1_score(y_valid, y_pred)
        mlflow.log_metric("valid_f1", f1)

    return f1


def optimize_model(n_trials=20):
    study = optuna.create_study(direction="maximize", study_name="xgb_water")
    study.optimize(objective, n_trials=n_trials)


    fig1 = vis.plot_optimization_history(study)
    fig2 = vis.plot_param_importances(study)

    fig1.write_image(PLOTS_PATH / "optimization_history.png")
    fig2.write_image(PLOTS_PATH / "param_importance.png")

  
    best_model = xgb.XGBClassifier(**study.best_params)
    best_model.fit(X_train, y_train)

    model_file = MODELS_PATH / "best_xgb_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(best_model, f)

  
    plt.figure(figsize=(10,6))
    plt.bar(range(len(best_model.feature_importances_)), best_model.feature_importances_)
    plt.title("Feature Importance")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.savefig(PLOTS_PATH / "feature_importance.png", bbox_inches="tight")
    plt.close()


    versions_path = BASE_DIR / "library_versions.txt"
    with open(versions_path, "w") as f:
        for pkg in pkg_resources.working_set:
            f.write(f"{pkg.key}=={pkg.version}\n")

    
    with mlflow.start_run(run_name="Best_Model_Summary"):
        mlflow.log_artifact(PLOTS_PATH / "optimization_history.png", artifact_path="plots")
        mlflow.log_artifact(PLOTS_PATH / "param_importance.png", artifact_path="plots")
        mlflow.log_artifact(PLOTS_PATH / "feature_importance.png", artifact_path="plots")
        mlflow.log_artifact(model_file, artifact_path="models")
        mlflow.log_artifact(versions_path, artifact_path="versions")
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_valid_f1", study.best_value)

    print("✅ Optimization complete")
    print("Best F1:", study.best_value)
    print("Best params:", study.best_params)
    return best_model


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_run_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
    return best_model


if __name__ == "__main__":
    optimize_model()

