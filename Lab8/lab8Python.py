import os
import mlflow
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# MLflow path
base_dir = Path(__file__).resolve().parent
mlruns_path = base_dir / "mlruns"
mlruns_path.mkdir(exist_ok=True)

# Use pathlib.as_uri() to make it valid (works cross-platform)
mlflow.set_tracking_uri(mlruns_path.as_uri())
mlflow.set_experiment("Water")

# Enable autolog
mlflow.autolog()

# Example model
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)

with mlflow.start_run(run_name="RandomForest_test"):
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)



