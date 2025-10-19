from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from pathlib import Path
from fastapi.responses import JSONResponse

# ==============================
# Paths
# ==============================
BASE_DIR = Path("C:/Users/admin/OneDrive/Documents/Repositorio-laboratorio-de-programaci-n-cient-fica-grupo-LosTcg-1/Lab8")
MODEL_PATH = BASE_DIR / "models" / "best_xgb_model.pkl"

# ==============================
# Load model
# ==============================
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ==============================
# Define FastAPI app
# ==============================
app = FastAPI(title="Potabilidad API", version="1.0")

# ==============================
# Request model
# ==============================
class WaterSample(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# ==============================
# Routes
# ==============================
@app.get("/")
async def home():
    """
    Home route describing the API
    """
    return {
        "description": "API para predecir potabilidad del agua usando XGBoost optimizado.",
        "input": {
            "ph": "float",
            "Hardness": "float",
            "Solids": "float",
            "Chloramines": "float",
            "Sulfate": "float",
            "Conductivity": "float",
            "Organic_carbon": "float",
            "Trihalomethanes": "float",
            "Turbidity": "float"
        },
        "output": {
            "potabilidad": "0 = no potable, 1 = potable"
        }
    }


@app.post("/potabilidad/")
async def predict(sample: WaterSample):
    """
    Predict if water is potable
    """
    # Convert input to numpy array for the model
    X = np.array([[sample.ph, sample.Hardness, sample.Solids,
                   sample.Chloramines, sample.Sulfate, sample.Conductivity,
                   sample.Organic_carbon, sample.Trihalomethanes, sample.Turbidity]])
    
    pred = model.predict(X)[0]
    return JSONResponse(content={"potabilidad": int(pred)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
