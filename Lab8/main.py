from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from pathlib import Path
from fastapi.responses import JSONResponse


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_xgb_model.pkl"


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


app = FastAPI(title="Potabilidad API", version="1.0")


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


@app.get("/")
async def home():
    """
    Home route describing the API
    """
    return {
        "description": "API para predecir potabilidad del agua usando XGBoost optimizado.",
        "problem": "Se busca determinar si agua con ciertas caracteristicas es potable o no",
        "input": {
            "ph": "float - Ph del agua",
            "Hardness": "float - Dureza del agua",
            "Solids": "float - Total de solidos disueltos",
            "Chloramines": "float - Nivel de cloro",
            "Sulfate": "float - Cantidad de sulfatos por litro",
            "Conductivity": "float - Conductividad del agua",
            "Organic_carbon": "float - Cantidad total de carbon organico",
            "Trihalomethanes": "float - Concentracion de THMs",
            "Turbidity": "float - Turbiedad del agua"
        },
        "output": {
            "potabilidad": "0 = no potable, 1 = potable"
        }
    }


@app.post("/potabilidad/")
async def predict(sample: WaterSample):
    """
    Predecir si el agua es potable o no
    """

    X = np.array([[sample.ph, sample.Hardness, sample.Solids,
                   sample.Chloramines, sample.Sulfate, sample.Conductivity,
                   sample.Organic_carbon, sample.Trihalomethanes, sample.Turbidity]])
    
    pred = model.predict(X)[0]
    return JSONResponse(content={"potabilidad": int(pred)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
