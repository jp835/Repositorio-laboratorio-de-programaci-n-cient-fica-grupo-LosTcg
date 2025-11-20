# app/frontend/app.py
import gradio as gr
import requests
import json
import os

# --- Configuración del Backend ---
# Usamos el nombre del servicio Docker para comunicarnos dentro del docker-compose network
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")


from datetime import datetime
import pandas as pd

# URL de la API FastAPI
API_URL = "http://backend:8000/predict"  # "backend" es el nombre del servicio en docker-compose

def predict_from_api(file=None, customer_id=None, product_id=None, purchase_date=None):
    try:
        if file:
            # Leer JSON y mandar cada fila como request
            input_data = pd.read_json(file)
            results = []
            for _, row in input_data.iterrows():
                payload = {
                    "customer_id": row["customer_id"],
                    "product_id": row["product_id"],
                    "purchase_date": row["purchase_date"]
                }
                res = requests.post(API_URL, json=payload).json()
                results.append(res)
            return results
        else:
            # Entrada individual desde campos
            payload = {
                "customer_id": customer_id,
                "product_id": product_id,
                "purchase_date": purchase_date.isoformat() if isinstance(purchase_date, datetime) else str(purchase_date)
            }
            res = requests.post(API_URL, json=payload).json()
            return res
    except Exception as e:
        return {"Error": str(e)}



def gradio_interface():


    interface = gr.Interface(
        fn=lambda file: predict_from_api(file),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Prediccion de compra para cliente y producto en una determinada semana",
        description='Sube un archivo JSON con el id del cliente, el id del producto, y la semana en forma "aa - mm - dd". Luego, se entregar si compro o no, con la probabilidad de haber comprado'
    )
    interface.launch(share=True)

if __name__ == "__main__":
    gradio_interface()
