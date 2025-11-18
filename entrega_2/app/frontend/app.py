# app/frontend/app.py
import gradio as gr
import requests
import json
import os

# --- Configuración del Backend ---
# Usamos el nombre del servicio Docker para comunicarnos dentro del docker-compose network
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

def get_base_data_structure():
    """Retorna la estructura base de datos de entrada."""
    # Nota: Los valores 'compró', 'items', 'año', 'semana' son placeholders para el backend/pipeline
    return {
        "customer_id": "61353",
        "product_id": "61364",
        "week": "2025-01", 
        "region_id": "80",
        "zone_id": "5148",
        "customer_type": "ABARROTES",
        "Y": -46.558718,
        "X": -107.860564,
        "num_deliver_per_week": 3,
        "num_visit_per_week": 1,
        "brand": "Brand 35",
        "category": "BEBIDAS CARBONATADAS",
        "sub_category": "GASEOSAS",
        "segment": "MEDIUM",
        "package": "BOTELLA",
        "size": 1.0,
        "compró": 0,
        "items": 0.0,
        "año": 2025,
        "semana": 1
    }

def generate_prediction(customer_id, product_id, week, region_id, zone_id, customer_type, Y, X, num_deliver_per_week, num_visit_per_week, brand, category, sub_category, segment, package, size):
    """Llama al endpoint /predict del backend."""
    
    # Mapear los inputs de Gradio a la estructura de Pydantic del Backend
    data = {
        "customer_id": str(customer_id),
        "product_id": str(product_id),
        "week": week,
        "region_id": str(region_id),
        "zone_id": str(zone_id),
        "customer_type": customer_type,
        "Y": float(Y),
        "X": float(X),
        "num_deliver_per_week": int(num_deliver_per_week),
        "num_visit_per_week": int(num_visit_per_week),
        "brand": brand,
        "category": category,
        "sub_category": sub_category,
        "segment": segment,
        "package": package,
        "size": float(size),
        # Placeholders (el backend los ignora o sobreescribe, pero son necesarios para Pydantic)
        "compró": 0,
        "items": 0.0,
        "año": 2025,
        "semana": 1
    }

    try:
        response = requests.post(f"{BACKEND_URL}/predict", json=data)
        
        if response.status_code == 200:
            result = response.json()
            prob = result['prob_compra']
            decision = result['decision']
            return (
                f"**RESULTADO DE PREDICCIÓN**\n"
                f"--- \n"
                f"Probabilidad de Compra (Clase 1): **{prob:.4f}**\n"
                f"Decisión (Umbral 0.5): **{decision}**"
            )
        else:
            return f"Error en el backend (Status: {response.status_code}): {response.text}"
            
    except requests.exceptions.RequestException as e:
        return f"Error de conexión con el backend ({BACKEND_URL}): {e}"


# --- Interfaz Gradio ---
default_data = get_base_data_structure()

with gr.Blocks(title="SodAI Drinks Predictor") as demo:
    gr.Markdown("# 🥤 SodAI Drinks Predictor (MLOps MDS7202)")
    gr.Markdown("Esta interfaz consume el modelo LightGBM desplegado via FastAPI/MLflow para predecir la probabilidad de que un cliente compre un producto en una semana dada.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Datos de la Solicitud")
            customer_id = gr.Textbox(label="Customer ID", value=default_data['customer_id'])
            product_id = gr.Textbox(label="Product ID", value=default_data['product_id'])
            week = gr.Textbox(label="Semana de Predicción (YYYY-WW)", value=default_data['week'], placeholder="Ej: 2025-01")
            
            gr.Markdown("### Features de Cliente")
            region_id = gr.Textbox(label="Region ID", value=default_data['region_id'])
            zone_id = gr.Textbox(label="Zone ID", value=default_data['zone_id'])
            customer_type = gr.Textbox(label="Customer Type", value=default_data['customer_type'])
            Y = gr.Number(label="Coordenada Y (Latitud)", value=default_data['Y'])
            X = gr.Number(label="Coordenada X (Longitud)", value=default_data['X'])
            num_deliver_per_week = gr.Number(label="Entregas/Semana", value=default_data['num_deliver_per_week'])
            num_visit_per_week = gr.Number(label="Visitas/Semana", value=default_data['num_visit_per_week'])
            
            gr.Markdown("### Features de Producto")
            brand = gr.Textbox(label="Brand", value=default_data['brand'])
            category = gr.Textbox(label="Category", value=default_data['category'])
            sub_category = gr.Textbox(label="Sub Category", value=default_data['sub_category'])
            segment = gr.Textbox(label="Segment", value=default_data['segment'])
            package = gr.Textbox(label="Package", value=default_data['package'])
            size = gr.Number(label="Size (Litros)", value=default_data['size'])
            
            predict_btn = gr.Button("Generar Predicción")
            
        with gr.Column():
            gr.Markdown("## Output")
            output_text = gr.Markdown("Presiona 'Generar Predicción' para ver el resultado.")
            
    inputs = [customer_id, product_id, week, region_id, zone_id, customer_type, Y, X, num_deliver_per_week, num_visit_per_week, brand, category, sub_category, segment, package, size]
    
    predict_btn.click(
        fn=generate_prediction, 
        inputs=inputs, 
        outputs=output_text
    )

demo.launch(server_name="0.0.0.0", server_port=8501)