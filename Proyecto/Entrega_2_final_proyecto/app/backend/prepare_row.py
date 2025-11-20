
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
clientes_path = BASE_DIR / "clientes.parquet"
products_path = BASE_DIR / "productos.parquet"


df_clientes = pd.read_parquet(clientes_path)
df_productos = pd.read_parquet(products_path)






def build_single_prediction_row(customer_id, product_id, purchase_date):
    # 1. Construir fila base
    week_str = purchase_date.strftime("%Y-%W")
    year = purchase_date.year
    week_num = int(purchase_date.strftime("%W"))
    
    df = pd.DataFrame([{
        "customer_id": customer_id,
        "product_id": product_id,
        "week": week_str,
        "año": year,
        "semana": week_num,
        "compró": 0,     # placeholder requerido
        "items": 0       # placeholder requerido
    }])

    # 2. Agregar features de cliente y producto
    df = df.merge(df_clientes, on="customer_id", how="left")
    df = df.merge(df_productos, on="product_id", how="left")
    df = df.drop("week", axis = 1)

    return df