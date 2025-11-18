# En airflow/scripts/prepare_data.py
"""
Script para la Tarea 1 de Airflow.
Carga los datos crudos (clientes, productos, transacciones) y crea
dos datasets "andamio" (scaffolds) para entrenamiento y predicción.
"""

import pandas as pd
import os

# --- Función Principal ---
def build_datasets_callable(
    clientes_path: str, 
    productos_path: str, 
    transacciones_path: str,
    train_output_path: str, 
    predict_output_path: str
):
    """
    Función principal de la Tarea 1. Carga, procesa y crea los datasets
    de entrenamiento y predicción.
    """
    
    print(f"Iniciando Tarea 1: build_datasets_callable")
    df_clientes = pd.read_parquet(clientes_path)
    df_productos = pd.read_parquet(productos_path)
    df_transacciones = pd.read_parquet(transacciones_path)

    # 1. Limpieza de datos y tipos optimizados
    df_clientes['customer_id'] = df_clientes['customer_id'].astype('category')
    df_productos['product_id'] = df_productos['product_id'].astype('category')
    df_transacciones['customer_id'] = df_transacciones['customer_id'].astype('category')
    df_transacciones['product_id'] = df_transacciones['product_id'].astype('category')
    df_transacciones["purchase_date"] = pd.to_datetime(df_transacciones["purchase_date"])
    df_transacciones["week"] = df_transacciones["purchase_date"].dt.strftime('%Y-%U').astype('category')
    
    # Entidades Activas
    unique_customers = df_transacciones["customer_id"].unique()
    unique_products = df_transacciones["product_id"].unique()
    unique_weeks = df_transacciones['week'].unique()

    # --- 2. Crear Dataset de Entrenamiento (df_train.parquet) ---
    print("Creando andamiaje de entrenamiento (Producto Cartesiano)...")
    index = pd.MultiIndex.from_product(
        [unique_customers, unique_products, unique_weeks], 
        names=['customer_id', 'product_id', 'week']
    )
    df_train = pd.DataFrame(index=index).reset_index()

    compras = (
        df_transacciones
        .groupby(['customer_id', 'product_id', 'week'], observed=True)
        .agg(compró=('order_id', 'count'), items=('items', 'sum'))
        .reset_index()
    )
    
    df_train = df_train.merge(compras, on=['customer_id', 'product_id', 'week'], how='left')
    df_train['compró'] = df_train['compró'].fillna(0).astype('int8') # Target
    df_train['items'] = df_train['items'].fillna(0).astype('float32')
    
    df_train[['año', 'semana']] = df_train['week'].astype(str).str.split('-', expand=True)
    df_train['año'] = df_train['año'].astype('int16')
    df_train['semana'] = df_train['semana'].astype('int8')
    
    df_train = df_train.merge(df_clientes, on="customer_id", how="left")
    df_train = df_train.merge(df_productos, on="product_id", how="left")
    
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    df_train.to_parquet(train_output_path, index=False, compression='gzip')
    
    # --- 3. Crear Andamiaje de Predicción (df_predict.parquet) ---
    print("Creando andamiaje de predicción 'df_predict.parquet'...")
    latest_date = df_transacciones["purchase_date"].max()
    prediction_date = latest_date + pd.DateOffset(weeks=1)
    prediction_week_str = prediction_date.strftime('%Y-%U')
    prediction_year = int(prediction_date.strftime('%Y'))
    prediction_week_num = int(prediction_date.strftime('%U'))

    index_pred = pd.MultiIndex.from_product(
        [unique_customers, unique_products], 
        names=['customer_id', 'product_id']
    )
    df_predict = pd.DataFrame(index=index_pred).reset_index()
    
    df_predict['week'] = prediction_week_str
    df_predict['año'] = prediction_year
    df_predict['semana'] = prediction_week_num
    df_predict['compró'] = 0 # Placeholder
    df_predict['items'] = 0.0 # Placeholder
    
    df_predict = df_predict.merge(df_clientes, on="customer_id", how="left")
    df_predict = df_predict.merge(df_productos, on="product_id", how="left")

    df_predict.to_parquet(predict_output_path, index=False, compression='gzip')
    print("Tarea 1 (prepare_data) finalizada exitosamente.")