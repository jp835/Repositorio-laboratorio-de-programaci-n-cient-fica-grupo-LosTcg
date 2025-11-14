# En airflow/scripts/prepare_data.py
import pandas as pd
import os
import gc # Garbage Collector

# --- Rutas ---
BASE_DIR = "/opt/airflow" 
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# --- Función Principal ---
def build_datasets_callable(
    clientes_path, productos_path, transacciones_path,
    train_output_path, predict_output_path
):
    
    print(f"Cargando datos desde: {os.path.dirname(clientes_path)}...")
    df_clientes = pd.read_parquet(clientes_path)
    df_productos = pd.read_parquet(productos_path)
    df_transacciones = pd.read_parquet(transacciones_path)

    # --- 1. Optimización de tipos (Como en E1) ---
    print("Optimizando tipos de datos...")
    
    # IDs como categorías (eficiente en memoria)
    df_clientes['customer_id'] = df_clientes['customer_id'].astype('category')
    df_productos['product_id'] = df_productos['product_id'].astype('category')
    df_transacciones['customer_id'] = df_transacciones['customer_id'].astype('category')
    df_transacciones['product_id'] = df_transacciones['product_id'].astype('category')
    
    for col in df_clientes.select_dtypes(include=['object', 'string']).columns:
        df_clientes[col] = df_clientes[col].astype('category')
    for col in df_productos.select_dtypes(include=['object', 'string']).columns:
        df_productos[col] = df_productos[col].astype('category')

    # Transacciones
    df_transacciones["purchase_date"] = pd.to_datetime(df_transacciones["purchase_date"])
    df_transacciones["week"] = df_transacciones["purchase_date"].dt.strftime('%Y-%U').astype('category')
    
    # --- 2. LÓGICA CLAVE (Corregida a tu E1) ---
    # Usamos solo clientes y productos ACTIVOS (de transacciones) para el cruce
    
    unique_customers = df_transacciones["customer_id"].unique()
    unique_products = df_transacciones["product_id"].unique()
    unique_weeks = df_transacciones['week'].unique()

    print(f"Dimensiones CORREGIDAS: {len(unique_customers)} clientes x {len(unique_products)} productos x {len(unique_weeks)} semanas")

    # --- 3. Crear Dataset de Entrenamiento ---
    print("Creando andamiaje de entrenamiento (MultiIndex)...")
    
    # Producto cartesiano eficiente
    index = pd.MultiIndex.from_product(
        [unique_customers, unique_products, unique_weeks], 
        names=['customer_id', 'product_id', 'week']
    )
    df_train = pd.DataFrame(index=index).reset_index()
    gc.collect()

    print("Agregando transacciones reales...")
    compras = (
        df_transacciones
        .groupby(['customer_id', 'product_id', 'week'], observed=True)
        .agg(compró=('order_id', 'count'), items=('items', 'sum'))
        .reset_index()
    )
    
    df_train = df_train.merge(compras, on=['customer_id', 'product_id', 'week'], how='left')
    del compras
    gc.collect()
    
    df_train['compró'] = df_train['compró'].fillna(0).astype('int8')
    df_train['items'] = df_train['items'].fillna(0).astype('float32')
    
    df_train[['año', 'semana']] = df_train['week'].astype(str).str.split('-', expand=True)
    df_train['año'] = df_train['año'].astype('int16')
    df_train['semana'] = df_train['semana'].astype('int8')
    
    # Merge con maestros
    df_train = df_train.merge(df_clientes, on="customer_id", how="left")
    df_train = df_train.merge(df_productos, on="product_id", how="left")
    
    print(f"Guardando train: {train_output_path}")
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    df_train.to_parquet(train_output_path, index=False)
    
    del df_train
    gc.collect()

    # --- 4. Crear Andamiaje Predicción (t+2) ---
    print("Creando andamiaje de predicción...")
    
    latest_date = df_transacciones["purchase_date"].max()
    prediction_date = latest_date + pd.DateOffset(weeks=1)
    prediction_week_str = prediction_date.strftime('%Y-%U')
    prediction_year = int(prediction_date.strftime('%Y'))
    prediction_week_num = int(prediction_date.strftime('%U'))

    print(f"Última semana en datos: {latest_date.strftime('%Y-%U')}. Prediciendo para semana: {prediction_week_str}")

    # Andamiaje solo clientes ACTIVOS x productos ACTIVOS
    index_pred = pd.MultiIndex.from_product(
        [unique_customers, unique_products], 
        names=['customer_id', 'product_id']
    )
    df_predict = pd.DataFrame(index=index_pred).reset_index()
    
    df_predict['week'] = prediction_week_str
    df_predict['año'] = prediction_year
    df_predict['semana'] = prediction_week_num
    df_predict['compró'] = 0 
    df_predict['items'] = 0.0
    
    # Merge maestros
    df_predict = df_predict.merge(df_clientes, on="customer_id", how="left")
    df_predict = df_predict.merge(df_productos, on="product_id", how="left")

    print(f"Guardando predict: {predict_output_path}")
    df_predict.to_parquet(predict_output_path, index=False)
    print("Tarea 1 finalizada exitosamente.")
    
if __name__ == "__main__":
    print("Ejecutando script de preparación de datos...")
    # Crear la carpeta de salida si no existe (para pruebas locales)
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
        
    build_datasets(
        clientes_path="data/clientes.parquet",
        productos_path="data/productos.parquet",
        transacciones_path="data/transacciones.parquet",
        train_output_path="data/processed/df_train.parquet",
        predict_output_path="data/processed/df_predict.parquet"
    )