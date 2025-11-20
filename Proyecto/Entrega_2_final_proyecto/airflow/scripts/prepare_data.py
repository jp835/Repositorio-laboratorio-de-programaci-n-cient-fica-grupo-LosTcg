# airflow/scripts/prepare_data.py
"""
Script para la Tarea 1 de Airflow.
Carga los datos crudos (clientes, productos, transacciones) y crea
dos datasets "andamio" (scaffolds) para entrenamiento y predicción.
"""

import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

# --- Función Principal ---
def build_datasets_callable(execution_date=None):
    """
    Función principal de la Tarea 1. Carga, procesa y crea los datasets
    de entrenamiento y predicción.
    """

    base_path = os.path.join(os.getcwd(), execution_date)
    current_dir = os.path.dirname(os.path.abspath(__file__))  
    # Esto da: /ruta/a/airflow/scripts

    
    project_root = os.path.dirname(current_dir)  
    # Esto da: /ruta/a/airflow

    # Ruta a la carpeta data
    data_dir = os.path.join(project_root, "Data")  
    clientes_path = os.path.join(data_dir, "clientes.parquet")
    productos_path = os.path.join(data_dir, "productos.parquet")
    transacciones_path = os.path.join(data_dir, "transacciones.parquet")
    preparado_path = os.path.join(base_path, "preparada")
    
    print(f"Iniciando Tarea 1: build_datasets_callable")
    print(f"Cargando datos crudos desde {clientes_path}, {productos_path}, {transacciones_path}")
    
    try:
        df_clientes = pd.read_parquet(clientes_path)
        df_productos = pd.read_parquet(productos_path)
        df_transacciones = pd.read_parquet(transacciones_path)
    except Exception as e:
        print(f"Error al leer archivos parquet crudos: {e}")
        raise

    
    print("Limpiando y optimizando tipos de datos...")
    df_clientes['customer_id'] = df_clientes['customer_id'].astype('category')
    df_productos['product_id'] = df_productos['product_id'].astype('category')
    df_transacciones['customer_id'] = df_transacciones['customer_id'].astype('category')
    df_transacciones['product_id'] = df_transacciones['product_id'].astype('category')
    df_transacciones["purchase_date"] = pd.to_datetime(df_transacciones["purchase_date"])
    df_transacciones["week"] = df_transacciones["purchase_date"].dt.strftime('%Y-%W').astype('category')
    
    
    unique_customers = df_transacciones["customer_id"].unique()
    unique_products = df_transacciones["product_id"].unique()
    unique_weeks = df_transacciones['week'].unique()

    # --- 2. Crear Dataset a trabajar (df_final.parquet) ---
    print("Creando dataset final(Producto Cartesiano)...")
    index = pd.MultiIndex.from_product(
        [unique_customers, unique_products, unique_weeks], 
        names=['customer_id', 'product_id', 'week']
    )
    df_final = pd.DataFrame(index=index).reset_index()

    print("Agregando datos de compras...")
    compras = (
        df_transacciones
        .groupby(['customer_id', 'product_id', 'week'], observed=True)
        .agg(compró=('order_id', 'count'), items=('items', 'sum'))
        .reset_index()
    )
    compras['compró'] = (compras['compró'] > 0).astype(int)
    df_final = df_final.merge(compras, on=['customer_id', 'product_id', 'week'], how='left')
    df_final['compró'] = df_final['compró'].fillna(0).astype('int8') # Target
    df_final['items'] = df_final['items'].fillna(0).astype('float32')
    
    
    df_final[['año', 'semana']] = df_final['week'].astype(str).str.split('-', expand=True)
    df_final['año'] = df_final['año'].astype('int16')
    df_final['semana'] = df_final['semana'].astype('int16')
    
    print("Realizando merge con datos de clientes y productos...")
    df_final = df_final.merge(df_clientes, on="customer_id", how="left")
    df_final = df_final.merge(df_productos, on="product_id", how="left")
    
    
    os.makedirs(preparado_path, exist_ok=True)
    df_final.to_parquet(os.path.join(preparado_path, "df_final.parquet"), index=False, compression='gzip')
    print(f"Archivo de entrenamiento guardado en {os.path.join(preparado_path, 'df_final.parquet')}")
    print(f"Tamaño del dataset de entrenamiento: {len(df_final)} filas")
    
    
    print("Creando data set de predicción 'df_predict.parquet'...")
    latest_date = df_transacciones["purchase_date"].max()
    prediction_date = latest_date + pd.DateOffset(weeks=1)
    prediction_week_str = prediction_date.strftime('%Y-%W')
    prediction_year = int(prediction_date.strftime('%Y'))
    prediction_week_num = int(prediction_date.strftime('%W'))

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

    df_predict.to_parquet(os.path.join(preparado_path, "df_predict.parquet"), index=False, compression='gzip')
    print(f"Archivo de predicción guardado en {os.path.join(preparado_path, 'df_predict.parquet')}")
    print(f"Tamaño del dataset de predicción: {len(df_predict)} filas")
    print(f"Semana de predicción: {prediction_week_str}")


if __name__ == "__main__":
    # Para pruebas locales (opcional)
    print("Ejecutando script prepare_data.py localmente...")
    
    # Rutas relativas para prueba local
    local_data_dir = "data"
    local_processed_dir = "data/processed"
    
    build_datasets_callable(
        clientes_path=os.path.join(local_data_dir, "clientes.parquet"),
        productos_path=os.path.join(local_data_dir, "productos.parquet"),
        transacciones_path=os.path.join(local_data_dir, "transacciones.parquet"),
        train_output_path=os.path.join(local_processed_dir, "df_train.parquet"),
        predict_output_path=os.path.join(local_processed_dir, "df_predict.parquet")
    )