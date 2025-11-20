import os
import pandas as pd

def split_and_save_train_test(execution_date = None, train_ratio=0.7):
    """
    Carga df_final.parquet desde 'preparada/' y realiza el split temporal
    train/test, luego guarda los datasets en 'splits/'.
    
    Parámetros:
    -----------
    execution_date : str
        Fecha de ejecución en formato 'YYYY-MM-DD'.
    train_ratio : float
        Proporción de semanas para entrenamiento (default=0.7)
    """

    
    base_path = os.path.join(os.getcwd(), execution_date)
    preparada_path = os.path.join(base_path, "preparada")
    splits_path = os.path.join(base_path, "splits")

    # Archivo df_final
    df_path = os.path.join(preparada_path, "df_final.parquet")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"No se encontró df_final.parquet en {df_path}")
    
    # Cargar dataset
    df_final = pd.read_parquet(df_path)
    
    # Ordenar y obtener semanas únicas
    dataset = df_final.sort_values('week')
    weeks = dataset['week'].sort_values().unique()
    n_weeks = len(weeks)
    
    # Definir semanas de train y test
    train_weeks = weeks[:int(train_ratio * n_weeks)]
    test_weeks = weeks[int(train_ratio * n_weeks):]
    
    # Crear DataFrames
    train_df = dataset[dataset['week'].isin(train_weeks)].copy()
    test_df = dataset[dataset['week'].isin(test_weeks)].copy()
    
    # Guardar en splits/
    os.makedirs(splits_path, exist_ok=True)
    train_file = os.path.join(splits_path, "train.parquet")
    test_file = os.path.join(splits_path, "test.parquet")
    
    train_df.to_parquet(train_file, index=False, compression='gzip')
    test_df.to_parquet(test_file, index=False, compression='gzip')
    
    print(f"Train guardado en: {train_file}, filas: {len(train_df)}")
    print(f"Test guardado en: {test_file}, filas: {len(test_df)}")
