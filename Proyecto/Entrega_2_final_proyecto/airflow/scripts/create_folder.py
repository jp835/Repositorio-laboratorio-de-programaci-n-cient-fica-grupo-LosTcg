
import os
def create_folders(**kwargs):

    """
    Se crean carpetas a utilizar
    """

    
    execution_date = kwargs.get('ds') 
    base_path = os.path.join(os.getcwd(), execution_date)

    
    for folder in ["preparada", 'splits', 'mlflow']:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)

    print(f"Carpetas creadas en: {base_path}")
    return base_path

