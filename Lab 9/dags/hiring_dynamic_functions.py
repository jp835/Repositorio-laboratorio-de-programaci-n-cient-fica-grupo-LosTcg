import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib


def create_folders(**kwargs):

    
    execution_date = kwargs.get('ds') 
    base_path = os.path.join(os.getcwd(), execution_date)

    
    for folder in ['raw', 'preprocessed', 'models','splits']:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)

    print(f"Carpetas creadas en: {base_path}")
    return base_path


def load_and_merge(execution_date = None):
    base_path = os.path.join(os.getcwd(), execution_date)
    raw_path = os.path.join(base_path, 'raw')
    preprocessed_path = os.path.join(base_path, 'preprocessed')

    data1_path = os.path.join(raw_path, 'data_1.csv')
    data2_path = os.path.join(raw_path, 'data_2.csv')

    datas = []
    if os.path.exists(data1_path):
        data1 = pd.read_csv(data1_path)
        datas.append(data1)
    else:
        print("Data 1 no cargada")

    if os.path.exists(data2_path):
        data2 = pd.read_csv(data2_path)
        datas.append(data2)
    else:
        print("Data 2 no cargada")
    #Esto de abajo ayuda a no retornar error si es que 
    #Ninguno de los data set esta en la carpeta
    if len(datas) == 0:
        print("Ningun data set cargado")
        return None
    
    df_concat = pd.concat(datas, ignore_index=True)
    df_concat.to_csv(os.path.join(preprocessed_path, 'merged.csv'), index=False)
    print('listo')



def split_data(execution_date=None):

    base_path = os.path.join(os.getcwd(), execution_date)
    preprocessed_path = os.path.join(base_path, 'preprocessed', 'merged.csv')
    splits_path = os.path.join(base_path, 'splits')

    df = pd.read_csv(preprocessed_path)


    target_col = 'HiringDecision'

    X = df.drop(columns=[target_col])
    y = df[target_col]

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    
    X_train[target_col] = y_train
    X_test[target_col] = y_test
    X_train.to_csv(os.path.join(splits_path, 'train.csv'), index=False)
    X_test.to_csv(os.path.join(splits_path, 'test.csv'), index=False)

    print("Datos divididos y guardados en la carpeta 'splits'.")


def train_model(modelo, execution_date=None):
    base_path = os.path.join(os.getcwd(), execution_date)
    splits_path = os.path.join(base_path, 'splits')
    models_path = os.path.join(base_path, 'models')

    train = pd.read_csv(os.path.join(splits_path, 'train.csv'))

    target_col = 'HiringDecision'
    X_train, y_train = train.drop(columns=[target_col]), train[target_col]

    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', modelo)
    ])

    pipeline.fit(X_train, y_train)

    
    model_name = type(modelo).__name__
    model_path = os.path.join(models_path, f"{model_name}_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Modelo {model_name} guardado en {model_path}")


def evaluate_models(execution_date = None):
    base_path = os.path.join(os.getcwd(), execution_date)
    splits_path = os.path.join(base_path, 'splits')
    models_path = os.path.join(base_path, 'models')

    test = pd.read_csv(os.path.join(splits_path, 'test.csv'))

    target_col = 'HiringDecision'

    X_test, y_test= test.drop(columns=[target_col]), test[target_col]

    modelos = os.listdir(models_path) #Esto tendra todos los archivos de los modleos generados

    best_accuracy = 0
    mejor_modelo = None
    nombre_mejor_modelo = ""
    #Al final lo anterior tendra el mejor modelo de todos luego del for y el if

    for modelo in modelos:
        model_path = os.path.join(models_path, modelo)
        pipeline = joblib.load(model_path)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Modelo {modelo}: Accuracy = {acc:.4f}") #Se imprime la accuracy de cada modelo
        if acc>best_accuracy:
            best_accuracy = acc
            mejor_modelo =  pipeline
            nombre_mejor_modelo = modelo

    
    if nombre_mejor_modelo != "":
        best_model_path = os.path.join(models_path, 'best_model.joblib')
        joblib.dump(mejor_modelo, best_model_path)
        print(f"Mejor modelo: {nombre_mejor_modelo} con accuracy {best_accuracy:.4f}")
    else:
        print("No hay modelos en carpeta Models para comparar")






