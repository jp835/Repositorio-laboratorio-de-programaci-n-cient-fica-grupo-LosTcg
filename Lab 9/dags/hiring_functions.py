import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import gradio as gr



def create_folders(**kwargs):
    
    execution_date = kwargs.get('ds') 
    base_path = os.path.join(os.getcwd(), execution_date)

    
    for folder in ['raw', 'splits', 'models']:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)

    print(f"Carpetas creadas en: {base_path}")
    return base_path


def split_data(execution_date=None):
    
    base_path = os.path.join(os.getcwd(), execution_date)
    raw_path = os.path.join(base_path, 'raw', 'data_1.csv')
    splits_path = os.path.join(base_path, 'splits')

    df = pd.read_csv(raw_path)

    
    target_col = 'HiringDecision'

    X = df.drop(columns=[target_col])
    y = df[target_col]

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Guardar los datos en splits/
    X_train[target_col] = y_train
    X_test[target_col] = y_test
    X_train.to_csv(os.path.join(splits_path, 'train.csv'), index=False)
    X_test.to_csv(os.path.join(splits_path, 'test.csv'), index=False)

    print("Datos divididos y guardados en la carpeta 'splits'.")



def preprocess_and_train(execution_date=None):
    base_path = os.path.join(os.getcwd(), execution_date)
    splits_path = os.path.join(base_path, 'splits')
    models_path = os.path.join(base_path, 'models')

    train = pd.read_csv(os.path.join(splits_path, 'train.csv'))
    test = pd.read_csv(os.path.join(splits_path, 'test.csv'))

    target_col = 'HiringDecision'
    X_train, y_train = train.drop(columns=[target_col]), train[target_col]
    X_test, y_test = test.drop(columns=[target_col]), test[target_col]

    # Detectar columnas numéricas y categóricas
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    
    model = RandomForestClassifier(random_state=42, n_estimators=200)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)

    
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score (clase positiva): {f1:.4f}")

    
    joblib.dump(pipeline, os.path.join(models_path, 'hiring_model.joblib'))
    print("Modelo guardado en carpeta 'models'.")





def predict(file,model_path):

    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    predictions = pipeline.predict(input_data)
    print(f'La prediccion es: {predictions}')
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]

    return {'Predicción': labels[0]}


def gradio_interface(execution_date=None):

    base_path = os.path.join(os.getcwd(), execution_date)
    model_path= os.path.join(base_path, 'models', 'hiring_model.joblib')

    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Vale será contratada o no."
    )
    interface.launch(share=True)

















