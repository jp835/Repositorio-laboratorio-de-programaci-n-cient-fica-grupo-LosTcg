# airflow/scripts/preprocess.py
"""
Script de soporte para el pipeline de Airflow (Preprocesamiento).
Contiene todas las clases de transformación personalizadas de Sklearn
y las funciones que construyen los pipelines de ingeniería y preprocesamiento.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.selection import DropConstantFeatures
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import warnings

warnings.filterwarnings("ignore")



class TotalComprasProductoSemana(BaseEstimator, TransformerMixin):
    """ Crea 'total_compras_producto_semana' y maneja imputación. """
    def __init__(self, group_cols=None, product_col='product_id', week_col='semana'):
        self.group_cols = group_cols or [product_col, week_col]
        self.product_col = product_col
        self.total_compras_ = None
        self.promedio_por_producto_ = None
        self.promedio_global_ = None

    def fit(self, X, y=None):
        df = X.copy()
        # Asegurarse de tener la columna 'compró' para filtrar
        if y is not None:
            df['compró'] = y.copy()
        elif 'compró' in df.columns:
            pass  # Ya existe la columna
        else:
            raise ValueError("Columna 'compró' no encontrada en X ni proporcionada en y")

        # 1. Calcular total de compras (compró=1) por grupo (producto y semana)
        df_compras = df[df['compró'] == 1]
        if len(df_compras) > 0:
            self.total_compras_ = (
                df_compras.groupby(self.group_cols, observed=True)['compró']
                .sum()
                .reset_index()
                .rename(columns={'compró': 'total_compras_producto_semana'})
            )
            
            # 2. Calcular promedio por producto (para imputación en transform)
            self.promedio_por_producto_ = (
                self.total_compras_
                .groupby(self.product_col, observed=True)['total_compras_producto_semana']
                .mean()
                .reset_index()
                .rename(columns={'total_compras_producto_semana': 'promedio_compras_producto'})
            )
            # 3. Calcular promedio global (respaldo)
            self.promedio_global_ = self.promedio_por_producto_['promedio_compras_producto'].mean()
        else:
            # Si no hay compras, crear estructuras vacías
            self.total_compras_ = pd.DataFrame(columns=self.group_cols + ['total_compras_producto_semana'])
            self.promedio_por_producto_ = pd.DataFrame(columns=[self.product_col, 'promedio_compras_producto'])
            self.promedio_global_ = 0
        
        if pd.isna(self.promedio_global_):
            self.promedio_global_ = 0 # Fallback si no hay compras en el fit
            
        return self

    def transform(self, X):
        df = X.copy()
        
        # Merge con las compras totales observadas
        if len(self.total_compras_) > 0:
            df = pd.merge(df, self.total_compras_, on=self.group_cols, how='left')
        
        # Merge con el promedio por producto (para imputar NaNs)
        if len(self.promedio_por_producto_) > 0:
            df = pd.merge(df, self.promedio_por_producto_, on=self.product_col, how='left')
        
        # Imputación (Usar promedio_compras_producto si es NaN)
        if 'promedio_compras_producto' in df.columns:
            df['total_compras_producto_semana'] = df['total_compras_producto_semana'].fillna(df['promedio_compras_producto'])
        
        # Imputación final (Usar promedio_global_ si sigue siendo NaN - caso de producto nuevo)
        df['total_compras_producto_semana'] = df['total_compras_producto_semana'].fillna(self.promedio_global_)
        
        df.drop(columns=['promedio_compras_producto'], inplace=True, errors='ignore')
        return df

class ProductosDiferentesPorClienteSemana(BaseEstimator, TransformerMixin):
    """ Crea 'productos_diferentes_semana'. """
    def __init__(self, columna_cliente='customer_id', columna_semana='semana',
                 columna_producto='product_id', columna_compra='compró'):
        self.col_cliente = columna_cliente
        self.col_semana = columna_semana
        self.col_producto = columna_producto
        self.col_compra = columna_compra
        self.resultado_ = None
        self.promedios_cliente_ = None
        self.promedio_global_ = None

    def fit(self, X, y=None):
        # Asegurarse de tener la columna 'compró' para filtrar
        df = X.copy()
        if y is not None:
            df[self.col_compra] = y.copy()
        elif self.col_compra in df.columns:
            pass  # Ya existe la columna
        else:
            raise ValueError(f"Columna '{self.col_compra}' no encontrada en X ni proporcionada en y")
        
        # Filtrar solo compras
        df_filtrado = df[df[self.col_compra] == 1]
        
        if len(df_filtrado) > 0:
            # 1. Productos diferentes por cliente y semana
            self.resultado_ = (
                df_filtrado
                .groupby([self.col_cliente, self.col_semana], observed=True)
                .agg(productos_diferentes_semana=(self.col_producto, 'nunique'))
                .reset_index()
            )
            
            # 2. Promedio por cliente (para imputación en transform)
            self.promedios_cliente_ = (
                self.resultado_
                .groupby(self.col_cliente, observed=True)['productos_diferentes_semana']
                .mean()
                .reset_index()
                .rename(columns={'productos_diferentes_semana': 'promedio_cliente'})
            )
            
            # 3. Promedio global (respaldo)
            self.promedio_global_ = self.promedios_cliente_['promedio_cliente'].mean()
        else:
            # Si no hay compras, crear estructuras vacías
            self.resultado_ = pd.DataFrame(columns=[self.col_cliente, self.col_semana, 'productos_diferentes_semana'])
            self.promedios_cliente_ = pd.DataFrame(columns=[self.col_cliente, 'promedio_cliente'])
            self.promedio_global_ = 0

        if pd.isna(self.promedio_global_):
            self.promedio_global_ = 0 # Fallback si no hay compras en el fit

        return self

    def transform(self, X):
        df = X.copy()
        if len(self.resultado_) > 0:
            df = df.merge(self.resultado_, on=[self.col_cliente, self.col_semana], how='left')
        if len(self.promedios_cliente_) > 0:
            df = df.merge(self.promedios_cliente_, on=self.col_cliente, how='left')
        
        # Imputación (Usar promedio_cliente si es NaN)
        if 'promedio_cliente' in df.columns:
            df['productos_diferentes_semana'] = df['productos_diferentes_semana'].fillna(df['promedio_cliente'])
        
        # Imputación final (Usar promedio_global_ si sigue siendo NaN - caso de cliente nuevo)
        df['productos_diferentes_semana'] = df['productos_diferentes_semana'].fillna(self.promedio_global_)
        
        df.drop(columns=['promedio_cliente'], inplace=True, errors='ignore')
        return df

class PromedioSemanalProducto(BaseEstimator, TransformerMixin):
    """ Crea 'items_promedio_por_semana'. """
    def __init__(self, product_col='product_id', week_col='semana', items_col='items'):
        self.product_col = product_col
        self.week_col = week_col
        self.items_col = items_col
        self.promedio_por_producto_ = None
        self.promedio_global_ = None

    def fit(self, X, y=None):
        # Suma de items por producto y semana (solo donde hay compras)
        df_compras = X[X['items'] > 0] if 'items' in X.columns else X
        
        if len(df_compras) > 0:
            compras_semanales = df_compras.groupby([self.product_col, self.week_col], observed=True)[self.items_col].sum().reset_index()
            
            # Promedio de items por producto (a lo largo de todas las semanas)
            self.promedio_por_producto_ = compras_semanales.groupby(self.product_col, observed=True)[self.items_col].mean().reset_index()
            self.promedio_por_producto_.rename(columns={self.items_col: 'items_promedio_por_semana'}, inplace=True)
            self.promedio_global_ = self.promedio_por_producto_['items_promedio_por_semana'].mean()
        else:
            self.promedio_por_producto_ = pd.DataFrame(columns=[self.product_col, 'items_promedio_por_semana'])
            self.promedio_global_ = 0
        
        if pd.isna(self.promedio_global_):
            self.promedio_global_ = 0 # Fallback si no hay compras en el fit
        
        return self

    def transform(self, X):
        X_out = X.copy()
        if len(self.promedio_por_producto_) > 0:
            X_out = pd.merge(
                X_out,
                self.promedio_por_producto_,
                on=self.product_col,
                how='left'
            )
        else:
            X_out['items_promedio_por_semana'] = 0.0
            
        # Imputación final con promedio global
        X_out['items_promedio_por_semana'] = X_out['items_promedio_por_semana'].fillna(self.promedio_global_)
        return X_out



class ReduceCorrelation(BaseEstimator, TransformerMixin):
    """ Elimina features numéricas altamente correlacionadas. """
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.drop_columns_ = None

    def fit(self, X, y=None):
        df = X.copy()
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty or len(numeric_df.columns) < 2:
            self.drop_columns_ = []
            return self
            
        remaining_vars = list(numeric_df.columns)
        corr_matrix = numeric_df.corr().abs()
        
        while True:
            sub_corr = corr_matrix.loc[remaining_vars, remaining_vars].copy()
            np.fill_diagonal(sub_corr.values, np.nan)
            
            if sub_corr.empty:
                break
                
            max_corr_value = sub_corr.max().max()
            
            if (max_corr_value <= self.threshold) or pd.isna(max_corr_value):
                break
                
            sum_corrs = sub_corr.sum()
            to_drop = sum_corrs.idxmax()

            remaining_vars.remove(to_drop)

        self.drop_columns_ = list(set(numeric_df.columns) - set(remaining_vars))
        return self

    def transform(self, X):
        if self.drop_columns_ is None:
             raise RuntimeError("Debes llamar a fit() antes de transform()")
        return X.drop(columns=self.drop_columns_, errors='ignore')

class ColumnTypeSelector(BaseEstimator, TransformerMixin):
    """ Detecta y almacena los nombres de las features numéricas y categóricas. """
    def __init__(self):
        self.numeric_features_ = []
        self.categorical_features_ = []

    def fit(self, X, y=None):
        # Excluir 'compró' e 'items' del set final de features
        cols_to_exclude = ['compró', 'items', 'week', 'product_id', 'customer_id']
        
        self.numeric_features_ = X.select_dtypes(include=['number']).columns.tolist()
        self.categorical_features_ = X.select_dtypes(exclude=['number', 'datetime64']).columns.tolist()
        
        self.numeric_features_ = [col for col in self.numeric_features_ if col not in cols_to_exclude]
        self.categorical_features_ = [col for col in self.categorical_features_ if col not in cols_to_exclude]
        return self
    
    def transform(self, X): return X
    
    def get_numeric_features(self): return self.numeric_features_
    
    def get_categorical_features(self): return self.categorical_features_

class DropColumns(BaseEstimator, TransformerMixin):
    """ Un wrapper de Scikit-learn para pd.DataFrame.drop(). """
    def __init__(self, columns): self.columns = columns
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')

class IQRTransformer(BaseEstimator, TransformerMixin):
    """ Winsorización simple con IQR (clip a los límites). """
    def __init__(self, lbda=1.5):
        self.lbda = lbda
        self.bounds_ = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        if not isinstance(X, pd.DataFrame):
            # Asumir que las columnas son numéricas si no es un DataFrame
             self.feature_names_in_ = [f"num__{i}" for i in range(X.shape[1])]
             X_df.columns = self.feature_names_in_
        else:
            self.feature_names_in_ = X.columns.tolist()

        for col in self.feature_names_in_:
            if pd.api.types.is_numeric_dtype(X_df[col]):
                Q1 = X_df[col].quantile(0.25)
                Q3 = X_df[col].quantile(0.75)
                IQR_val = Q3 - Q1
                lower = Q1 - self.lbda * IQR_val
                upper = Q3 + self.lbda * IQR_val
                self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X_out = pd.DataFrame(X, columns=self.feature_names_in_)
        for col, (lower, upper) in self.bounds_.items():
            X_out[col] = np.clip(X_out[col], lower, upper)
        return X_out
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

# --- 2. FUNCIONES DE PIPELINE ---

def get_engineering_pipeline():
    """
    Pipeline que aplica la ingeniería de características (previo a ColumnTransformer).
    """
    # Columnas que contienen leakage de información o son ID sin valor predictivo
    cols_to_drop = [
        'product_id', 
        'customer_id', 
        'compró',      
        'items',       
        
    ]

    engineering_pipeline = Pipeline([
        ('feat_prods_diferentes', ProductosDiferentesPorClienteSemana()),
        ('feat_total_compras', TotalComprasProductoSemana()),
        ('feat_promedio_semanal', PromedioSemanalProducto()),
        
        ("drop_ids_leakage", DropColumns(cols_to_drop)),
        ("detectar_tipos_inicial", ColumnTypeSelector()), # Captura los tipos antes de la reducción
        ("drop_constantes", DropConstantFeatures(tol=0.85)), # Elimina features con poca varianza
        ("drop_correlacionadas", ReduceCorrelation(threshold=0.90)), # Reduce multicolinealidad en numéricas
        ("detectar_tipos_finales", ColumnTypeSelector()) # Captura la lista final de features
    ])
    return engineering_pipeline



def get_preprocessor(numeric_features, categorical_features):
    """
    Este pipeline toma el DataFrame de ingeniería y aplica transformaciones finales.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputador_mediana", SimpleImputer(strategy="median")),
        ("outlier_iqr", IQRTransformer(lbda=1.5)), # Aplicado a las features numéricas
        ("scaler", MinMaxScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputador_moda", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("transform_num", numeric_pipeline, numeric_features),
            ("transform_cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor