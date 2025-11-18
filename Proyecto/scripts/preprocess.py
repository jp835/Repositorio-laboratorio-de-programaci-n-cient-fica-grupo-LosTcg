# En airflow/scripts/preprocess.py
"""
Script de soporte para la Tarea 2 de Airflow (Preprocesamiento).
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
import warnings

warnings.filterwarnings("ignore")

# --- 1. CLASES DE TRANSFORMACIÓN ---

class TotalComprasProductoSemana(BaseEstimator, TransformerMixin):
    """ Crea 'total_compras_producto_semana'. """
    def __init__(self, group_cols=None, value_col='compró'):
        self.group_cols = group_cols or ['product_id', 'semana']
        self.value_col = value_col
        self.total_compras_ = None
        self.promedio_por_producto_ = None
        self.promedio_global_ = None

    def fit(self, X, y=None):
        df = X.copy()
        self.total_compras_ = (
            df.groupby(self.group_cols, observed=True)[self.value_col]
            .sum()
            .reset_index()
            .rename(columns={self.value_col: 'total_compras_producto_semana'})
        )
        # ... (Lógica de fit: calcula promedios para imputación) ...
        producto_col = self.group_cols[0]
        self.promedio_por_producto_ = (
            self.total_compras_
            .groupby(producto_col, observed=True)['total_compras_producto_semana']
            .mean()
            .reset_index()
            .rename(columns={'total_compras_producto_semana': 'promedio_compras_producto'})
        )
        self.promedio_global_ = self.promedio_por_producto_['promedio_compras_producto'].mean()
        return self

    def transform(self, X):
        df = X.copy()
        df = pd.merge(df, self.total_compras_, on=self.group_cols, how='left')
        producto_col = self.group_cols[0]
        df = pd.merge(df, self.promedio_por_producto_, on=producto_col, how='left')
        
        # Imputa usando el promedio por producto o global si no hay match
        df['total_compras_producto_semana'] = df['total_compras_producto_semana'].fillna(df['promedio_cliente'])
        df['total_compras_producto_semana'] = df['total_compras_producto_semana'].fillna(self.promedio_global_)
        
        df.drop(columns=['promedio_cliente'], inplace=True, errors='ignore')
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
        compras = y if y is not None else X[self.col_compra]
        df_filtrado = X[compras == 1]
        
        self.resultado_ = (
            df_filtrado
            .groupby([self.col_cliente, self.col_semana], observed=True)
            .agg(productos_diferentes_semana=(self.col_producto, 'nunique'))
            .reset_index()
        )
        self.promedios_cliente_ = (
            self.resultado_
            .groupby(self.col_cliente, observed=True)['productos_diferentes_semana']
            .mean()
            .reset_index()
            .rename(columns={'productos_diferentes_semana': 'promedio_cliente'})
        )
        self.promedio_global_ = self.promedios_cliente_['promedio_cliente'].mean()
        return self

    def transform(self, X):
        df = X.copy()
        df = df.merge(self.resultado_, on=[self.col_cliente, self.col_semana], how='left')
        df = df.merge(self.promedios_cliente_, on=self.col_cliente, how='left')
        
        df['productos_diferentes_semana'] = df['productos_diferentes_semana'].fillna(df['promedio_cliente'])
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
        compras_semanales = X.groupby([self.product_col, self.week_col], observed=True)[self.items_col].sum().reset_index()
        self.promedio_por_producto_ = compras_semanales.groupby(self.product_col, observed=True)[self.items_col].mean().reset_index()
        self.promedio_por_producto_.rename(columns={self.items_col: 'items_promedio_por_semana'}, inplace=True)
        self.promedio_global_ = self.promedio_por_producto_['items_promedio_por_semana'].mean()
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out = pd.merge(
            X_out,
            self.promedio_por_producto_,
            on=self.product_col,
            how='left'
        )
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
        
        if numeric_df.empty:
            self.drop_columns_ = []
            return self
            
        remaining_vars = list(numeric_df.columns)
        corr_matrix = numeric_df.corr().abs()
        
        # ... (Lógica para determinar qué columnas dropear basada en correlación) ...
        while True:
            sub_corr = corr_matrix.loc[remaining_vars, remaining_vars].copy()
            np.fill_diagonal(sub_corr.values, np.nan)
            max_corr_value = sub_corr.max().max()
            
            if (max_corr_value <= self.threshold) or pd.isna(max_corr_value):
                break
                
            var1, var2 = sub_corr.unstack().idxmax()
            sum_corr1 = sub_corr.loc[var1].sum()
            sum_corr2 = sub_corr.loc[var2].sum()
            to_drop = var1 if sum_corr1 > sum_corr2 else var2
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
        self.numeric_features_ = X.select_dtypes(include='number').columns.tolist()
        self.categorical_features_ = X.select_dtypes(exclude='number').columns.tolist()
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


# --- 2. FUNCIONES DE PIPELINE ---

def get_engineering_pipeline():
    """
    Pipeline que aplica la ingeniería de características.
    """
    engineering_pipeline = Pipeline([
        ('feat_prods_diferentes', ProductosDiferentesPorClienteSemana(columna_compra='compró')),
        ('feat_total_compras', TotalComprasProductoSemana(value_col='compró')),
        ('feat_promedio_semanal', PromedioSemanalProducto(items_col='items')),
        
        ("drop_cols_originales", DropColumns([
            'product_id', 
            'customer_id', 
            'compró',      # Target
            'items',       # Leakage
            'week',        # Reemplazada por 'semana' y 'año'
            'sub_category' 
        ])), 
        
        ("drop_constantes", DropConstantFeatures(tol=0.85)),
        ("drop_correlacionadas", ReduceCorrelation(threshold=0.90)),
        ("detectar_tipos_finales", ColumnTypeSelector())
    ])
    return engineering_pipeline

# --- 3. PREPROCESSOR SIMPLIFICADO ---

def get_preprocessor(numeric_features, categorical_features):
    """
    Este pipeline toma el DataFrame de ingeniería y SOLO imputa NaNs.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputador_mediana", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputador_moda", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("transform_num", numeric_pipeline, numeric_features),
            ("transform_cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor