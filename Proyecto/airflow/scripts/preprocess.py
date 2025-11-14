
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.selection import DropConstantFeatures


class TotalComprasProductoSemana(BaseEstimator, TransformerMixin):
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
        df['total_compras_producto_semana'] = df['total_compras_producto_semana'].fillna(df['promedio_compras_producto'])
        df['total_compras_producto_semana'] = df['total_compras_producto_semana'].fillna(self.promedio_global_)
        df.drop(columns=['promedio_compras_producto'], inplace=True)
        return df

class ProductosDiferentesPorClienteSemana(BaseEstimator, TransformerMixin):
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
        df_filtrado = X[X[self.col_compra] == 1]
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
        df.drop(columns=['promedio_cliente'], inplace=True)
        return df

class PromedioSemanalProducto(BaseEstimator, TransformerMixin):
    def __init__(self, product_col='product_id', week_col='semana', items_col='items'):
        self.product_col = product_col
        self.week_col = week_col
        self.items_col = items_col

    def fit(self, X, y=None):
        compras_semanales = X.groupby([self.product_col, self.week_col], observed=True)[self.items_col].sum().reset_index()
        self.promedio_por_producto_ = compras_semanales.groupby(self.product_col, observed=True)[self.items_col].mean().reset_index()
        self.promedio_por_producto_.rename(columns={self.items_col: 'items_promedio_por_semana'}, inplace=True)
        return self

    def transform(self, X):
        X = X.copy()
        X = pd.merge(
            X,
            self.promedio_por_producto_,
            on=self.product_col,
            how='left'
        )
        return X

class ReduceCorrelation(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.drop_columns_ = None

    def fit(self, X, y=None):
        df = X.copy()
        numeric_df = df.select_dtypes(include=[np.number])
        remaining_vars = list(numeric_df.columns)
        corr_matrix = numeric_df.corr().abs()

        while True:
            sub_corr = corr_matrix.loc[remaining_vars, remaining_vars].copy()
            np.fill_diagonal(sub_corr.values, np.nan)
            max_corr_value = sub_corr.max().max()
            if max_corr_value <= self.threshold:
                break
            var1, var2 = sub_corr.unstack().idxmax()
            sum_corr1 = sub_corr.loc[var1].sum()
            sum_corr2 = sub_corr.loc[var2].sum()
            to_drop = var1 if sum_corr1 > sum_corr2 else var2
            remaining_vars.remove(to_drop)

        self.drop_columns_ = list(set(numeric_df.columns) - set(remaining_vars))
        return self

    def transform(self, X):
        return X.drop(columns=self.drop_columns_, errors='ignore')

class IQR(BaseEstimator, TransformerMixin):
    def __init__(self, cols, lbda=1.5):
        self.cols = cols
        self.lbda = lbda

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.cols)
        self.bounds_ = {}
        for col in self.cols:
            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR_val = Q3 - Q1
            lower = Q1 - self.lbda * IQR_val
            upper = Q3 + self.lbda * IQR_val
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.cols)
        for col, (lower, upper) in self.bounds_.items():
            X_df[col] = np.clip(X_df[col], lower, upper)
        return X_df
    
    def get_feature_names_out(self, input_features=None):
        return input_features

class PassFeatureNames(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X
    def get_feature_names_out(self, input_features=None): return input_features

class ColumnTypeSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.numeric_features_ = X.select_dtypes(include='number').columns.tolist()
        self.categorical_features_ = X.select_dtypes(exclude='number').columns.tolist()
        return self
    def transform(self, X): return X
    def get_numeric_features(self): return self.numeric_features_
    def get_categorical_features(self): return self.categorical_features_

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')


def get_engineering_pipeline():
    engineering_pipeline = Pipeline([
        # ... (otros pasos)
        ("drop_cols", DropColumns(['product_id', 'customer_id', 'compró', 'items', 'week', 'año', 'semana', 'sub_category'])), # <-- CORRECCIÓN AQUÍ
        ("Eliminar_cte", DropConstantFeatures(tol=0.85)),
        ("reduce_corr", ReduceCorrelation(threshold=0.90)),
        ("detect_types", ColumnTypeSelector())
    ])
    return engineering_pipeline

def get_preprocessor(numeric_features, categorical_features):
    """
    Retorna el ColumnTransformer para escalar y codificar.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("pass_names_for_iqr", PassFeatureNames()), # Pasa nombres para IQR
        ("outlier", IQR(cols=numeric_features, lbda=1.5)),
        ("scaler", MinMaxScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False # Mantiene los nombres de las columnas limpios
    )
    return preprocessor