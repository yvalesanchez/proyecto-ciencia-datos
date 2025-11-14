from typing import List, Tuple
import pandas as pd
def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rellena valores faltantes en el dataset.
    Por ahora, solo se rellena 'bmi' con la mediana.
    """
    df = df.copy()
    if "bmi" in df.columns and df["bmi"].isna().any():
        df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    return df
def detect_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Devuelve dos listas:
    - columnas numéricas
    - columnas categóricas
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    return numeric_cols, categorical_cols
