import pandas as pd
from . import config


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Carga el dataset de ACV desde CSV."""
    if path is None:
        path = config.DATA_PATH
    df = pd.read_csv(path)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Verifica que el dataset tenga las columnas esperadas."""
    missing = set(config.EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {missing}")


def split_features_target(df: pd.DataFrame):
    """
    Separa el dataset en:
    - X: variables predictoras
    - y: variable objetivo (stroke)
    Elimina la columna 'id' porque no aporta información al modelo.
    """
    df = df.copy()

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    return X, y
