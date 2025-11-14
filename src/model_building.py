from typing import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """Crea el preprocesador para datos numéricos y categóricos."""
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return transformer


def build_classifier() -> RandomForestClassifier:
    """Crea el modelo de clasificación (Random Forest)."""
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return clf


def build_pipeline(
    preprocessor: ColumnTransformer,
    classifier: RandomForestClassifier,
) -> Pipeline:
    """
    Une el preprocesador y el clasificador en un único Pipeline
    para facilitar entrenamiento y predicción.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline
