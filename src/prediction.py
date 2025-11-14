from typing import Dict, Any

import pandas as pd
from sklearn.pipeline import Pipeline


def predict_single(model: Pipeline, features: pd.DataFrame) -> Dict[str, Any]:
    """
    Realiza la predicción para una sola fila (un paciente).
    'features' debe ser un DataFrame con 1 fila.
    """
    proba = model.predict_proba(features)[0, 1]
    label = int(model.predict(features)[0])

    return {
        "probability": float(proba),
            "label": label,
    }
