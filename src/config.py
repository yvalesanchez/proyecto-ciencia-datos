from pathlib import Path

# Carpeta base del proyecto (dos niveles arriba de este archivo)
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas importantes del proyecto
DATA_PATH = BASE_DIR / "data" / "healthcare-dataset-stroke-data.csv"
MODEL_PATH = BASE_DIR / "models" / "stroke_model.pkl"

# Columnas esperadas en el dataset de Kaggle de ACV
EXPECTED_COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]
