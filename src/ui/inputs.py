from dataclasses import dataclass

import pandas as pd


@dataclass
class StrokeInput:
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

    def to_dataframe(self) -> pd.DataFrame:
        """Convierte los datos a un DataFrame con una sola fila."""
        data = {
            "gender": self.gender,
            "age": self.age,
            "hypertension": self.hypertension,
            "heart_disease": self.heart_disease,
            "ever_married": self.ever_married,
            "work_type": self.work_type,
            "Residence_type": self.residence_type,
            "avg_glucose_level": self.avg_glucose_level,
            "bmi": self.bmi,
            "smoking_status": self.smoking_status,
        }
        return pd.DataFrame([data])