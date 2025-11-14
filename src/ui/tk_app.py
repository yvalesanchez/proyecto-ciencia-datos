import tkinter as tk
from tkinter import ttk, messagebox

import joblib

from src import config
from src.prediction import predict_single
from . import mappings
from .inputs import StrokeInput
from .validators import parse_float


class StrokeApp(tk.Tk):
    def __init__(self, model=None):
        super().__init__()
        self.title("Predicción de Riesgo de ACV")
        self.geometry("650x520")
        self.resizable(False, False)

        self.model = model or self._load_model()
        self._create_widgets()

    # ---------- Carga del modelo ----------

    def _load_model(self):
        try:
            model = joblib.load(config.MODEL_PATH)
            return model
        except FileNotFoundError:
            messagebox.showerror(
                "Modelo no encontrado",
                "No se encontró el modelo entrenado.\n"
                "Ejecuta primero 'train_model.py' para entrenarlo.",
            )
            self.destroy()
            raise

    # ---------- Construcción de la interfaz ----------

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill="both", expand=True)

        title = ttk.Label(
            main_frame,
            text="Predicción de Riesgo de Accidente Cerebrovascular (ACV)",
            font=("Segoe UI", 12, "bold"),
        )
        title.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        subtitle = ttk.Label(
            main_frame,
            text="Introduce los datos del paciente y pulsa 'Predecir riesgo'.",
            font=("Segoe UI", 10),
        )
        subtitle.grid(row=1, column=0, columnspan=2, pady=(0, 20))

        row = 2

        # Variables de interfaz
        self.age_var = tk.StringVar()
        self.gender_var = tk.StringVar(value="Mujer")
        self.hta_var = tk.StringVar(value="No")
        self.cardio_var = tk.StringVar(value="No")
        self.married_var = tk.StringVar(value="Sí")
        self.work_var = tk.StringVar(value="Privado")
        self.residence_var = tk.StringVar(value="Urbano")
        self.glucose_var = tk.StringVar()
        self.bmi_var = tk.StringVar()
        self.smoke_var = tk.StringVar(value="Nunca fumó")

        # Campos de formulario
        self._add_labeled_entry(main_frame, "Edad (años):", self.age_var, row)
        row += 1

        self._add_labeled_combo(
            main_frame,
            "Género:",
            self.gender_var,
            list(mappings.GENDER_MAP.keys()),
            row,
        )
        row += 1

        self._add_labeled_combo(
            main_frame,
            "Hipertensión:",
            self.hta_var,
            list(mappings.YES_NO_MAP.keys()),
            row,
        )
        row += 1

        self._add_labeled_combo(
            main_frame,
            "Enfermedad cardíaca:",
            self.cardio_var,
            list(mappings.YES_NO_MAP.keys()),
            row,
        )
        row += 1

        self._add_labeled_combo(
            main_frame,
            "¿Alguna vez casado/a?:",
            self.married_var,
            list(mappings.MARRIED_MAP.keys()),
            row,
        )
        row += 1

        self._add_labeled_combo(
            main_frame,
            "Tipo de trabajo:",
            self.work_var,
            list(mappings.WORK_TYPE_MAP.keys()),
            row,
        )
        row += 1

        self._add_labeled_combo(
            main_frame,
            "Tipo de residencia:",
            self.residence_var,
            list(mappings.RESIDENCE_MAP.keys()),
            row,
        )
        row += 1

        self._add_labeled_entry(
            main_frame,
            "Glucosa promedio:",
            self.glucose_var,
            row,
        )
        row += 1

        self._add_labeled_entry(
            main_frame,
            "Índice de masa corporal (BMI):",
            self.bmi_var,
            row,
        )
        row += 1

        self._add_labeled_combo(
            main_frame,
            "Estado de fumador:",
            self.smoke_var,
            list(mappings.SMOKING_MAP.keys()),
            row,
        )
        row += 1

        # Botón
        predict_button = ttk.Button(
            main_frame,
            text="Predecir riesgo",
            command=self.on_predict_clicked,
        )
        predict_button.grid(row=row, column=0, columnspan=2, pady=20)

        # Ajustar columnas
        main_frame.columnconfigure(0, weight=0)
        main_frame.columnconfigure(1, weight=1)

    def _add_labeled_entry(self, parent, label_text, text_var, row):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)

        entry = ttk.Entry(parent, textvariable=text_var)
        entry.grid(row=row, column=1, sticky="ew", pady=4)

    def _add_labeled_combo(self, parent, label_text, text_var, values, row):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)

        combo = ttk.Combobox(
            parent,
            textvariable=text_var,
            values=values,
            state="readonly",
        )
        combo.grid(row=row, column=1, sticky="ew", pady=4)

    # ---------- Lógica de predicción ----------

    def _build_input(self) -> StrokeInput | None:
        # Validar numéricos
        age_val = parse_float(self.age_var.get(), "Edad")
        glucose_val = parse_float(self.glucose_var.get(), "Glucosa promedio")
        bmi_val = parse_float(self.bmi_var.get(), "BMI")

        errors = [f.error for f in (age_val, glucose_val, bmi_val) if f.error]
        if errors:
            messagebox.showerror("Datos inválidos", "\n".join(errors))
            return None

        stroke_input = StrokeInput(
            gender=mappings.GENDER_MAP[self.gender_var.get()],
            age=age_val.value,
            hypertension=mappings.YES_NO_MAP[self.hta_var.get()],
            heart_disease=mappings.YES_NO_MAP[self.cardio_var.get()],
            ever_married=mappings.MARRIED_MAP[self.married_var.get()],
            work_type=mappings.WORK_TYPE_MAP[self.work_var.get()],
            residence_type=mappings.RESIDENCE_MAP[self.residence_var.get()],
            avg_glucose_level=glucose_val.value,
            bmi=bmi_val.value,
            smoking_status=mappings.SMOKING_MAP[self.smoke_var.get()],
        )
        return stroke_input

    def on_predict_clicked(self):
        stroke_input = self._build_input()
        if stroke_input is None:
            return

        df = stroke_input.to_dataframe()
        result = predict_single(self.model, df)

        proba_percent = result["probability"] * 100
        label = result["label"]

        if label == 1:
            title = "⚠ Riesgo estimado ALTO"
            text = (
                f"El modelo estima un riesgo ALTO de accidente cerebrovascular.\n\n"
                f"Probabilidad aproximada: {proba_percent:.1f}%"
            )
        else:
            title = "✔ Riesgo estimado BAJO"
            text = (
                f"El modelo estima un riesgo BAJO de accidente cerebrovascular.\n\n"
                f"Probabilidad aproximada: {proba_percent:.1f}%"
            )

        messagebox.showinfo(title, text)


def run_app():
    app = StrokeApp()
    app.mainloop()
