import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
)

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Análisis de ACV (Stroke) — Datos limpios",
    layout="wide",
    page_icon="🧠",
)

# ----------------- ESTILOS (CSS) -----------------
st.markdown(
    """
    <style>
        body {
            background-color: #020617;
            color: #e2e8f0;
        }
        .main {
            background: radial-gradient(circle at top left, #1e293b 0%, #020617 55%);
            color: #e2e8f0;
        }
        .big-title {
            font-size: 2.6rem;
            font-weight: 900;
            color: #f9fafb;
        }
        .subtitle {
            font-size: 1.05rem;
            color: #cbd5e1;
        }
        .card {
            background: rgba(15, 23, 42, 0.9);
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.6);
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.8);
        }
        .pill {
            display: inline-block;
            padding: 0.28rem 0.8rem;
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e, #0ea5e9);
            color: #020617;
            font-size: 0.8rem;
            font-weight: 700;
            margin-right: 0.35rem;
            margin-bottom: 0.25rem;
        }
        h1, h2, h3, h4 {
            color: #f9fafb !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.35rem 0.9rem;
            background-color: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.5);
            color: #e5e7eb;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: linear-gradient(90deg, #22c55e, #0ea5e9);
            color: #020617;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #22c55e, #0ea5e9) !important;
            color: #020617 !important;
            border-color: transparent !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# RUTAS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

# Usamos SIEMPRE el dataset limpio exportado desde el notebook
DATA_PATH = BASE_DIR.parent / "notebooks" / "data" / "stroke_clean.csv"

IMAGE1 = BASE_DIR / "acv.jpg"
IMAGE2 = BASE_DIR / "acv2.jpg"
IMAGE3 = BASE_DIR / "acv3.jpg"
IMAGE4 = BASE_DIR / "acv4.jpg"


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# =========================================================
# CARGA DE DATOS LIMPIOS
# =========================================================
if not DATA_PATH.exists():
    st.error(
        f"❌ No se encontró el archivo limpio `stroke_clean.csv`.\n\n"
        f"Se esperaba en:\n`{DATA_PATH}`"
    )
    st.stop()

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error("Ocurrió un error al cargar los datos limpios:")
    st.exception(e)
    st.stop()

if "stroke" not in df.columns:
    st.error("El dataset limpio no tiene la columna 'stroke'. Revisa el archivo.")
    st.write("Columnas encontradas:", list(df.columns))
    st.stop()

df_viz = df.copy()
df_viz["stroke_label"] = df_viz["stroke"].map({0: "Sin ataque", 1: "Ataque"})

# =========================================================
# FUNCIÓN AUXILIAR PARA BOXPLOT SIN ERRORES
# =========================================================
def boxplot_matplotlib_by_stroke(ax, df_in: pd.DataFrame, y_col: str, ylabel: str):
    """
    Hace un boxplot manual con matplotlib (sin seaborn) para evitar errores
    raros de seaborn cuando las posiciones o las estadísticas no coinciden.
    """
    data = df_in[["stroke_label", y_col]].dropna()
    if data.empty:
        ax.text(0.5, 0.5, "No hay datos suficientes", ha="center", va="center")
        ax.set_axis_off()
        return

    grupos = []
    etiquetas = []
    for label in data["stroke_label"].unique():
        vals = data.loc[data["stroke_label"] == label, y_col].values
        if len(vals) > 0:
            grupos.append(vals)
            etiquetas.append(label)

    if len(grupos) == 0:
        ax.text(0.5, 0.5, "Sin datos para graficar", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.boxplot(grupos, labels=etiquetas)
    ax.set_xlabel("ACV")
    ax.set_ylabel(ylabel)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🧠 ACV Dashboard (Datos limpios)")
    st.markdown(
        """
        Proyecto realizado por:

        **✨ Yeimy Valentina Sánchez**  
        **✨ Darikson Pérez**

        Utilizando datos clínicos limpios y balanceados para
        analizar el riesgo de Accidente Cerebrovascular (ACV).
        """
    )
    st.markdown("---")
    st.caption("Ciencia de Datos en Salud • Streamlit • RandomForest")

# =========================================================
# PORTADA
# =========================================================
c1, c2 = st.columns([2, 1])

with c1:
    st.markdown(
        '<div class="big-title">Análisis de Accidente Cerebrovascular (ACV)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p class="subtitle">
        Este panel interactivo se basa en un dataset <b>previamente limpiado y balanceado</b> 
        (<code>stroke_clean.csv</code>), donde se han tratado valores atípicos, nulos y se ha
        equilibrado la variable objetivo <b>stroke</b>.  
        El objetivo es explorar los datos, visualizar patrones y construir un modelo de 
        <b>Machine Learning (RandomForest)</b> para estimar el riesgo de ACV.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
        <b>🎯 Objetivos principales</b>
        <ul>
            <li>Explorar las características clínicas y demográficas de los pacientes.</li>
            <li>Visualizar la relación entre las variables y la aparición de ACV.</li>
            <li>Entrenar un modelo predictivo (RandomForest) con datos limpios y balanceados.</li>
            <li>Permitir una <b>predicción interactiva</b> del riesgo de ACV para un paciente simulado.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <span class="pill">Datos limpios</span>
        <span class="pill">SMOTE</span>
        <span class="pill">RandomForest</span>
        <span class="pill">Predicción de riesgo</span>
        <span class="pill">Salud</span>
        """,
        unsafe_allow_html=True,
    )

with c2:
    if IMAGE1.exists():
        st.image(str(IMAGE1), caption="Ilustración de ACV", use_column_width=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**📊 Resumen de stroke_clean.csv**")
    st.write(f"Filas: **{df.shape[0]}**")
    st.write(f"Columnas: **{df.shape[1]}**")
    st.write("Variable objetivo: **stroke** (0 = Sin ataque, 1 = Ataque)")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# TABS PRINCIPALES
# =========================================================
tab_data, tab_eda, tab_model, tab_pred = st.tabs(
    [
        "📄 Datos limpios",
        "📊 Análisis Exploratorio",
        "🤖 Modelo RandomForest",
        "🎈 Predicción interactiva",
    ]
)

# =========================================================
# TAB 1: DATOS
# =========================================================
with tab_data:
    st.header("📄 Vista general de los datos limpios")

    st.dataframe(df.head(), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Tipos de columnas y nulos")
        info = pd.DataFrame(
            {
                "Columna": df.columns,
                "Tipo": df.dtypes.astype(str),
                "Nulos": df.isna().sum(),
            }
        )
        st.dataframe(info, use_container_width=True)

    with col_b:
        st.markdown("### Estadísticos descriptivos (numéricos)")
        st.dataframe(df.describe().T, use_container_width=True)

    if IMAGE2.exists():
        st.image(
            str(IMAGE2),
            caption="Proceso de limpieza y depuración de los datos",
            use_column_width=True,
        )

# =========================================================
# TAB 2: EDA
# =========================================================
with tab_eda:
    st.header("📊 Análisis Exploratorio de los datos limpios")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Edad según presencia de ACV")
        fig, ax = plt.subplots()
        boxplot_matplotlib_by_stroke(ax, df_viz, "age", "Edad")
        st.pyplot(fig)

    with col2:
        st.markdown("#### Glucosa promedio según ACV")
        fig, ax = plt.subplots()
        boxplot_matplotlib_by_stroke(ax, df_viz, "avg_glucose_level", "avg_glucose_level")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        if "bmi" in df_viz.columns:
            st.markdown("#### IMC (bmi) según ACV")
            fig, ax = plt.subplots()
            boxplot_matplotlib_by_stroke(ax, df_viz, "bmi", "bmi")
            st.pyplot(fig)

    with col4:
        st.markdown("#### Frecuencia de ACV")
        fig, ax = plt.subplots()
        sns.countplot(data=df_viz, x="stroke_label", ax=ax)
        ax.set_xlabel("ACV")
        ax.set_ylabel("Conteo")
        st.pyplot(fig)

        st.write("Conteos:")
        st.write(df_viz["stroke_label"].value_counts())
        st.write("Porcentajes (%):")
        st.write(df_viz["stroke_label"].value_counts(normalize=True) * 100)

    st.markdown("---")
    if IMAGE3.exists():
        st.image(
            str(IMAGE3),
            caption="Variables y patrones asociados al ACV",
            use_column_width=True,
        )

# =========================================================
# FUNCIONES DE MODELO
# =========================================================
def preparar_datos(df_base: pd.DataFrame):
    X = df_base.drop(columns=["stroke"], errors="ignore")
    y = df_base["stroke"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    return X, y, preprocessor


def entrenar_random_forest(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )

    modelo = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf),
        ]
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    return modelo, X_test, y_test, y_pred, y_proba


# =========================================================
# TAB 3: MODELO
# =========================================================
with tab_model:
    st.header("🤖 Modelo predictivo — RandomForest (datos limpios)")

    st.markdown(
        """
        El modelo utilizado es un <b>RandomForestClassifier</b>, entrenado sobre el dataset
        limpio y balanceado. Este modelo captura relaciones no lineales y suele ofrecer
        un mejor rendimiento que modelos lineales simples.
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚀 Entrenar modelo RandomForest"):
        with st.spinner("Entrenando modelo con datos limpios..."):
            X, y, preprocessor = preparar_datos(df)
            modelo, X_test, y_test, y_pred, y_proba = entrenar_random_forest(
                X, y, preprocessor
            )
            st.session_state["rf_model"] = modelo

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        st.success("✅ Modelo entrenado correctamente con datos limpios.")
        st.balloons()  # 🎈 globos al entrenar

        c1_m, c2_m, c3_m = st.columns(3)
        c1_m.metric("Accuracy", f"{acc:.3f}")
        c2_m.metric("F1-score", f"{f1:.3f}")
        c3_m.metric("ROC-AUC", f"{roc:.3f}")

        st.markdown("### Matriz de confusión")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="rocket", ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        st.pyplot(fig)

        st.markdown("### Reporte de clasificación")
        st.text(classification_report(y_test, y_pred))

        if IMAGE4.exists():
            st.image(
                str(IMAGE4),
                caption="RandomForest — conjunto de árboles de decisión",
                use_column_width=True,
            )
    else:
        st.info("Pulsa el botón para entrenar el modelo con RandomForest.")

# =========================================================
# TAB 4: PREDICCIÓN
# =========================================================
with tab_pred:
    st.header("🎈 Predicción interactiva de riesgo de ACV")

    st.markdown(
        """
        Completa la información del paciente y el modelo entrenado con datos limpios
        estimará la probabilidad de que sufra un ACV.  
        Si el riesgo es bajo, 🎈 lanzaremos globos.
        """,
        unsafe_allow_html=True,
    )

    if "rf_model" not in st.session_state:
        st.warning(
            "Primero debes entrenar el modelo en la pestaña "
            "'🤖 Modelo RandomForest'."
        )
        st.stop()

    modelo = st.session_state["rf_model"]

    col_izq, col_der = st.columns(2)

    with col_izq:
        age = st.slider("Edad", 0, 100, 50)
        avg_glucose_level = st.slider(
            "Glucosa promedio",
            float(df["avg_glucose_level"].min()),
            float(df["avg_glucose_level"].max()),
            float(df["avg_glucose_level"].median()),
        )
        bmi = st.slider(
            "IMC (bmi)",
            float(df["bmi"].min()),
            float(df["bmi"].max()),
            float(df["bmi"].median()),
        )
        hypertension = st.selectbox("Hipertensión", options=[0, 1])
        heart_disease = st.selectbox("Enfermedad cardiaca", options=[0, 1])

    with col_der:
        gender = st.selectbox("Género", df["gender"].unique())
        ever_married = st.selectbox("¿Alguna vez casado/a?", df["ever_married"].unique())
        work_type = st.selectbox("Tipo de trabajo", df["work_type"].unique())
        residence_type = st.selectbox(
            "Tipo de residencia", df["Residence_type"].unique()
        )
        smoking_status = st.selectbox(
            "Estado de fumador", df["smoking_status"].unique()
        )

    input_df = pd.DataFrame(
        [
            {
                "age": age,
                "avg_glucose_level": avg_glucose_level,
                "bmi": bmi,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "gender": gender,
                "ever_married": ever_married,
                "work_type": work_type,
                "Residence_type": residence_type,
                "smoking_status": smoking_status,
            }
        ]
    )

    st.markdown("#### Vista previa de los datos del paciente")
    st.dataframe(input_df, use_container_width=True)

    if st.button("🔍 Calcular riesgo de ACV"):
        proba = modelo.predict_proba(input_df)[0, 1]
        pred = modelo.predict(input_df)[0]

        riesgo_pct = proba * 100

        c1_r, c2_r = st.columns([1, 2])
        with c1_r:
            st.metric("Probabilidad estimada", f"{riesgo_pct:.1f} %")
            st.write(
                "Clasificación:",
                "🟥 **Alto riesgo**" if pred == 1 else "🟩 **Bajo riesgo**",
            )

        with c2_r:
            if pred == 1:
                st.error(
                    "El modelo estima un **alto riesgo de ACV** para este paciente.\n\n"
                    "Este resultado NO sustituye una evaluación médica real, "
                    "pero sugiere revisar factores de riesgo con un profesional de la salud."
                )
            else:
                st.success(
                    "El modelo estima un **bajo riesgo de ACV** para este paciente "
                    "según las variables ingresadas."
                )
                st.balloons()  # 🎈 globos cuando el riesgo es bajo
