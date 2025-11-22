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

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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
    page_title="Análisis de ACV (Stroke) — Modelos comparados",
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

# Dataset limpio si existe, si no, dataset original
CLEAN_PATH = BASE_DIR.parent / "notebooks" / "data" / "stroke_clean.csv"
RAW_PATH = BASE_DIR.parent / "data" / "healthcare-dataset-stroke-data.csv"

IMAGE1 = BASE_DIR / "acv.jpg"
IMAGE2 = BASE_DIR / "acv2.jpg"
IMAGE3 = BASE_DIR / "acv3.jpg"
IMAGE4 = BASE_DIR / "acv4.jpg"


# =========================================================
# CARGA Y LIMPIEZA DE DATOS
# =========================================================
@st.cache_data
def load_clean_data():
    """
    Devuelve:
      - df limpio listo para modelar
      - texto explicando de dónde se cargó
    """
    if CLEAN_PATH.exists():
        df = pd.read_csv(CLEAN_PATH)
        source = f"Archivo limpio: {CLEAN_PATH.name} (notebooks/data)"
    else:
        if not RAW_PATH.exists():
            raise FileNotFoundError(
                f"No se encontró ni {CLEAN_PATH} ni {RAW_PATH}"
            )
        df = pd.read_csv(RAW_PATH)
        source = f"Archivo crudo: {RAW_PATH.name} (data) — limpieza aplicada en la app"

        # === LIMPIEZA BÁSICA ===
        df = df.drop_duplicates()

        for col in ["age", "avg_glucose_level", "bmi"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "age" in df.columns:
            df = df[df["age"] >= 0]

        if "bmi" in df.columns:
            df["bmi"] = df["bmi"].fillna(df["bmi"].median())

        df = df.dropna(subset=["stroke"])

        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

    # === Mapeo de la variable objetivo a 0 / 1 (por si viene como texto) ===
    if not np.issubdtype(df["stroke"].dtype, np.number):
        df["stroke"] = df["stroke"].replace(
            {
                "Sin ataque": 0,
                "Ataque": 1,
                "No": 0,
                "Yes": 1,
                "NO": 0,
                "SI": 1,
                "Si": 1,
                "0": 0,
                "1": 1,
            }
        )

    df["stroke"] = df["stroke"].astype(int)

    # 🔥 TODAS LAS CATEGÓRICAS A STRING PURO
    cat_cols = df.select_dtypes(exclude=["int64", "float64", "bool"]).columns
    df[cat_cols] = df[cat_cols].astype(str).fillna("Desconocido")

    return df, source


try:
    df, data_source = load_clean_data()
except Exception as e:
    st.error("No se pudieron cargar los datos:")
    st.exception(e)
    st.stop()

df_viz = df.copy()
df_viz["stroke_label"] = df_viz["stroke"].map({0: "Sin ataque", 1: "Ataque"})

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

        Utilizando datos clínicos limpios y, en el estudio original,
        balanceados con SMOTE para analizar el riesgo de ACV.
        """
    )
    st.markdown("---")
    st.markdown(
        f"📁 Datos cargados desde:<br><code>{data_source}</code>",
        unsafe_allow_html=True,
    )
    st.caption("Ciencia de Datos en Salud • Streamlit • RandomForest y otros modelos")

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
        Este panel interactivo resume el proceso de análisis y modelado realizado
        sobre un conjunto de datos de pacientes, utilizando un dataset
        <b>previamente limpiado</b> (y balanceado con SMOTE en el cuadernillo original).
        Aquí comparamos varios modelos de clasificación y, finalmente, nos enfocamos
        en el <b>RandomForest</b>, que resultó ser el más robusto.
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
            <li>Comparar varios modelos de Machine Learning.</li>
            <li>Seleccionar y analizar con detalle <b>RandomForest</b> como modelo final.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <span class="pill">Regresión Logística</span>
        <span class="pill">KNN</span>
        <span class="pill">SVM</span>
        <span class="pill">Árbol de decisión</span>
        <span class="pill">RandomForest</span>
        """,
        unsafe_allow_html=True,
    )

with c2:
    if IMAGE1.exists():
        st.image(str(IMAGE1), caption="Ilustración de ACV", use_column_width=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**📊 Resumen del dataset utilizado**")
    st.write(f"Filas: **{df.shape[0]}**")
    st.write(f"Columnas: **{df.shape[1]}**")
    st.write("Variable objetivo: **stroke** (0 = Sin ataque, 1 = Ataque)")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# TABS PRINCIPALES
# =========================================================
tab_data, tab_eda, tab_modelos, tab_pred = st.tabs(
    [
        "📄 Datos limpios",
        "📊 Análisis Exploratorio",
        "🤖 Modelos comparados",
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
    st.header("📊 Análisis Exploratorio")

    st.markdown(
        """
        Se muestran distribuciones de variables numéricas por presencia de ACV y la
        frecuencia total de casos.
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Distribución de edad por ACV")
        fig, ax = plt.subplots()
        sns.histplot(
            data=df_viz,
            x="age",
            hue="stroke_label",
            multiple="stack",
            bins=30,
            ax=ax,
        )
        ax.set_xlabel("Edad")
        ax.set_ylabel("Conteo")
        st.pyplot(fig)

    with col2:
        st.markdown("#### Distribución de glucosa promedio por ACV")
        fig, ax = plt.subplots()
        sns.histplot(
            data=df_viz,
            x="avg_glucose_level",
            hue="stroke_label",
            multiple="stack",
            bins=30,
            ax=ax,
        )
        ax.set_xlabel("avg_glucose_level")
        ax.set_ylabel("Conteo")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        if "bmi" in df_viz.columns:
            st.markdown("#### Distribución de IMC (bmi) por ACV")
            fig, ax = plt.subplots()
            sns.histplot(
                data=df_viz,
                x="bmi",
                hue="stroke_label",
                multiple="stack",
                bins=30,
                ax=ax,
            )
            ax.set_xlabel("bmi")
            ax.set_ylabel("Conteo")
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
    X = df_base.drop(columns=["stroke", "id"], errors="ignore")
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


def entrenar_varios_modelos(X, y, preprocessor):
    """
    Entrena varios modelos con el mismo split y devuelve
    un diccionario con métricas y objetos entrenados.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    modelos = {
        "Regresión Logística": LogisticRegression(max_iter=1000),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
        "SVM (RBF)": SVC(kernel="rbf", probability=True),
        "Árbol de Decisión": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
    }

    resultados = {}

    for nombre, clf in modelos.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        resultados[nombre] = {
            "modelo": pipe,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    return resultados

# =========================================================
# TAB 3: MODELOS COMPARADOS
# =========================================================
with tab_modelos:
    st.header("🤖 Comparación de modelos de clasificación")

    st.markdown(
        """
        En el cuadernillo se probaron varios modelos de Machine Learning para predecir la
        variable <b>stroke</b>. Aquí comparamos:
        <ul>
            <li><b>Regresión Logística</b></li>
            <li><b>KNN</b></li>
            <li><b>SVM</b></li>
            <li><b>Árbol de Decisión</b></li>
            <li><b>RandomForest</b></li>
        </ul>
        Finalmente, el modelo que se toma como <b>modelo final</b> es
        <b>RandomForest</b>, por su mejor equilibrio general.
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚀 Entrenar y comparar todos los modelos"):
        with st.spinner("Entrenando modelos..."):
            X, y, preprocessor = preparar_datos(df)
            resultados = entrenar_varios_modelos(X, y, preprocessor)

        st.session_state["model_results"] = resultados
        st.session_state["rf_model"] = resultados["RandomForest"]["modelo"]

        # === Tabla de métricas ===
        resumen = []
        for nombre, res in resultados.items():
            resumen.append(
                {
                    "Modelo": nombre,
                    "Accuracy": res["accuracy"],
                    "F1-score": res["f1"],
                    "ROC-AUC": res["roc_auc"],
                }
            )
        resumen_df = pd.DataFrame(resumen).set_index("Modelo")

        st.markdown("### 📈 Resultados globales de todos los modelos")
        st.dataframe(resumen_df.style.format("{:.3f}"), use_container_width=True)

        mejor_nombre = max(resultados, key=lambda m: resultados[m]["roc_auc"])
        st.success(
            f"En esta ejecución, el mejor ROC-AUC lo obtiene: **{mejor_nombre}**. "
            "En el proyecto tomamos **RandomForest** como modelo final."
        )

        # === Sección específica para RandomForest ===
        rf_res = resultados["RandomForest"]

        st.markdown("### 🔍 Análisis detallado del modelo final: RandomForest")

        c1_m, c2_m, c3_m = st.columns(3)
        c1_m.metric("Accuracy RF", f"{rf_res['accuracy']:.3f}")
        c2_m.metric("F1-score RF", f"{rf_res['f1']:.3f}")
        c3_m.metric("ROC-AUC RF", f"{rf_res['roc_auc']:.3f}")

        st.markdown("#### Matriz de confusión — RandomForest")
        fig, ax = plt.subplots()
        cm = confusion_matrix(rf_res["y_test"], rf_res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="rocket", ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        st.pyplot(fig)

        st.markdown("#### Reporte de clasificación — RandomForest")
        st.text(classification_report(rf_res["y_test"], rf_res["y_pred"]))

        if IMAGE4.exists():
            st.image(
                str(IMAGE4),
                caption="RandomForest — modelo final seleccionado",
                use_column_width=True,
            )

    else:
        st.info("Pulsa el botón para entrenar y comparar todos los modelos.")

# =========================================================
# TAB 4: PREDICCIÓN (CON RANDOMFOREST)
# =========================================================
with tab_pred:
    st.header("🎈 Predicción interactiva de riesgo de ACV (RandomForest)")

    st.markdown(
        """
        El modelo utilizado para la predicción final es <b>RandomForest</b>, 
        el mejor modelo según el estudio.  
        Completa la información del paciente y estima su probabilidad de ACV.
        """,
        unsafe_allow_html=True,
    )

    if "rf_model" not in st.session_state:
        st.warning(
            "Primero debes entrenar los modelos en la pestaña "
            "'🤖 Modelos comparados'."
        )
        st.stop()

    modelo_rf = st.session_state["rf_model"]

    # -------- Formularios de entrada --------
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
            float(df["bmi"].min()) if "bmi" in df.columns else 10.0,
            float(df["bmi"].max()) if "bmi" in df.columns else 60.0,
            float(df["bmi"].median()) if "bmi" in df.columns else 25.0,
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

    # -------- Construimos fila de predicción con MISMAS columnas que X --------
    X_base, _, _ = preparar_datos(df)
    feature_cols = X_base.columns

    # valores por defecto (mediana para numéricos, moda para categóricos)
    fila = {}
    for col in feature_cols:
        serie = df[col]
        if np.issubdtype(serie.dtype, np.number):
            fila[col] = float(serie.median())
        else:
            fila[col] = serie.mode().iloc[0]

    # sobrescribimos con lo que ingresó el usuario
    overrides = {
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
    for k, v in overrides.items():
        if k in fila:
            fila[k] = v

    input_df = pd.DataFrame([fila], columns=feature_cols)

    st.markdown("#### Vista previa de los datos del paciente (formato modelo)")
    st.dataframe(input_df, use_container_width=True)

    if st.button("🔍 Calcular riesgo de ACV"):
        proba = modelo_rf.predict_proba(input_df)[0, 1]
        pred = modelo_rf.predict(input_df)[0]

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
                    "El modelo final (RandomForest) estima un **ALTO RIESGO de ACV** "
                    "para este paciente.\n\n"
                    "Este resultado NO sustituye una evaluación médica real, "
                    "pero sugiere revisar factores de riesgo con un profesional de la salud."
                )
            else:
                st.success(
                    "El modelo final (RandomForest) estima un **BAJO RIESGO de ACV** "
                    "para este paciente según las variables ingresadas."
                )
                st.balloons()  # 🎈 globos cuando el riesgo es bajo
