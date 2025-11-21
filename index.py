import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from modelo.modelo_random_forest import optimizar_modelo
from modelo.modelo_logistic_regression import entrenar_modelo_logistic
from modelo.modelo_decision_tree import entrenar_modelo_tree
from modelo.modelo_svc import entrenar_modelo_svc
from modelo.modelo_naive_bayes import entrenar_modelo_nb
from modelo.smote_balanceo import balancear_datos
from PIL import Image

# Título y descripción
st.title('Análisis de Datos de Ataques Cerebrovasculares')
st.write("""
    Este proyecto analiza un conjunto de datos relacionados con ataques cerebrovasculares. 
    Se implementan modelos de Machine Learning para predecir la probabilidad de un ataque 
    en función de diversos factores.
""")

# Mostrar la imagen
img = Image.open("acv.jpg")
st.image(img, caption="Ataque Cerebrovascular")

# Cargar los datos
df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

# Muestra las primeras filas
if st.checkbox("Mostrar primeros datos", False):
    st.write(df.head())

# Preprocesamiento
df['stroke'] = df['stroke'].replace({0: 'Sin ataque', 1: 'Ataque'})
df['stroke'] = df['stroke'].astype('category')
X = df.drop(columns=['stroke'])
y = df['stroke'].map({'Sin ataque': 0, 'Ataque': 1})

# Balanceo de clases con SMOTE
X_res, y_res = balancear_datos(X, y)

# Dividir en datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=42)

# Entrenar modelos y mostrar resultados
modelos = {
    "Random Forest": optimizar_modelo(X_train, y_train),
    "Regresión Logística": entrenar_modelo_logistic(X_train, y_train),
    "Árbol de Decisión": entrenar_modelo_tree(X_train, y_train),
    "SVC": entrenar_modelo_svc(X_train, y_train),
    "Naive Bayes": entrenar_modelo_nb(X_train, y_train)
}

for nombre_modelo, modelo in modelos.items():
    st.subheader(f"Modelo: {nombre_modelo}")
    y_pred = modelo.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(f"Informe de clasificación para {nombre_modelo}:")
    st.write(report)

    # Mostrar la matriz de confusión
    st.subheader(f"Matriz de Confusión de {nombre_modelo}")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Sin ataque", "Ataque"],
                yticklabels=["Sin ataque", "Ataque"])
    st.pyplot()

    # Mostrar importancia de características para Random Forest
    if nombre_modelo == "Random Forest":
        importances = modelo.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X.columns[indices]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=features)
        st.pyplot()
