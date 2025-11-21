import matplotlib.pyplot as plt
import seaborn as sns

# Función para graficar la distribución de una columna
def graficar_distribucion(df, columna):
    """
    Muestra un histograma de la distribución de la columna dada.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[columna], bins=20, kde=True)
    plt.title(f"Distribución de {columna}", fontsize=16)
    plt.xlabel(columna, fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.show()

# Función para graficar la distribución de clases de 'stroke'
def graficar_distribucion_clases(df):
    """
    Muestra un gráfico de barras con la distribución de las clases de 'stroke'.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='stroke', data=df, palette='viridis')
    plt.title("Distribución de la Variable 'Stroke'", fontsize=16)
    plt.xlabel("Stroke", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.show()

# Función para graficar la importancia de las características (para Random Forest)
def graficar_importancia_caracteristicas(importances, features):
    """
    Muestra un gráfico de barras con la importancia de las características de un modelo.
    """
    indices = importances.argsort()[::-1]
    features_sorted = features[indices]
    importances_sorted = importances[indices]

    plt.figure(figsize=(12, 7))
    sns.barplot(x=importances_sorted, y=features_sorted, palette='viridis')
    plt.title('Importancia de las Características', fontsize=16)
    plt.xlabel('Importancia', fontsize=12)
    plt.ylabel('Características', fontsize=12)
    plt.tight_layout()
    plt.show()

# Función para graficar una matriz de confusión
def graficar_matriz_confusion(cm, labels):
    """
    Muestra una matriz de confusión utilizando heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusión', fontsize=16)
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Realidad', fontsize=12)
    plt.show()
