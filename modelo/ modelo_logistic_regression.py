from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def entrenar_modelo_logistic(X_train, y_train):
    """
    Entrena un modelo de regresión logística.
    """
    modelo = LogisticRegression(max_iter=1000, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo
