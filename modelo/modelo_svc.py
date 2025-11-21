from sklearn.svm import SVC

def entrenar_modelo_svc(X_train, y_train):
    modelo = SVC(probability=True, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo
