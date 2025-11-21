from sklearn.naive_bayes import GaussianNB

def entrenar_modelo_nb(X_train, y_train):
    modelo = GaussianNB()
    modelo.fit(X_train, y_train)
    return modelo
