from sklearn.tree import DecisionTreeClassifier

def entrenar_modelo_tree(X_train, y_train):
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    return modelo
