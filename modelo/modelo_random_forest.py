from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def entrenar_modelo(X_train, y_train):
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

def optimizar_modelo(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
