from imblearn.over_sampling import SMOTE

def balancear_datos(X, y):
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res
