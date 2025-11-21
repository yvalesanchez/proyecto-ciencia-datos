import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Función para cargar los datos
def cargar_datos(ruta):
    """
    Carga un archivo CSV y retorna un dataframe de pandas.
    """
    return pd.read_csv(ruta)

# Función para limpiar los datos
def limpiar_datos(df):
    """
    Realiza la limpieza y transformación de datos, como reemplazo de valores,
    conversión de columnas categóricas y eliminación de outliers.
    """
    # Reemplazo de valores categóricos
    df['stroke'] = df['stroke'].replace({0: 'Sin ataque', 1: 'Ataque'})
    df['hypertension'] = df['hypertension'].replace({0: 'No', 1: 'Sí'})
    df['heart_disease'] = df['heart_disease'].replace({0: 'No', 1: 'Sí'})
    df['smoking_status'] = df['smoking_status'].replace({
        'Unknown': 'Desconocido',
        'formerly smoked': 'Fumó anteriormente',
        'never smoked': 'Nunca fumó',
        'smokes': 'Fuma'
    })
    df['work_type'] = df['work_type'].replace({
        'Govt_job': 'Empleado público',
        'Never_worked': 'Nunca trabajó',
        'Private': 'Empleado privado',
        'Self-employed': 'Independiente',
        'children': 'Niño/a'
    })
    df['gender'] = df['gender'].replace({
        'Male': 'Hombre',
        'Female': 'Mujer',
        'Other': 'Otro'
    })

    # Convertir columnas categóricas a tipo 'category'
    categoricas = df.select_dtypes(include=['object']).columns
    for col in categoricas:
        df[col] = df[col].astype('category')

    # Rellenar valores faltantes en la columna 'bmi' con la mediana
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

    # Eliminar columnas innecesarias
    df = df.drop(columns='id', errors='ignore')

    # Eliminar outliers utilizando el rango intercuartílico (IQR) en las columnas 'avg_glucose_level' y 'bmi'
    for col in ['avg_glucose_level', 'bmi']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        LimiteInferior = Q1 - 1.5 * IQR
        LimiteSuperior = Q3 + 1.5 * IQR
        df = df[(df[col] >= LimiteInferior) & (df[col] <= LimiteSuperior)]

    return df

# Función para balancear los datos con SMOTE
def balancear_datos(df):
    """
    Aplica SMOTE (Synthetic Minority Over-sampling Technique) para balancear las clases.
    """
    X = df.drop(columns=['stroke'])
    y = df['stroke'].map({'Sin ataque': 0, 'Ataque': 1})

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    df_limpio = pd.concat([X_res, y_res], axis=1)
    df_limpio['stroke'] = df_limpio['stroke'].map({0: 'Sin ataque', 1: 'Ataque'})  # Mapear las clases a texto

    return df_limpio

# Función para guardar los datos limpios
def guardar_datos_limpios(df_limpio, ruta):
    """
    Guarda el DataFrame limpio en un archivo CSV.
    """
    df_limpio.to_csv(ruta, index=False)
    print(f"Archivo CSV de datos limpios guardado como {ruta}")
