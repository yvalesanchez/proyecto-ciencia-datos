# Ejecutar el preprocesamiento
from src.procesamiento_datos import cargar_datos, limpiar_datos, balancear_datos, guardar_datos_limpios

# Cargar el archivo de datos original
df = cargar_datos('data/healthcare-dataset-stroke-data.csv')

# Limpiar los datos (reemplazar valores, convertir columnas, eliminar outliers, etc.)
df_limpio = limpiar_datos(df)

# Balancear las clases usando SMOTE
df_limpio = balancear_datos(df_limpio)

# Guardar el dataset limpio y balanceado en un nuevo archivo CSV
guardar_datos_limpios(df_limpio, 'data/healthcare-dataset-stroke-data-limpio.csv')
