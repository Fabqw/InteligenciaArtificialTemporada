import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Cargar el dataset
try:
    df = pd.read_csv('./dataset_tiempo_graduacion.csv')
    print("Dataset cargado correctamente.")
    print("Primeras 5 filas del dataset:")
    print(df.head())
    print("Información del dataset:")
    df.info()
except FileNotFoundError:
    print("Error: El archivo 'dataset_tiempo_graduacion.csv' no fue encontrado. Asegúrate de que esté en la ubicación correcta.")
    exit()

if 'tiempo_graduacion' in df.columns:
    target_column = 'tiempo_graduacion'
elif df.columns[-1] not in ['Unnamed: 0'] and len(df.columns) > 1:
    target_column = df.columns[-1]
    print(f"El objetivo es la última: '{target_column}'")
else:
    print("No se pudo identificar automáticamente la columna objetivo. Por favor, especifica el nombre de la columna a predecir.")
    print("Columnas disponibles:", df.columns.tolist())
    exit()

# Dividir el dataset en características (X) y variable objetivo (y)
X = df.drop(columns=[target_column])
y = df[target_column]

# Manejar columnas no numéricas (si las hay, por simplicidad, las eliminaremos o codificaremos)
# Para este ejemplo, eliminaremos las columnas no numéricas que no sean la columna objetivo.
X = X.select_dtypes(include=np.number)

# Verificar si X está vacío después de eliminar no numéricas
if X.empty:
    print("Error: No hay características numéricas para el modelo después de la limpieza. Revisa tu dataset.")
    exit()

# Eliminar filas con valores faltantes (otra opción sería imputarlos)
X.dropna(inplace=True)
y = y[X.index] # Asegurar que y coincida con las filas de X

if X.empty:
    print("Error: No quedan datos después de eliminar valores faltantes.")
    exit()

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el modelo de Regresión Lineal
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModelo de Regresión Lineal entrenado con éxito.")

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nError Cuadrático Medio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.2f}")

# Mostrar coeficientes del modelo
print("\nCoeficientes del modelo:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.2f}")
print(f"  Intercepto: {model.intercept_:.2f}")

# Ejemplo de predicción para una nueva instancia (usando los primeros valores del conjunto de prueba)
print("\nEjemplo de predicción para una nueva instancia:")
sample_data = X_test.iloc[0].values.reshape(1, -1)
predicted_graduation_time = model.predict(sample_data)

print(f"Valores de entrada para la predicción: {X_test.iloc[0].tolist()}")
print(f"Tiempo de graduación predicho: {predicted_graduation_time[0]:.2f}")
print(f"Tiempo de graduación real (para comparación): {y_test.iloc[0]:.2f}")