import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. LECTURA Y LIMPIEZA DE DATOS


# Leer los datos saltando las 2 primeras líneas
df = pd.read_csv('data3.csv', skiprows=2)

# Renombramos a 'Fecha' (porque son meses, no semanas) y usamos los términos de la imagen
df.columns = ['Fecha', 'Samsung', 'Apple']

# Limpiar datos (eliminar el carácter '<' si aparece y convertir a numérico)
df['Samsung'] = pd.to_numeric(df['Samsung'].astype(str).str.replace('<', ''), errors='coerce').fillna(0)
df['Apple'] = pd.to_numeric(df['Apple'].astype(str).str.replace('<', ''), errors='coerce').fillna(0)

# 2. ANÁLISIS DE CORRELACIÓN

correlacion = df['Samsung'].corr(df['Apple'])

print("=====================================================")
print(f'Correlación entre Samsung y Apple: {correlacion:.4f}')
print("=====================================================")

# 3. GRÁFICA DE TENDENCIAS

plt.figure(figsize=(12, 6))

# Usamos los colores exactos de la interfaz de Google Trends (Azul y Rojo)
plt.plot(df['Fecha'], df['Samsung'], label='Samsung (Categoría)', color='#4285F4', linewidth=2)
plt.plot(df['Fecha'], df['Apple'], label='Apple (Categoría)', color='#EA4335', linewidth=2)

# Como son datos mensuales de 16 años, saltamos de 12 en 12 para mostrar 1 año por etiqueta
plt.xticks(df['Fecha'][::12], rotation=45)

# Actualizamos el título con el contexto real
plt.title('Tendencias de búsqueda Mundial: Samsung vs Apple (Jul 2010 - Jul 2026)')
plt.legend()
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5) 

# Mostrar la gráfica
plt.show()

# 4. PREDICCIÓN CON REGRESIÓN LINEAL

# Preparar los datos
X = df[['Samsung']] # Feature (2D)
y = df['Apple'] # Target (1D)

# Entrenar el modelo
modelo = LinearRegression() 
modelo.fit(X, y) 

# Predicción de prueba
nivel_samsung_hoy = pd.DataFrame({'Samsung': [95]})
prediccion = modelo.predict(nivel_samsung_hoy)

print("=====================================================")
print(f'Si el interés de "Samsung" es 95, la predicción para "Apple" es: {prediccion[0]:.2f}')
print("=====================================================")