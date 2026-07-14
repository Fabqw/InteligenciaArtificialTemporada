import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Leer los datos saltando las dos primeras lineras  
df = pd.read_csv('data2.csv', skiprows=2)

# Renombras columnas para facilitar el codifo
df.columns = ['Semana', 'Mundial', 'Televisor']

# Limpiar datos( eliminar los caracteres raros como <1 que a veces eexporta google) 
df['Mundial'] = pd.to_numeric(df['Mundial'].astype(str).str.replace('<', ''), errors='coerce').fillna(0)
df['Televisor'] = pd.to_numeric(df['Televisor'].astype(str).str.replace('<', ''), errors='coerce').fillna(0)

# Calcualr las correlaciones matematicas
correlation = df['Mundial'].corr(df['Televisor'])
print("==============================================================")
print(f"Correlación entre Mundial y Televisor: {correlation:.2f}")
print("==============================================================")

# graficar ambas tendecias
plt.figure(figsize=(10, 6))
plt.plot(df['Semana'], df['Mundial'], label='Interes Mundial', color='blue')
plt.plot(df['Semana'], df['Televisor'], label='Interes Televisor', color='red')
plt.title('Tendencia de Mundial y Televisor a lo largo de las Semanas')
plt.xlabel('Semana')
plt.ylabel('Cantidad')
plt.legend()
plt.grid()
plt.show()


# Preparar los datos para el modelo de regresión lineal
x = df[['Mundial']]
y = df['Televisor']

# inicializar y entrenar el modelo de regresión lineal
modelo_m1 = LinearRegression()
modelo_m1.fit(x, y)

# hacer una predficcion interactiva
# supongamos que por la final del mundial la busqueda de "munidial" llegara a un pico de 95%
nivel_mundial = [[95]]
prediccion_televisores = modelo_m1.predict(nivel_mundial)
print(f"Si el interes por el mundial es del {nivel_mundial[0][0]:.2f}%, se espera un interes del {prediccion_televisores[0]:.2f}% en los televisores.")
