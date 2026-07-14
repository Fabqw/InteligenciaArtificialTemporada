import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Leer los datos saltando las dos primeras lineras  
df = pd.read_csv('data1.csv', skiprows=2)

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
plt.plot(df['Semana'], df['Mundial'], label='Interes Mundial', color='blue', marker='o')
plt.plot(df['Semana'], df['Televisor'], label='Interes Televisor', color='red', marker='o')
plt.title('Tendencia de Mundial y Televisor a lo largo de las Semanas')
plt.xlabel('Semana')
plt.ylabel('Cantidad')



# 
plt.xticks(df['Semana'][::26], rotation=45)
plt.title('Impacto del mudial en la busqueda de Televisores en los ultimos 5 años')
plt.legend()
plt.tight_layout()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()