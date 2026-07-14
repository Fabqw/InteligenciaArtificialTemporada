import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Leer los datos saltando las dos primeras lineras  
df = pd.read_csv('data1.csv', skiprows=2)

# Renombras columnas para facilitar el codifo
df.columns = ['Semana', 'Mundial', 'Televisor']

# Limpiar datos( eliminar los caracteres raros como <1 que a veces eexporta google) 
df['Mundial'] = pd.to_numeric(df['Mundial'].astype(str).str.replace('<'), errors='coerce').fillna(0)
df['Televisor'] = pd.to_numeric(df['Televisor'].astype(str).str.replace('<'), errors='coerce').fillna(0)

# Calcualr las correlaciones matematicas
correlation = df['Mundial'].corr(df['Televisor'])
print("==============================================================")
print(f'Correlación entre Mundial y Televisor: {correlation}')