import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ============================================================
# 1. CARGA DE DATOS
# ============================================================
df = pd.read_csv('ejercicios/archive/Marriage_Divorce_DB.csv')

print("=" * 60)
print("PREDICCIÓN DE DIVORCIO - NAIVE BAYES")
print("=" * 60)
print(f"Total de muestras: {df.shape[0]}")
print(f"Características: {df.shape[1] - 1}")

# ============================================================
# 2. DISCRETIZAR OBJETIVO EN 3 CATEGORÍAS
# ============================================================
def categorizar(prob):
    if prob < 1.66:
        return 0  # Bajo
    elif prob < 2.33:
        return 1  # Medio
    else:
        return 2  # Alto

df['Categoria'] = df['Divorce Probability'].apply(categorizar)

print(f"\nDistribución de categorías:")
for cat, label in [(0, 'Bajo'), (1, 'Medio'), (2, 'Alto')]:
    n = (df['Categoria'] == cat).sum()
    subset = df[df['Categoria'] == cat]['Divorce Probability']
    print(f"  {label}: {n} muestras  [{subset.min():.4f} - {subset.max():.4f}]")

# ============================================================
# 3. SEPARAR CARACTERÍSTICAS Y OBJETIVO
# ============================================================
X = df.drop(['Divorce Probability', 'Categoria'], axis=1)
y = df['Categoria']

# ============================================================
# 4. DIVIDIR EN ENTRENAMIENTO Y PRUEBA
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nEntrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")

# ============================================================
# 5. ENTRENAR MODELO NAIVE BAYES
# ============================================================
modelo = GaussianNB()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'=' * 60}")
print(f"RESULTADOS")
print(f"{'=' * 60}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\nReporte de clasificación:")
print(classification_report(y_test, y_pred, digits=4,
                            target_names=['Bajo', 'Medio', 'Alto']))

# ============================================================
# 6. MATRIZ DE CONFUSIÓN
# ============================================================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Bajo', 'Medio', 'Alto'])
disp.plot(cmap='Blues', values_format='d')
plt.title(f'Matriz de Confusión - Naive Bayes (Accuracy: {accuracy:.2%})')
plt.show()

# ============================================================
# 7. PREDICCIÓN DE EJEMPLO
# ============================================================
print(f"\n{'=' * 60}")
print(f"PREDICCIÓN DE EJEMPLO")
print(f"{'=' * 60}")

indice = 0
ejemplo = X_test.iloc[[indice]]
real = y_test.iloc[indice]
real_prob = df.loc[X_test.index[indice], 'Divorce Probability']
etiquetas = ['Bajo', 'Medio', 'Alto']

pred = modelo.predict(ejemplo)[0]

print(f"\nCaracterísticas del ejemplo:")
for col in X.columns[:5]:
    print(f"  {col}: {ejemplo[col].values[0]:.4f}")
print(f"  ... y {len(X.columns) - 5} más")

print(f"\nCategoría REAL:      {etiquetas[real]} (probabilidad: {real_prob:.4f})")
print(f"Categoría PREDICHA:  {etiquetas[pred]}")
print(f"¿Acertó?            {'✓ Sí' if pred == real else '✗ No'}")
