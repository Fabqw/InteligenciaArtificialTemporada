import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ============================================================
# 1. CARGA DE DATOS
# ============================================================
df = pd.read_csv('ejercicios/divorce.csv')

print("=" * 60)
print("DATASET DE DIVORCIO")
print("=" * 60)
print(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"\nColumnas: {list(df.columns)}")
print(f"\nDistribución de la variable objetivo (Atr10):")
print(df['Atr10'].value_counts().sort_index().to_string())
print()

# ============================================================
# 2. SEPARAR CARACTERÍSTICAS Y OBJETIVO
# ============================================================
X = df.drop('Atr10', axis=1)
y = df['Atr10']

# ============================================================
# 3. DIVIDIR EN ENTRENAMIENTO Y PRUEBA
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")
print()

# ============================================================
# 4. ENTRENAR MODELOS
# ============================================================
modelos = {
    'Naive Bayes (Gaussian)': GaussianNB(),
    'Árbol de Decisión': DecisionTreeClassifier(max_depth=5, random_state=42)
}

resultados = {}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    resultados[nombre] = {
        'modelo': modelo,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'reporte': classification_report(y_test, y_pred, digits=4)
    }

    print(f"\n{'=' * 60}")
    print(f"  {nombre}")
    print(f"{'=' * 60}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\n  Reporte de clasificación:")
    print(resultados[nombre]['reporte'])

# ============================================================
# 5. COMPARACIÓN VISUAL
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Predicción de Divorcio - Comparación de Modelos', fontsize=16, fontweight='bold')

for idx, (nombre, res) in enumerate(resultados.items()):
    # Matriz de confusión
    cm = confusion_matrix(y_test, res['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=sorted(y.unique()))
    disp.plot(ax=axes[idx, 0], cmap='Blues', values_format='d')
    axes[idx, 0].set_title(f'{nombre} - Matriz de Confusión')

    # Comparación real vs predicho (barras)
    real_counts = y_test.value_counts().sort_index()
    pred_counts = pd.Series(res['y_pred']).value_counts().sort_index()

    clases = sorted(set(y_test.unique()) | set(res['y_pred']))
    real_vals = [real_counts.get(c, 0) for c in clases]
    pred_vals = [pred_counts.get(c, 0) for c in clases]

    x = np.arange(len(clases))
    width = 0.35
    axes[idx, 1].bar(x - width/2, real_vals, width, label='Real', color='steelblue')
    axes[idx, 1].bar(x + width/2, pred_vals, width, label='Predicho', color='coral')
    axes[idx, 1].set_xlabel('Clase (Atr10)')
    axes[idx, 1].set_ylabel('Cantidad')
    axes[idx, 1].set_title(f'{nombre} - Real vs Predicho')
    axes[idx, 1].set_xticks(x)
    axes[idx, 1].set_xticklabels(clases)
    axes[idx, 1].legend()

# Accuracy comparison
nombres = list(resultados.keys())
accs = [resultados[n]['accuracy'] for n in nombres]
bars = axes[0, 2].bar(nombres, accs, color=['#2E86AB', '#A23B72'])
axes[0, 2].set_ylabel('Accuracy')
axes[0, 2].set_title('Comparación de Accuracy')
axes[0, 2].set_ylim(0, 1)
for bar, acc in zip(bars, accs):
    axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.4f}', ha='center', fontweight='bold')

# Árbol de decisión visual
plot_tree(resultados['Árbol de Decisión']['modelo'],
          feature_names=list(X.columns),
          class_names=[str(c) for c in sorted(y.unique())],
          filled=True, rounded=True,
          ax=axes[1, 2], fontsize=8)
axes[1, 2].set_title('Árbol de Decisión')

plt.tight_layout()
plt.show(block=False)
plt.pause(2)
plt.close('all')

# ============================================================
# 6. PREDICCIÓN DE EJEMPLO
# ============================================================
print(f"\n{'=' * 60}")
print("  PREDICCIÓN DE EJEMPLO")
print(f"{'=' * 60}")

ejemplo = pd.DataFrame([{
    'Atr1': 2, 'Atr2': 2, 'Atr3': 3, 'Atr4': 1,
    'Atr5': 0, 'Atr6': 0, 'Atr7': 0, 'Atr8': 0, 'Atr9': 0
}])

print(f"\nDatos de entrada: {ejemplo.to_dict('records')[0]}")
for nombre, res in resultados.items():
    pred = res['modelo'].predict(ejemplo)[0]
    print(f"  {nombre:25s} -> Predicción: {pred}")
