import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ============================================================
# 1. CARGA Y EXPLORACIÓN
# ============================================================
df = pd.read_csv('ejercicios/archive/Marriage_Divorce_DB.<<<<<<<<<<<<<<<<<<<<<<<csv')

print("=" * 70)
print("DATASET MARRIAGE DIVORCE DB - CLASIFICACIÓN")
print("=" * 70)
print(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")

# Discretizar la probabilidad de divorcio en 3 categorías
def categorizar_divorcio(prob):
    if prob < 1.66:
        return 0  # Bajo
    elif prob < 2.33:
        return 1  # Medio
    else:
        return 2  # Alto

df['DivorceCategory'] = df['Divorce Probability'].apply(categorizar_divorcio)

print(f"\nDistribución de categorías:")
print(f"  0 - Bajo : {(df['DivorceCategory'] == 0).sum()}")
print(f"  1 - Medio: {(df['DivorceCategory'] == 1).sum()}")
print(f"  2 - Alto : {(df['DivorceCategory'] == 2).sum()}")

# Mostrar rangos
for cat, label in [(0, 'Bajo'), (1, 'Medio'), (2, 'Alto')]:
    subset = df[df['DivorceCategory'] == cat]['Divorce Probability']
    print(f"  {label}: [{subset.min():.4f} - {subset.max():.4f}]")

# ============================================================
# 2. SEPARAR CARACTERÍSTICAS Y OBJETIVO
# ============================================================
X = df.drop(['Divorce Probability', 'DivorceCategory'], axis=1)
y = df['DivorceCategory']

# ============================================================
# 3. DIVIDIR EN ENTRENAMIENTO Y PRUEBA
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nEntrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")

# ============================================================
# 4. ENTRENAR MODELOS
# ============================================================
modelos = {
    'Árbol de Decisión (max_depth=4)': DecisionTreeClassifier(max_depth=4, random_state=42),
    'Árbol de Decisión (max_depth=6)': DecisionTreeClassifier(max_depth=6, random_state=42),
    'Naive Bayes (Gaussian)': GaussianNB(),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
}

resultados = {}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Validación cruzada
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')

    resultados[nombre] = {
        'modelo': modelo,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'reporte': classification_report(y_test, y_pred, digits=4,
                                         target_names=['Bajo', 'Medio', 'Alto'])
    }

    print(f"\n{'=' * 70}")
    print(f"  {nombre}")
    print(f"{'=' * 70}")
    print(f"  Accuracy (prueba):     {accuracy:.4f}")
    print(f"  Accuracy (CV media):   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n  Reporte de clasificación:")
    print(resultados[nombre]['reporte'])

# ============================================================
# 5. IMPORTANCIA DE CARACTERÍSTICAS (mejor árbol)
# ============================================================
print(f"\n{'=' * 70}")
print("  IMPORTANCIA DE CARACTERÍSTICAS (Árbol de Decisión)")
print("=" * 70)

mejor_arbol = 'Árbol de Decisión (max_depth=4)'
importancias = pd.DataFrame({
    'Característica': X.columns,
    'Importancia': resultados[mejor_arbol]['modelo'].feature_importances_
}).sort_values('Importancia', ascending=False)

print(importancias.head(10).to_string(index=False))

# ============================================================
# 6. VISUALIZACIONES
# ============================================================
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Clasificación de Probabilidad de Divorcio - Comparación de Modelos',
             fontsize=16, fontweight='bold')

# 6.1 Comparación de Accuracy
ax1 = plt.subplot(3, 3, 1)
nombres = list(resultados.keys())
acc_vals = [resultados[n]['accuracy'] for n in nombres]
cv_vals = [resultados[n]['cv_mean'] for n in nombres]

x = np.arange(len(nombres))
width = 0.35
bars1 = ax1.bar(x - width/2, acc_vals, width, label='Accuracy (prueba)', color='steelblue')
bars2 = ax1.bar(x + width/2, cv_vals, width, label='Accuracy (CV media)', color='coral')
ax1.set_ylabel('Accuracy')
ax1.set_title('Comparación de Accuracy')
ax1.set_xticks(x)
ax1.set_xticklabels([n[:12] + '...' if len(n) > 14 else n for n in nombres], rotation=15, ha='right')
ax1.legend()
ax1.set_ylim(0, 1)
for bar, v in zip(bars1, acc_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
for bar, v in zip(bars2, cv_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{v:.3f}', ha='center', fontsize=9, fontweight='bold', color='coral')

# Matrices de confusión
etiquetas = ['Bajo', 'Medio', 'Alto']
for idx, (nombre, res) in enumerate(resultados.items()):
    ax = plt.subplot(3, 4, 5 + idx)
    cm = confusion_matrix(y_test, res['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=etiquetas)
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    ax.set_title(f'{nombre[:18]}', fontsize=9)

# 6.2 Comparación real vs predicho por modelo
ax_ref = plt.subplot(3, 1, 3)
x_barras = np.arange(len(etiquetas))
width_bar = 0.2

# Datos reales
real_counts = y_test.value_counts().sort_index()
real_vals = [real_counts.get(i, 0) for i in range(3)]

for idx, (nombre, res) in enumerate(resultados.items()):
    pred_counts = pd.Series(res['y_pred']).value_counts().sort_index()
    pred_vals = [pred_counts.get(i, 0) for i in range(3)]
    ax_ref.bar(x_barras + idx * width_bar, pred_vals, width_bar,
               label=nombre[:18], alpha=0.8)

ax_ref.bar(x_barras + len(resultados) * width_bar / 2 - width_bar/2,
           real_vals, width_bar, label='Real', color='black', alpha=0.5)
ax_ref.set_xticks(x_barras + len(resultados) * width_bar / 2)
ax_ref.set_xticklabels(etiquetas)
ax_ref.set_ylabel('Cantidad')
ax_ref.set_title('Real vs Predicho por Modelo')
ax_ref.legend(fontsize=7, loc='upper right')

plt.tight_layout()
plt.savefig('ejercicios/comparacion_clasificacion_divorcio.png', dpi=150, bbox_inches='tight')
plt.show(block=False)
plt.pause(2)
plt.close('all')

# ============================================================
# 7. PREDICCIÓN DE EJEMPLO
# ============================================================
print(f"\n{'=' * 70}")
print("  PREDICCIÓN DE EJEMPLO")
print("=" * 70)

np.random.seed(42)
indice_ejemplo = 0
ejemplo = X_test.iloc[[indice_ejemplo]].copy()
valor_real_categoria = y_test.iloc[indice_ejemplo]
valor_real_prob = df.loc[X_test.index[indice_ejemplo], 'Divorce Probability']

print(f"\nCaracterísticas del ejemplo:")
for col in X.columns[:5]:
    print(f"  {col:45s} = {ejemplo[col].values[0]:.4f}")
print(f"  ... y {len(X.columns) - 5} características más")

print(f"\n  Valor REAL:      {etiquetas[valor_real_categoria]} (probabilidad: {valor_real_prob:.4f})")
print()
for nombre, res in resultados.items():
    pred = res['modelo'].predict(ejemplo)[0]
    acertó = "✓" if pred == valor_real_categoria else "✗"
    print(f"  {nombre:40s} -> {etiquetas[pred]:10s} {acertó}")

print(f"\n{'=' * 70}")
print("  ¡Gráfica guardada como 'comparacion_clasificacion_divorcio.png'!")
print("=" * 70)
