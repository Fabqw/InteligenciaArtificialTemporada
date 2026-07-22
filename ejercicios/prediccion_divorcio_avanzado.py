import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

# ============================================================
# 1. CARGA Y EXPLORACIÓN
# ============================================================
df = pd.read_csv('ejercicios/archive/Marriage_Divorce_DB.csv')

print("=" * 70)
print("DATASET MARRIAGE DIVORCE DB")
print("=" * 70)
print(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"\nVariable objetivo: Divorce Probability")
print(f"  Min: {df['Divorce Probability'].min():.4f}")
print(f"  Max: {df['Divorce Probability'].max():.4f}")
print(f"  Media: {df['Divorce Probability'].mean():.4f}")
print(f"  Desv. estándar: {df['Divorce Probability'].std():.4f}")

# ============================================================
# 2. SEPARAR CARACTERÍSTICAS Y OBJETIVO
# ============================================================
X = df.drop('Divorce Probability', axis=1)
y = df['Divorce Probability']

# ============================================================
# 3. DIVIDIR EN ENTRENAMIENTO Y PRUEBA
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nEntrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")

# ============================================================
# 4. ENTRENAR MODELOS
# ============================================================
modelos = {
    'Regresión Lineal': LinearRegression(),
    'Árbol de Decisión (max_depth=4)': DecisionTreeRegressor(max_depth=4, random_state=42),
    'Árbol de Decisión (max_depth=6)': DecisionTreeRegressor(max_depth=6, random_state=42),
}

resultados = {}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Validación cruzada
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='r2')

    resultados[nombre] = {
        'modelo': modelo,
        'y_pred': y_pred,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred_train': modelo.predict(X_train)
    }

    print(f"\n{'=' * 70}")
    print(f"  {nombre}")
    print(f"{'=' * 70}")
    print(f"  R² (prueba):       {r2:.4f}")
    print(f"  RMSE (prueba):     {rmse:.4f}")
    print(f"  MAE (prueba):      {mae:.4f}")
    print(f"  R² CV (media):     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 5. IMPORTANCIA DE CARACTERÍSTICAS (Árbol de Decisión)
# ============================================================
print(f"\n{'=' * 70}")
print("  IMPORTANCIA DE CARACTERÍSTICAS (Árbol de Decisión)")
print("=" * 70)

arbol = resultados['Árbol de Decisión (max_depth=6)']['modelo']
importancias = pd.DataFrame({
    'Característica': X.columns,
    'Importancia': arbol.feature_importances_
}).sort_values('Importancia', ascending=False)

print(importancias.head(10).to_string(index=False))

# ============================================================
# 6. VISUALIZACIONES
# ============================================================
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Predicción de Probabilidad de Divorcio - Comparación de Modelos',
             fontsize=16, fontweight='bold')

# 6.1 Comparación R²
ax1 = plt.subplot(3, 3, 1)
nombres = list(resultados.keys())
r2_vals = [resultados[n]['r2'] for n in nombres]
cv_vals = [resultados[n]['cv_mean'] for n in nombres]

x = np.arange(len(nombres))
width = 0.35
bars1 = ax1.bar(x - width/2, r2_vals, width, label='R² (prueba)', color='steelblue')
bars2 = ax1.bar(x + width/2, cv_vals, width, label='R² (CV media)', color='coral')
ax1.set_ylabel('R²')
ax1.set_title('Comparación de R²')
ax1.set_xticks(x)
ax1.set_xticklabels([n[:12] + '...' if len(n) > 14 else n for n in nombres], rotation=15, ha='right')
ax1.legend()
ax1.set_ylim(0, 1)
for bar, v in zip(bars1, r2_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

# 6.2 Real vs Predicho scatter
ax2 = plt.subplot(3, 3, 2)
colors = ['#2E86AB', '#A23B72', '#F18F01']
for idx, (nombre, res) in enumerate(resultados.items()):
    ax2.scatter(y_test, res['y_pred'], alpha=0.7, label=nombre, color=colors[idx], s=40)
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', alpha=0.5, label='Perfecto')
ax2.set_xlabel('Valor Real')
ax2.set_ylabel('Valor Predicho')
ax2.set_title('Real vs Predicho')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# 6.3 Errores (residuals)
ax3 = plt.subplot(3, 3, 3)
for idx, (nombre, res) in enumerate(resultados.items()):
    residuos = y_test - res['y_pred']
    ax3.scatter(res['y_pred'], residuos, alpha=0.6, label=nombre, color=colors[idx], s=30)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax3.set_xlabel('Valor Predicho')
ax3.set_ylabel('Residuo (Real - Predicho)')
ax3.set_title('Gráfico de Residuos')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 6.4 Árbol de Decisión (max_depth=4)
ax4 = plt.subplot(3, 2, 3)
plot_tree(resultados['Árbol de Decisión (max_depth=4)']['modelo'],
          feature_names=list(X.columns),
          filled=True, rounded=True,
          ax=ax4, fontsize=7,
          max_depth=3)
ax4.set_title('Árbol de Decisión (max_depth=4) - primeros niveles', fontsize=10)

# 6.5 Árbol de Decisión (max_depth=6) - feature importance
ax5 = plt.subplot(3, 2, 4)
top10 = importancias.head(10)
ax5.barh(range(len(top10)), top10['Importancia'].values, color='#2E86AB')
ax5.set_yticks(range(len(top10)))
ax5.set_yticklabels(top10['Característica'].values)
ax5.set_xlabel('Importancia')
ax5.set_title('Top 10 Características más Importantes', fontsize=10)
ax5.invert_yaxis()

# 6.6 Distribución de errores
ax6 = plt.subplot(3, 3, 6)
for idx, (nombre, res) in enumerate(resultados.items()):
    errores = y_test - res['y_pred']
    ax6.hist(errores, alpha=0.5, bins=10, label=nombre, color=colors[idx])
ax6.set_xlabel('Error')
ax6.set_ylabel('Frecuencia')
ax6.set_title('Distribución de Errores')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# 6.7 Comparación visual de predicciones (muestras de test)
ax7 = plt.subplot(3, 3, 7)
sample_idx = np.arange(len(y_test))
ax7.plot(sample_idx, y_test.values, 'ko-', label='Real', markersize=4, linewidth=1.5)
for idx, (nombre, res) in enumerate(resultados.items()):
    ax7.plot(sample_idx, res['y_pred'], 's-', label=nombre, markersize=3, linewidth=1, alpha=0.7, color=colors[idx])
ax7.set_xlabel('Muestra de prueba')
ax7.set_ylabel('Probabilidad de Divorcio')
ax7.set_title('Predicciones por Muestra')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# Tabla de métricas
ax8 = plt.subplot(3, 3, 8)
ax8.axis('tight')
ax8.axis('off')
tabla_data = []
for nombre, res in resultados.items():
    tabla_data.append([
        nombre[:20],
        f"{res['r2']:.4f}",
        f"{res['rmse']:.4f}",
        f"{res['mae']:.4f}",
        f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}"
    ])
tabla = ax8.table(cellText=tabla_data,
                   colLabels=['Modelo', 'R²', 'RMSE', 'MAE', 'R² CV'],
                   cellLoc='center',
                   loc='center',
                   colWidths=[0.22, 0.12, 0.12, 0.12, 0.22])
tabla.auto_set_font_size(False)
tabla.set_fontsize(9)
tabla.scale(1, 1.5)
ax8.set_title('Resumen de Métricas', fontsize=10, pad=20)

plt.tight_layout()
plt.savefig('ejercicios/comparacion_modelos_divorcio.png', dpi=150, bbox_inches='tight')
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
ejemplo = X_test.iloc[[0]].copy()
print(f"\nCaracterísticas del ejemplo (fila real de prueba):")
for col in X.columns:
    print(f"  {col:45s} = {ejemplo[col].values[0]:.4f}")

print(f"\n  Valor REAL de Divorce Probability: {y_test.iloc[0]:.4f}")
print()
for nombre, res in resultados.items():
    pred = res['modelo'].predict(ejemplo)[0]
    error = abs(pred - y_test.iloc[0])
    print(f"  {nombre:40s} -> Predicción: {pred:.4f}  |  Error: {error:.4f}")

print(f"\n{'=' * 70}")
print("  ¡Gráfica guardada como 'comparacion_modelos_divorcio.png'!")
print("=" * 70)
