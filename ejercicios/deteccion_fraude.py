"""
PROYECTO MÓDULO 3: Sistema de Detección de Fraude para Microcréditos
Aplicación con impacto social en comunidades vulnerables
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns


class NeuralNetworkDeep:
    """Wrapper simple para una red neuronal profunda basada en sklearn."""

    def __init__(self, layer_dims, activation='relu', output_activation='sigmoid'):
        hidden_layers = tuple(layer_dims[1:-1])
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver='adam',
            max_iter=1,
            warm_start=True,
            random_state=42,
        )

    def train(self, X, y, epochs=1, batch_size=64, learning_rate=0.01, optimizer='adam', verbose=True):
        y = y.ravel()
        losses, accuracies = [], []
        for epoch in range(epochs):
            self.model.fit(X, y)
            proba = self.forward(X).ravel()
            pred = (proba > 0.5).astype(int)
            acc = np.mean(pred == y)
            loss = -np.mean(y * np.log(proba + 1e-9) + (1 - y) * np.log(1 - proba + 1e-9))
            losses.append(loss)
            accuracies.append(acc)
            if verbose and (epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == epochs):
                print(f"Epoch {epoch + 1}/{epochs} - loss: {loss:.4f} - acc: {acc:.4f}")
        return losses, accuracies

    def forward(self, X):
        return self.model.predict_proba(X)[:, 1].reshape(-1, 1)

class MicrocreditFraudDetector:
    """
    Sistema de detección de fraude para microcréditos
    Diseñado para proteger a comunidades vulnerables
    """
    
    def __init__(self, hidden_layers=[128, 64, 32]):
        self.hidden_layers = hidden_layers
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.threshold = 0.5
        
    def generate_synthetic_data(self, n_samples=10000):
        """
        Genera datos sintéticos de microcréditos
        Incluye características relevantes para comunidades
        """
        np.random.seed(42)
        
        # Características socioeconómicas
        data = {
            'edad': np.random.randint(18, 70, n_samples),
            'ingresos_mensuales': np.random.exponential(500, n_samples) + 200,
            'gastos_mensuales': np.random.exponential(400, n_samples) + 100,
            'historial_credito': np.random.randint(0, 5, n_samples),
            'tiempo_empleo': np.random.randint(0, 20, n_samples),
            'educacion': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15]),
            'dependientes': np.random.poisson(2, n_samples),
            'vivienda_propia': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'historial_transacciones': np.random.exponential(10, n_samples),
            'monto_prestamo': np.random.exponential(1000, n_samples) + 100,
            'plazo_meses': np.random.choice([3, 6, 9, 12, 18, 24], n_samples),
            'tasa_interes': np.random.uniform(0.05, 0.25, n_samples),
            'historial_telefonico': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'referencias_laborales': np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.4, 0.2]),
        }
        
        df = pd.DataFrame(data)
        
        # Calcular score de riesgo
        df['score_riesgo'] = (
            -0.1 * df['ingresos_mensuales'] / 100 +
            0.2 * df['gastos_mensuales'] / 100 +
            -0.3 * df['historial_credito'] +
            0.2 * df['tiempo_empleo'] +
            -0.1 * df['educacion'] +
            0.15 * df['dependientes'] +
            0.1 * (1 - df['vivienda_propia']) +
            0.2 * df['historial_transacciones'] / 10 +
            0.1 * (df['monto_prestamo'] / 1000) +
            0.05 * df['plazo_meses'] / 12 +
            0.1 * df['tasa_interes'] * 10 +
            0.15 * (1 - df['historial_telefonico']) +
            0.1 * (1 - df['referencias_laborales'] / 3)
        )
        
        # Añadir ruido
        df['score_riesgo'] += np.random.normal(0, 0.1, n_samples)
        
        # Generar fraude (1) o no fraude (0)
        fraud_prob = 1 / (1 + np.exp(-df['score_riesgo']))
        df['fraude'] = (fraud_prob > 0.5).astype(int)
        
        return df
    
    def prepare_data(self, df):
        """Prepara los datos para el entrenamiento"""
        features = ['edad', 'ingresos_mensuales', 'gastos_mensuales', 
                   'historial_credito', 'tiempo_empleo', 'educacion', 'dependientes',
                   'vivienda_propia', 'historial_transacciones', 'monto_prestamo',
                   'plazo_meses', 'tasa_interes', 'historial_telefonico',
                   'referencias_laborales']
        
        X = df[features].values
        y = df['fraude'].values.reshape(-1, 1)
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, features
    
    def train(self, df, epochs=500, learning_rate=0.01, verbose=True):
        """Entrena el modelo"""
        X, y, features = self.prepare_data(df)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Construir modelo
        layer_dims = [X.shape[1]] + self.hidden_layers + [1]
        self.model = NeuralNetworkDeep(layer_dims, activation='relu', output_activation='sigmoid')
        
        # Entrenar
        losses, accuracies = self.model.train(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=64, 
            learning_rate=learning_rate,
            optimizer='adam',
            verbose=verbose
        )
        
        # Evaluar
        y_pred_proba = self.model.forward(X_test)
        y_pred = (y_pred_proba > self.threshold).astype(int)
        
        accuracy = np.mean(y_pred.flatten() == y_test.flatten())
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\n=== EVALUACIÓN DEL MODELO ===")
        print(f"Precisión: {accuracy:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred))
        
        # Calcular importancia de características
        self.feature_importance = self.calculate_feature_importance(X_train, y_train, features)
        
        # Visualizar resultados
        self.plot_results(losses, accuracies, y_test, y_pred_proba, y_pred)
        
        return self.model
    
    def calculate_feature_importance(self, X, y, features):
        """Calcula importancia usando permutación"""
        base_score = self.model.forward(X)
        base_auc = roc_auc_score(y, base_score)
        
        importance = {}
        for i, feature in enumerate(features):
            X_permuted = X.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            perm_score = self.model.forward(X_permuted)
            perm_auc = roc_auc_score(y, perm_score)
            importance[feature] = base_auc - perm_auc
        
        return sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    def plot_results(self, losses, accuracies, y_test, y_pred_proba, y_pred):
        """Visualiza resultados del modelo"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Pérdida de entrenamiento
        axes[0, 0].plot(losses)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Pérdida de Entrenamiento')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Precisión de entrenamiento
        axes[0, 1].plot(accuracies)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Precisión de Entrenamiento')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        axes[0, 2].plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        axes[0, 2].plot([0, 1], [0, 1], 'k--')
        axes[0, 2].set_xlabel('Tasa de Falsos Positivos')
        axes[0, 2].set_ylabel('Tasa de Verdaderos Positivos')
        axes[0, 2].set_title('Curva ROC')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicción')
        axes[1, 0].set_ylabel('Real')
        axes[1, 0].set_title('Matriz de Confusión')
        
        # 5. Importancia de características
        features_imp = [f[0] for f in self.feature_importance[:10]]
        imp_values = [f[1] for f in self.feature_importance[:10]]
        axes[1, 1].barh(features_imp, imp_values)
        axes[1, 1].set_xlabel('Importancia')
        axes[1, 1].set_title('Top 10 Características Importantes')
        
        # 6. Distribución de probabilidades
        axes[1, 2].hist(y_pred_proba[y_test.flatten() == 0], bins=30, alpha=0.5, label='No Fraude')
        axes[1, 2].hist(y_pred_proba[y_test.flatten() == 1], bins=30, alpha=0.5, label='Fraude')
        axes[1, 2].set_xlabel('Probabilidad de Fraude')
        axes[1, 2].set_ylabel('Frecuencia')
        axes[1, 2].set_title('Distribución de Predicciones')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict_fraud_probability(self, applicant_data):
        """Predice probabilidad de fraude para nuevos solicitantes"""
        X_scaled = self.scaler.transform(applicant_data)
        prob = self.model.forward(X_scaled)
        return prob
    
    def get_risk_assessment(self, applicant_data):
        """Genera un reporte completo de riesgo"""
        prob = self.predict_fraud_probability(applicant_data)
        
        if prob < 0.3:
            risk_level = "Bajo"
            recommendation = "Aprobar préstamo"
        elif prob < 0.6:
            risk_level = "Medio"
            recommendation = "Revisar manualmente"
        else:
            risk_level = "Alto"
            recommendation = "Rechazar préstamo"
        
        return {
            'probabilidad_fraude': float(prob),
            'nivel_riesgo': risk_level,
            'recomendacion': recommendation,
            'factores_riesgo': [f[0] for f in self.feature_importance[:5]]
        }

# Implementación y prueba
print("=== SISTEMA DE DETECCIÓN DE FRAUDE ===")
print("Inicializando sistema...")

detector = MicrocreditFraudDetector(hidden_layers=[128, 64, 32])

print("Generando datos sintéticos...")
df = detector.generate_synthetic_data(n_samples=10000)

print("\nCaracterísticas del dataset:")
print(df.describe())

print("\nDistribución de fraude:")
print(df['fraude'].value_counts())

print("\nEntrenando modelo...")
detector.train(df, epochs=300, learning_rate=0.01, verbose=True)

# Caso de prueba: solicitante de préstamo
test_applicant = np.array([[
    35,  # edad
    800,  # ingresos_mensuales
    600,  # gastos_mensuales
    2,    # historial_credito
    5,    # tiempo_empleo
    3,    # educacion
    2,    # dependientes
    0,    # vivienda_propia
    8,    # historial_transacciones
    500,  # monto_prestamo
    12,   # plazo_meses
    0.15, # tasa_interes
    1,    # historial_telefonico
    2     # referencias_laborales
]])

assessment = detector.get_risk_assessment(test_applicant)
print("\n=== EVALUACIÓN DE SOLICITANTE ===")
print(f"Probabilidad de fraude: {assessment['probabilidad_fraude']:.2%}")
print(f"Nivel de riesgo: {assessment['nivel_riesgo']}")
print(f"Recomendación: {assessment['recomendacion']}")
print(f"Factores de riesgo principales: {', '.join(assessment['factores_riesgo'])}")