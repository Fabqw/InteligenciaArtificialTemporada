"""
PROYECTO SOCIAL: Sistema de Detección de Neumonía en Radiografías de Tórax
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import seaborn as sns


# Minimal classifier to replace missing CNN implementation
class CNNClassifier:
    """Simplified classifier using a flatten + softmax logistic model.
    Provides train and forward methods used by PneumoniaDetector.
    This is a lightweight stand-in for a real CNN for testing purposes.
    """
    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.w = None
        self.b = None

    def _softmax(self, z):
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, X, training=False):
        Xf = X.reshape(X.shape[0], -1)
        logits = Xf.dot(self.w) + self.b
        return self._softmax(logits)

    def train(self, X, y, epochs=10, batch_size=32, learning_rate=0.01, verbose=True):
        Xf = X.reshape(X.shape[0], -1)
        n_samples, n_features = Xf.shape
        # one-hot labels
        Y = np.zeros((n_samples, self.n_classes))
        Y[np.arange(n_samples), y.astype(int)] = 1

        # initialize
        rng = np.random.RandomState(42)
        self.w = rng.normal(scale=0.01, size=(n_features, self.n_classes))
        self.b = np.zeros((1, self.n_classes))

        losses = []
        accuracies = []

        for epoch in range(epochs):
            # simple full-batch gradient descent for stability in tests
            probs = self._softmax(Xf.dot(self.w) + self.b)
            loss = -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))

            grad_logits = (probs - Y) / n_samples
            grad_w = Xf.T.dot(grad_logits)
            grad_b = np.sum(grad_logits, axis=0, keepdims=True)

            self.w -= learning_rate * grad_w
            self.b -= learning_rate * grad_b

            preds = np.argmax(probs, axis=1)
            acc = np.mean(preds == y)

            losses.append(loss)
            accuracies.append(acc)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f} - acc: {acc:.4f}")

        return losses, accuracies


class PneumoniaDetector:
    """
    Sistema de detección de neumonía usando CNN para radiografías de tórax
    Aplicación con impacto social en áreas rurales sin acceso a especialistas
    """
    
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
    def generate_synthetic_chest_xray(self, n_samples=1000, img_size=64):
        """
        Genera radiografías sintéticas simulando patrones de neumonía
        
        Args:
            n_samples: Número de muestras
            img_size: Tamaño de la imagen (img_size x img_size)
        
        Returns:
            X: Imágenes simuladas (n_samples, img_size, img_size, 1)
            y: Etiquetas (0: normal, 1: neumonía)
        """
        np.random.seed(42)
        
        X = np.zeros((n_samples, img_size, img_size, 1))
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Crear imagen base: estructura pulmonar simulada
            img = np.zeros((img_size, img_size))
            
            # Simular campos pulmonares (elipses)
            center_y, center_x = img_size // 2, img_size // 2
            for _ in range(2):  # Dos pulmones
                cx = center_x + np.random.randint(-10, 10)
                cy = center_y + np.random.randint(-5, 5)
                rx = np.random.randint(15, 25)
                ry = np.random.randint(10, 20)
                
                y_grid, x_grid = np.ogrid[-cy:img_size-cy, -cx:img_size-cx]
                mask = (x_grid**2 / rx**2 + y_grid**2 / ry**2) <= 1
                img[mask] += np.random.uniform(0.4, 0.8) * (1 + np.random.randn() * 0.1)
            
            # Añadir ruido
            img += np.random.normal(0, 0.02, (img_size, img_size))
            img = np.clip(img, 0, 1)
            
            # Decidir si es neumonía (40% de los casos)
            if np.random.rand() < 0.4:
                y[i] = 1
                
                # Añadir opacidades (neumonía)
                num_opacities = np.random.randint(2, 6)
                for _ in range(num_opacities):
                    ox = np.random.randint(10, img_size-10)
                    oy = np.random.randint(10, img_size-10)
                    oradius = np.random.randint(3, 10)
                    
                    y_grid, x_grid = np.ogrid[-oy:img_size-oy, -ox:img_size-ox]
                    opacity_mask = (x_grid**2 + y_grid**2) <= oradius**2
                    img[opacity_mask] += np.random.uniform(0.1, 0.4)
                
                # Añadir consolidación
                if np.random.rand() < 0.6:
                    cx = np.random.randint(15, img_size-15)
                    cy = np.random.randint(15, img_size-15)
                    consolidation = np.random.uniform(0.6, 0.9)
                    img[cy-5:cy+5, cx-5:cx+5] = consolidation
                
                img = np.clip(img, 0, 1)
            
            X[i, :, :, 0] = img
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2):
        """Prepara datos para entrenamiento"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"Datos de entrenamiento: {X_train.shape}")
        print(f"Datos de prueba: {X_test.shape}")
        print(f"Distribución de entrenamiento: {np.sum(y_train)/len(y_train):.2%} neumonía")
        print(f"Distribución de prueba: {np.sum(y_test)/len(y_test):.2%} neumonía")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, img_size=64):
        """Construye la CNN para detección de neumonía"""
        self.model = CNNClassifier((img_size, img_size, 1), 2)
        return self.model
    
    def train(self, epochs=30, batch_size=32, learning_rate=0.001, verbose=True):
        """Entrena el modelo"""
        if self.model is None:
            self.build_model()
        
        losses, accuracies = self.model.train(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose
        )
        
        return losses, accuracies
    
    def evaluate(self):
        """Evalúa el modelo con métricas completas"""
        # Predicciones
        y_pred_proba = self.model.forward(self.X_test, training=False)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Métricas
        accuracy = np.mean(y_pred == self.y_test)
        auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
        
        print("\n=== EVALUACIÓN COMPLETA DEL MODELO ===")
        print(f"Precisión: {accuracy:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        
        print("\nReporte de Clasificación:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Normal', 'Neumonía']))
        
        # Análisis detallado
        self.analyze_predictions(y_pred_proba, y_pred)
        
        return accuracy, auc
    
    def analyze_predictions(self, y_pred_proba, y_pred):
        """Análisis detallado de predicciones"""
        
        # Matriz de confusión
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Matriz de confusión
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Neumonía'],
                   yticklabels=['Normal', 'Neumonía'], ax=axes[0, 0])
        axes[0, 0].set_title('Matriz de Confusión')
        axes[0, 0].set_xlabel('Predicción')
        axes[0, 0].set_ylabel('Real')
        
        # 2. Curva ROC
        axes[0, 1].plot(fpr, tpr, linewidth=2)
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_xlabel('Tasa de Falsos Positivos')
        axes[0, 1].set_ylabel('Tasa de Verdaderos Positivos')
        axes[0, 1].set_title('Curva ROC')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribución de predicciones
        axes[1, 0].hist(y_pred_proba[self.y_test==0, 1], bins=20, alpha=0.5, 
                       label='Normal', color='blue')
        axes[1, 0].hist(y_pred_proba[self.y_test==1, 1], bins=20, alpha=0.5, 
                       label='Neumonía', color='red')
        axes[1, 0].set_xlabel('Probabilidad de Neumonía')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].set_title('Distribución de Predicciones')
        axes[1, 0].legend()
        
        # 4. Ejemplos de clasificación
        # Seleccionar ejemplos representativos
        correct_normal = np.where((y_pred == 0) & (self.y_test == 0))[0][:2]
        correct_pneumonia = np.where((y_pred == 1) & (self.y_test == 1))[0][:2]
        wrong_normal = np.where((y_pred == 1) & (self.y_test == 0))[0][:1]
        wrong_pneumonia = np.where((y_pred == 0) & (self.y_test == 1))[0][:1]
        
        indices = list(correct_normal) + list(correct_pneumonia) + \
                  list(wrong_normal) + list(wrong_pneumonia)
        
        for i, idx in enumerate(indices[:4]):
            ax = axes[1, 1] if i < 4 else None
            if ax is not None:
                img = self.X_test[idx]
                ax.imshow(img.squeeze(), cmap='gray')
                pred_class = 'Neumonía' if y_pred[idx] == 1 else 'Normal'
                true_class = 'Neumonía' if self.y_test[idx] == 1 else 'Normal'
                prob = y_pred_proba[idx, 1] if y_pred[idx] == 1 else 1 - y_pred_proba[idx, 0]
                color = 'green' if y_pred[idx] == self.y_test[idx] else 'red'
                ax.set_title(f'Pred: {pred_class}\nReal: {true_class}\nConf: {prob:.2f}', 
                            color=color)
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def predict_single(self, xray_image):
        """Predice para una sola radiografía"""
        if len(xray_image.shape) == 2:
            xray_image = xray_image.reshape(1, *xray_image.shape, 1)
        elif len(xray_image.shape) == 3:
            xray_image = xray_image.reshape(1, *xray_image.shape)
        
        proba = self.model.forward(xray_image, training=False)
        pred = np.argmax(proba, axis=1)[0]
        confidence = proba[0, pred]
        
        result = {
            'prediccion': 'Neumonía' if pred == 1 else 'Normal',
            'probabilidad_neumonia': float(proba[0, 1]),
            'confianza': float(confidence),
            'recomendacion': self._get_recommendation(proba[0, 1])
        }
        
        return result
    
    def _get_recommendation(self, prob_neumonia):
        """Genera recomendación basada en la probabilidad"""
        if prob_neumonia < 0.2:
            return "Sin evidencia de neumonía. El paciente parece estar sano."
        elif prob_neumonia < 0.5:
            return "Baja probabilidad de neumonía. Se recomienda monitoreo."
        elif prob_neumonia < 0.7:
            return "Probabilidad moderada. Se recomienda evaluación por especialista."
        elif prob_neumonia < 0.9:
            return "Alta probabilidad. Derivar a atención médica inmediata."
        else:
            return "Muy alta probabilidad. Atención médica urgente requerida."

# Implementación y prueba
print("=== SISTEMA DE DETECCIÓN DE NEUMONÍA ===")
print("Inicializando sistema...")

detector = PneumoniaDetector()

print("\nGenerando radiografías sintéticas...")
X, y = detector.generate_synthetic_chest_xray(n_samples=1500, img_size=64)

# Visualizar algunas muestras
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    ax = axes[i//4, i%4]
    ax.imshow(X[i].squeeze(), cmap='gray')
    ax.set_title(f'Clase: {"Neumonía" if y[i] == 1 else "Normal"}')
    ax.axis('off')
plt.suptitle('Ejemplos de Radiografías Sintéticas')
plt.tight_layout()
plt.show()

print("\nPreparando datos...")
X_train, X_test, y_train, y_test = detector.prepare_data(X, y, test_size=0.2)

print("\nConstruyendo modelo CNN...")
detector.build_model(img_size=64)

print("\nEntrenando modelo...")
losses, accuracies = detector.train(epochs=30, batch_size=32, learning_rate=0.001)

# Evaluación
print("\nEvaluando modelo...")
detector.evaluate()

# Caso de prueba
print("\n=== PRUEBA CON NUEVO PACIENTE ===")
test_patient = X_test[0:1]
result = detector.predict_single(test_patient)
print(f"Predicción: {result['prediccion']}")
print(f"Probabilidad de neumonía: {result['probabilidad_neumonia']:.2%}")
print(f"Confianza: {result['confianza']:.2%}")
print(f"Recomendación: {result['recomendacion']}")

# Analizar sensibilidad del modelo
def analyze_model_sensitivity(detector, X_test, y_test):
    """Analiza la sensibilidad del modelo a diferentes niveles de confianza"""
    
    y_pred_proba = detector.model.forward(X_test, training=False)
    
    thresholds = np.linspace(0.1, 0.9, 9)
    sensitivities = []
    specificities = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
        
        tn = np.sum((y_pred == 0) & (y_test == 0))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        tp = np.sum((y_pred == 1) & (y_test == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, sensitivities, 'b-', label='Sensibilidad', linewidth=2)
    plt.plot(thresholds, specificities, 'r-', label='Especificidad', linewidth=2)
    plt.xlabel('Umbral de Decisión')
    plt.ylabel('Métrica')
    plt.title('Análisis de Sensibilidad del Modelo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Encontrar mejor umbral (maximizar sensibilidad + especificidad)
    optimal_idx = np.argmax(np.array(sensitivities) + np.array(specificities))
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nMejor umbral: {optimal_threshold:.2f}")
    print(f"  Sensibilidad: {sensitivities[optimal_idx]:.3f}")
    print(f"  Especificidad: {specificities[optimal_idx]:.3f}")
    
    return optimal_threshold

print("\n=== ANÁLISIS DE SENSIBILIDAD ===")
optimal_threshold = analyze_model_sensitivity(detector, X_test, y_test)