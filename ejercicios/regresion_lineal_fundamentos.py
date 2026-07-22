import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionFromScratch:
    """
    Implementación completa de regresión lineal desde cero
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, 
                 regularization=None, lambda_reg=0.01):
        """
        Args:
            learning_rate: Tasa de aprendizaje para SGD
            n_iterations: Número de iteraciones
            regularization: 'l1', 'l2', 'elasticnet' o None
            lambda_reg: Factor de regularización
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.weight_history = []
    
    def _add_bias(self, X):
        """Agrega columna de bias para implementación matricial"""
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    def _loss(self, X, y, weights):
        """Calcula la pérdida con regularización"""
        m = len(y)
        predictions = X @ weights
        loss = (1/(2*m)) * np.sum((predictions - y) ** 2)
        
        # Regularización
        if self.regularization == 'l2':
            loss += (self.lambda_reg/(2*m)) * np.sum(weights[1:] ** 2)
        elif self.regularization == 'l1':
            loss += (self.lambda_reg/m) * np.sum(np.abs(weights[1:]))
        elif self.regularization == 'elasticnet':
            loss += (self.lambda_reg/(2*m)) * np.sum(weights[1:] ** 2)
            loss += (self.lambda_reg/m) * np.sum(np.abs(weights[1:]))
        
        return loss
    
    def _gradient(self, X, y, weights):
        """Calcula el gradiente con regularización"""
        m = len(y)
        predictions = X @ weights
        gradient = (1/m) * (X.T @ (predictions - y))
        
        # Gradiente de regularización (sin incluir bias)
        if self.regularization == 'l2':
            gradient[1:] += (self.lambda_reg/m) * weights[1:]
        elif self.regularization == 'l1':
            gradient[1:] += (self.lambda_reg/m) * np.sign(weights[1:])
        elif self.regularization == 'elasticnet':
            gradient[1:] += (self.lambda_reg/m) * weights[1:]
            gradient[1:] += (self.lambda_reg/m) * np.sign(weights[1:])
        
        return gradient
    
    def fit(self, X, y, verbose=False):
        """
        Entrena el modelo usando descenso de gradiente
        
        Args:
            X: Matriz de características (m×n)
            y: Vector de etiquetas (m×1)
            verbose: Si mostrar progreso
        """
        # Preparar datos
        X_with_bias = self._add_bias(X)
        m, n = X_with_bias.shape
        
        # Inicializar pesos
        np.random.seed(42)
        self.weights = np.random.randn(n) * 0.01
        self.loss_history = []
        self.weight_history = []
        
        # Descenso de gradiente
        for iteration in range(self.n_iterations):
            # Calcular pérdida
            loss = self._loss(X_with_bias, y, self.weights)
            self.loss_history.append(loss)
            self.weight_history.append(self.weights.copy())
            
            # Calcular gradiente
            gradient = self._gradient(X_with_bias, y, self.weights)
            
            # Actualizar pesos
            self.weights -= self.learning_rate * gradient
            
            # Verificar convergencia
            if iteration > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-8:
                if verbose:
                    print(f"Convergencia alcanzada en iteración {iteration}")
                break
            
            if verbose and iteration % 100 == 0:
                print(f"Iteración {iteration}, Loss: {loss:.6f}")
        
        # Separar bias de pesos
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
        
        return self
    
    def predict(self, X):
        """Hace predicciones"""
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """Calcula R² score"""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
    def plot_convergence(self):
        """Visualiza la convergencia del algoritmo"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Pérdida
        ax1.plot(self.loss_history)
        ax1.set_xlabel('Iteración')
        ax1.set_ylabel('Pérdida')
        ax1.set_title('Convergencia de la Pérdida')
        ax1.grid(True, alpha=0.3)
        
        # Pesos
        weight_history = np.array(self.weight_history)
        for i in range(weight_history.shape[1]):
            ax2.plot(weight_history[:, i], label=f'w_{i}')
        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('Peso')
        ax2.set_title('Evolución de los Pesos')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, feature_names=None):
        """Retorna la importancia de las características"""
        importance = np.abs(self.weights)
        importance = importance / importance.sum()
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        return sorted(zip(feature_names, importance), 
                     key=lambda x: x[1], reverse=True)

# Ejemplo práctico: Predicción de precios de viviendas
np.random.seed(42)

# Generar datos sintéticos
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar modelo
model = LinearRegressionFromScratch(learning_rate=0.1, n_iterations=500, 
                                   regularization='l2', lambda_reg=0.01)
model.fit(X_train, y_train, verbose=True)

# Evaluar
y_pred = model.predict(X_test)
print(f"\nR² Score: {model.score(X_test, y_test):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

# Importancia de características
importance = model.get_feature_importance()
print("\nImportancia de características:")
for feature, imp in importance:
    print(f"  {feature}: {imp:.4f}")

# Visualizar convergencia
model.plot_convergence()

# Comparar con sklearn
from sklearn.linear_model import LinearRegression as SklearnLinear

sk_model = SklearnLinear()
sk_model.fit(X_train, y_train)
sk_score = sk_model.score(X_test, y_test)
print(f"\nComparación con sklearn:")
print(f"  Modelo desde cero: {model.score(X_test, y_test):.4f}")
print(f"  Modelo sklearn: {sk_score:.4f}")