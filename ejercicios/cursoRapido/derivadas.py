import numpy as np
import sympy as sp
from scipy.optimize import minimize

class DerivativeCalculator:
    """Calculadora educativa de derivadas"""
    
    @staticmethod
    def numerical_derivative(f, x, h=1e-7):
        """Derivada numérica (diferencia central)"""
        return (f(x + h) - f(x - h)) / (2 * h)
    
    @staticmethod
    def numerical_gradient(f, x, h=1e-7):
        """Gradiente numérico"""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
        return grad
    
    @staticmethod
    def symbolic_derivative(expression, variable):
        """Derivada simbólica con SymPy"""
        x = sp.symbols(variable)
        f = sp.sympify(expression)
        return sp.diff(f, x)

# Ejemplo de uso
calc = DerivativeCalculator()

# Función de costo para regresión
def cost_function(theta, X, y):
    m = len(y)
    predictions = X @ theta
    return (1/(2*m)) * np.sum((predictions - y) ** 2)

# Crear datos sintéticos
np.random.seed(42)
X = np.random.randn(20, 3)
theta_true = np.array([2.5, -1.3, 0.8])
y = X @ theta_true + 0.1 * np.random.randn(20)

# Calcular gradiente numérico
def f(theta):
    return cost_function(theta, X, y)

theta_init = np.random.randn(3)
grad = calc.numerical_gradient(f, theta_init)
print(f"Gradiente numérico: {grad}")

# Derivada simbólica
expression = "x**2 + 3*x + 2"
derivative = calc.symbolic_derivative(expression, 'x')
print(f"f(x) = {expression}")
print(f"f'(x) = {derivative}")


class GradientDescent:
    """
    Implementación educativa del descenso de gradiente
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Args:
            learning_rate: Tasa de aprendizaje (α)
            max_iterations: Número máximo de iteraciones
            tolerance: Tolerancia para convergencia
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = {'loss': [], 'theta': []}
        
    def fit(self, X, y, theta_init=None):
        """
        Entrena modelo de regresión lineal con SGD
        
        Args:
            X: Matriz de características (m×n)
            y: Vector de etiquetas (m×1)
            theta_init: Parámetros iniciales
        """
        m, n = X.shape
        theta = theta_init if theta_init is not None else np.zeros(n)
        
        for iteration in range(self.max_iterations):
            # Forward pass: calcular predicciones
            predictions = X @ theta
            
            # Calcular error
            error = predictions - y
            
            # Calcular costo
            loss = (1/(2*m)) * np.sum(error ** 2)
            self.history['loss'].append(loss)
            self.history['theta'].append(theta.copy())
            
            # Calcular gradiente
            gradient = (1/m) * (X.T @ error)
            
            # Actualizar parámetros
            theta_new = theta - self.learning_rate * gradient
            
            # Verificar convergencia
            if np.linalg.norm(theta_new - theta) < self.tolerance:
                print(f"Convergencia alcanzada en iteración {iteration}")
                break
                
            theta = theta_new
            
        self.theta = theta
        return self
    
    def predict(self, X):
        """Hace predicciones con el modelo entrenado"""
        return X @ self.theta
    
    def plot_convergence(self):
        """Visualiza la convergencia del algoritmo"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Costo vs iteraciones
        ax1.plot(self.history['loss'])
        ax1.set_xlabel('Iteración')
        ax1.set_ylabel('Costo')
        ax1.set_title('Convergencia del Costo')
        ax1.grid(True, alpha=0.3)
        
        # Trayectoria de parámetros
        theta_history = np.array(self.history['theta'])
        for i in range(theta_history.shape[1]):
            ax2.plot(theta_history[:, i], label=f'θ_{i}')
        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('Valor del parámetro')
        ax2.set_title('Evolución de Parámetros')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Ejemplo práctico
# Datos sintéticos: relación no lineal con ruido
np.random.seed(42)
X = np.random.randn(100, 2)  # 2 características
theta_true = np.array([3.0, -2.0])
y = X @ theta_true + 0.5 * np.random.randn(100)

# Entrenar modelo
gd = GradientDescent(learning_rate=0.1, max_iterations=200)
gd.fit(X, y)

print(f"θ verdadero: {theta_true}")
print(f"θ aprendido: {gd.theta}")
print(f"Error: {np.linalg.norm(gd.theta - theta_true):.4f}")

# Visualizar convergencia
gd.plot_convergence()

class NeuralNetwork:
    """
    Implementación educativa de red neuronal con backpropagation
    """
    
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Args:
            layer_sizes: Lista de tamaños de capas [input, hidden1, hidden2, ..., output]
            learning_rate: Tasa de aprendizaje
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Inicialización Xavier
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            
    def sigmoid(self, x):
        """Función de activación sigmoid"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivada de sigmoid"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """
        Forward pass completo
        
        Returns:
            activations: Lista de salidas de cada capa
            z_values: Lista de valores antes de activación
        """
        activations = [X]
        z_values = []
        
        current = X
        for w, b in zip(self.weights, self.biases):
            z = current @ w + b
            z_values.append(z)
            current = self.sigmoid(z)
            activations.append(current)
            
        return activations, z_values
    
    def backward(self, activations, z_values, y):
        """
        Backward pass (backpropagation)
        
        Args:
            activations: Salidas de cada capa
            z_values: Valores antes de activación
            y: Etiquetas verdaderas
        """
        m = y.shape[0]
        grad_weights = [np.zeros_like(w) for w in self.weights]
        grad_biases = [np.zeros_like(b) for b in self.biases]
        
        # Error en la capa de salida
        delta = activations[-1] - y
        
        # Propagación hacia atrás
        for l in range(len(self.weights) - 1, -1, -1):
            # Gradiente de pesos
            grad_weights[l] = (activations[l].T @ delta) / m
            grad_biases[l] = np.sum(delta, axis=0, keepdims=True) / m
            
            # Propagar delta a la capa anterior (si no es la primera)
            if l > 0:
                delta = (delta @ self.weights[l].T) * self.sigmoid_derivative(z_values[l-1])
                
        return grad_weights, grad_biases
    
    def update_parameters(self, grad_weights, grad_biases):
        """Actualiza parámetros con SGD"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_weights[i]
            self.biases[i] -= self.learning_rate * grad_biases[i]
    
    def train(self, X, y, epochs=1000):
        """
        Entrenamiento completo de la red
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            activations, z_values = self.forward(X)
            
            # Calcular pérdida (MSE)
            loss = np.mean((activations[-1] - y) ** 2)
            losses.append(loss)
            
            # Backward pass
            grad_weights, grad_biases = self.backward(activations, z_values, y)
            
            # Actualizar parámetros
            self.update_parameters(grad_weights, grad_biases)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
                
        return losses
    
    def predict(self, X):
        """Predicción"""
        activations, _ = self.forward(X)
        return activations[-1]

# Aplicación: Clasificación de dígitos MNIST (versión simplificada)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar datos
digits = load_digits()
X, y = digits.data, digits.target

# Normalizar
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encoding para clasificación multiclase
def one_hot(y, n_classes=10):
    return np.eye(n_classes)[y].astype(int)

y_onehot = one_hot(y)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

# Crear y entrenar red neuronal
nn = NeuralNetwork([64, 128, 64, 10], learning_rate=0.5)
losses = nn.train(X_train, y_train, epochs=500)

# Evaluación
predictions = nn.predict(X_test)
accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
print(f"\nPrecisión en test: {accuracy:.4f}")

# Visualizar pérdida
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evolución de la Pérdida durante el Entrenamiento')
plt.grid(True, alpha=0.3)
plt.show()