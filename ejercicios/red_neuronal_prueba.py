import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetworkDeep:
    """
    Implementación completa de red neuronal profunda desde cero
    """
    
    def __init__(self, layer_dims, activation='relu', output_activation='sigmoid'):
        """
        Args:
            layer_dims: Lista de dimensiones [input, hidden1, hidden2, ..., output]
            activation: Función de activación ('relu', 'tanh', 'sigmoid')
            output_activation: Activación de salida ('sigmoid', 'softmax', 'linear')
        """
        self.layer_dims = layer_dims
        self.activation = activation
        self.output_activation = output_activation
        self.parameters = {}
        self.cache = {}
        self.gradients = {}
        
        self._initialize_parameters()
        self._initialize_optimizers()
    
    def _initialize_parameters(self):
        """Inicialización Xavier/He"""
        np.random.seed(42)
        
        for l in range(1, len(self.layer_dims)):
            # Inicialización He para ReLU
            if self.activation == 'relu':
                self.parameters[f'W{l}'] = np.random.randn(self.layer_dims[l-1], self.layer_dims[l]) * np.sqrt(2/self.layer_dims[l-1])
            else:
                self.parameters[f'W{l}'] = np.random.randn(self.layer_dims[l-1], self.layer_dims[l]) * 0.01
            
            self.parameters[f'b{l}'] = np.zeros((1, self.layer_dims[l]))
    
    def _initialize_optimizers(self):
        """Inicializa parámetros para optimizadores avanzados"""
        self.optimizer_cache = {}
        for l in range(1, len(self.layer_dims)):
            self.optimizer_cache[f'dW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
            self.optimizer_cache[f'db{l}'] = np.zeros_like(self.parameters[f'b{l}'])
            self.optimizer_cache[f'vW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
            self.optimizer_cache[f'vb{l}'] = np.zeros_like(self.parameters[f'b{l}'])
            self.optimizer_cache[f'sW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
            self.optimizer_cache[f'sb{l}'] = np.zeros_like(self.parameters[f'b{l}'])
    
    def _activation_function(self, Z):
        """Aplica función de activación"""
        if self.activation == 'relu':
            return np.maximum(0, Z)
        elif self.activation == 'tanh':
            return np.tanh(Z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        else:
            return Z
    
    def _activation_derivative(self, Z):
        """Derivada de la función de activación"""
        if self.activation == 'relu':
            return (Z > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - np.tanh(Z) ** 2
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
            return s * (1 - s)
        else:
            return np.ones_like(Z)
    
    def _output_activation(self, Z):
        """Aplica activación de salida"""
        if self.output_activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        elif self.output_activation == 'softmax':
            exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        else:  # linear
            return Z
    
    def _output_activation_derivative(self, Z):
        """Derivada de la activación de salida"""
        if self.output_activation == 'sigmoid':
            s = self._output_activation(Z)
            return s * (1 - s)
        elif self.output_activation == 'softmax':
            # Para softmax, la derivada es más compleja
            s = self._output_activation(Z)
            return s * (1 - s)  # Aproximación
        else:
            return np.ones_like(Z)
    
    def forward(self, X):
        """Forward pass completo"""
        self.cache['A0'] = X
        A_prev = X
        
        # Capas ocultas
        for l in range(1, len(self.layer_dims) - 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            Z = A_prev @ W + b
            A = self._activation_function(Z)
            
            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A
            A_prev = A
        
        # Capa de salida
        L = len(self.layer_dims) - 1
        W = self.parameters[f'W{L}']
        b = self.parameters[f'b{L}']
        
        Z = A_prev @ W + b
        A = self._output_activation(Z)
        
        self.cache[f'Z{L}'] = Z
        self.cache[f'A{L}'] = A
        
        return A
    
    def compute_loss(self, y_pred, y_true, epsilon=1e-15):
        """Calcula la pérdida (entropía cruzada o MSE)"""
        m = y_true.shape[0]
        
        if self.output_activation == 'sigmoid' or self.output_activation == 'softmax':
            # Entropía cruzada
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # MSE
            loss = np.mean((y_pred - y_true) ** 2)
        
        return loss
    
    def backward(self, y_pred, y_true):
        """Backward pass con backpropagation"""
        m = y_true.shape[0]
        L = len(self.layer_dims) - 1
        
        # Gradiente de la capa de salida
        if self.output_activation == 'sigmoid' or self.output_activation == 'softmax':
            dA = -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
        else:
            dA = 2 * (y_pred - y_true)
        
        # Propagar hacia atrás
        for l in range(L, 0, -1):
            Z = self.cache[f'Z{l}']
            
            if l == L:
                dZ = dA * self._output_activation_derivative(Z)
            else:
                dZ = dA * self._activation_derivative(Z)
            
            A_prev = self.cache[f'A{l-1}']
            self.gradients[f'dW{l}'] = (A_prev.T @ dZ) / m
            self.gradients[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True) / m
            
            dA = dZ @ self.parameters[f'W{l}'].T
    
    def update_parameters(self, learning_rate, optimizer='sgd', 
                         beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Actualiza parámetros usando diferentes optimizadores"""
        
        if optimizer == 'sgd':
            # SGD simple
            for l in range(1, len(self.layer_dims)):
                self.parameters[f'W{l}'] -= learning_rate * self.gradients[f'dW{l}']
                self.parameters[f'b{l}'] -= learning_rate * self.gradients[f'db{l}']
        
        elif optimizer == 'momentum':
            # SGD con momentum
            for l in range(1, len(self.layer_dims)):
                self.optimizer_cache[f'vW{l}'] = beta1 * self.optimizer_cache[f'vW{l}'] + (1 - beta1) * self.gradients[f'dW{l}']
                self.optimizer_cache[f'vb{l}'] = beta1 * self.optimizer_cache[f'vb{l}'] + (1 - beta1) * self.gradients[f'db{l}']
                
                self.parameters[f'W{l}'] -= learning_rate * self.optimizer_cache[f'vW{l}']
                self.parameters[f'b{l}'] -= learning_rate * self.optimizer_cache[f'vb{l}']
        
        elif optimizer == 'adam':
            # Adam
            for l in range(1, len(self.layer_dims)):
                # Momentum
                self.optimizer_cache[f'vW{l}'] = beta1 * self.optimizer_cache[f'vW{l}'] + (1 - beta1) * self.gradients[f'dW{l}']
                self.optimizer_cache[f'vb{l}'] = beta1 * self.optimizer_cache[f'vb{l}'] + (1 - beta1) * self.gradients[f'db{l}']
                
                # RMSprop
                self.optimizer_cache[f'sW{l}'] = beta2 * self.optimizer_cache[f'sW{l}'] + (1 - beta2) * (self.gradients[f'dW{l}'] ** 2)
                self.optimizer_cache[f'sb{l}'] = beta2 * self.optimizer_cache[f'sb{l}'] + (1 - beta2) * (self.gradients[f'db{l}'] ** 2)
                
                # Corrección de sesgo
                vW_corrected = self.optimizer_cache[f'vW{l}'] / (1 - beta1 ** 5)
                vb_corrected = self.optimizer_cache[f'vb{l}'] / (1 - beta1 ** 5)
                sW_corrected = self.optimizer_cache[f'sW{l}'] / (1 - beta2 ** 5)
                sb_corrected = self.optimizer_cache[f'sb{l}'] / (1 - beta2 ** 5)
                
                # Actualización
                self.parameters[f'W{l}'] -= learning_rate * vW_corrected / (np.sqrt(sW_corrected) + epsilon)
                self.parameters[f'b{l}'] -= learning_rate * vb_corrected / (np.sqrt(sb_corrected) + epsilon)
    
    def train(self, X, y, epochs=1000, batch_size=32, learning_rate=0.01,
              optimizer='adam', verbose=False):
        """Entrenamiento completo"""
        m = X.shape[0]
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            # Mini-batch
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward
                y_pred = self.forward(X_batch)
                
                # Loss
                loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += loss * len(X_batch)
                
                # Backward
                self.backward(y_pred, y_batch)
                
                # Update
                self.update_parameters(learning_rate, optimizer)
            
            avg_loss = epoch_loss / m
            losses.append(avg_loss)
            
            # Precisión
            y_pred = self.forward(X)
            if self.output_activation == 'sigmoid':
                y_pred_class = (y_pred > 0.5).astype(int)
            elif self.output_activation == 'softmax':
                y_pred_class = np.argmax(y_pred, axis=1)
                y = np.argmax(y, axis=1)
            else:
                y_pred_class = np.round(y_pred).astype(int)
            
            if len(y.shape) > 1 and y.shape[1] > 1:
                y_true = np.argmax(y, axis=1)
            else:
                y_true = y.flatten()
            
            accuracy = np.mean(y_pred_class.flatten() == y_true)
            accuracies.append(accuracy)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}, Accuracy = {accuracy:.4f}")
        
        return losses, accuracies

# Ejemplo práctico: Clasificación de moons
np.random.seed(42)

# Generar datos no lineales
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape y para formato adecuado
y_train_reshaped = y_train.reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)

# Crear y entrenar red neuronal
nn = NeuralNetworkDeep([2, 64, 64, 32, 1], activation='relu', output_activation='sigmoid')
losses, accuracies = nn.train(X_train_scaled, y_train_reshaped, 
                            epochs=500, batch_size=32, 
                            learning_rate=0.01, optimizer='adam', verbose=True)

# Evaluar
y_pred_proba = nn.forward(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)
accuracy = np.mean(y_pred.flatten() == y_test)
print(f"\nPrecisión en test: {accuracy:.4f}")

# Visualizar resultados
def plot_training_results(losses, accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Pérdida de Entrenamiento')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(accuracies)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Precisión de Entrenamiento')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_training_results(losses, accuracies)

# Visualizar frontera de decisión
def plot_decision_boundary_nn(nn, X, y, title='Frontera de Decisión'):
    plt.figure(figsize=(10, 8))
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.forward(X_grid)
    Z = (Z > 0.5).astype(int).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.6, label='Clase 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.6, label='Clase 1')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

plot_decision_boundary_nn(nn, X_test_scaled, y_test)