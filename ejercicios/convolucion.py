import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time

class Conv2DLayer:
    """
    Implementación educativa de capa convolucional 2D desde cero
    """
    
    def __init__(self, num_filters, filter_size, stride=1, padding=0, 
                 activation='relu', use_batch_norm=False):
        """
        Args:
            num_filters: Número de filtros (canales de salida)
            filter_size: Tamaño del filtro (asume cuadrado)
            stride: Paso de la convolución
            padding: Cantidad de padding (ceros)
            activation: Función de activación
            use_batch_norm: Si usar batch normalization
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        
        # Inicializar pesos (He initialization)
        self.weights = None
        self.bias = None
        
        # Para batch norm
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
        self.epsilon = 1e-8
        
        # Cache para backpropagation
        self.cache = {}
        
    def initialize_parameters(self, input_shape):
        """
        Inicializa parámetros basados en la forma de entrada
        
        Args:
            input_shape: (height, width, channels)
        """
        h, w, c = input_shape
        
        # Inicialización He para convolución
        fan_in = c * self.filter_size * self.filter_size
        self.weights = np.random.randn(
            self.num_filters, self.filter_size, self.filter_size, c
        ) * np.sqrt(2 / fan_in)
        
        self.bias = np.zeros((self.num_filters, 1))
        
        if self.use_batch_norm:
            self.gamma = np.ones((self.num_filters, 1))
            self.beta = np.zeros((self.num_filters, 1))
            self.running_mean = np.zeros((self.num_filters, 1))
            self.running_var = np.ones((self.num_filters, 1))
    
    def _pad_input(self, X):
        """Aplica padding a la entrada"""
        if self.padding == 0:
            return X
        
        pad = self.padding
        if len(X.shape) == 4:
            # (batch, height, width, channels)
            return np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 
                         mode='constant', constant_values=0)
        else:
            # (height, width, channels)
            return np.pad(X, ((pad, pad), (pad, pad), (0, 0)), 
                         mode='constant', constant_values=0)
    
    def _activation_function(self, Z):
        """Aplica función de activación"""
        if self.activation == 'relu':
            return np.maximum(0, Z)
        elif self.activation == 'tanh':
            return np.tanh(Z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        else:  # linear
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
    
    def forward(self, X, training=True):
        """
        Forward pass de convolución
        
        Args:
            X: (batch_size, height, width, channels)
            training: Si está en modo entrenamiento
        
        Returns:
            output: (batch_size, out_height, out_width, num_filters)
        """
        batch_size, h, w, c = X.shape
        
        # Inicializar parámetros si no están
        if self.weights is None:
            self.initialize_parameters((h, w, c))
        
        # Padding
        X_padded = self._pad_input(X)
        self.cache['X_padded'] = X_padded
        
        # Dimensiones de salida
        out_h = (h + 2*self.padding - self.filter_size) // self.stride + 1
        out_w = (w + 2*self.padding - self.filter_size) // self.stride + 1
        
        output = np.zeros((batch_size, out_h, out_w, self.num_filters))
        
        # Convolución
        for i in range(batch_size):
            for f in range(self.num_filters):
                for y in range(out_h):
                    for x in range(out_w):
                        # Extraer región
                        y_start = y * self.stride
                        y_end = y_start + self.filter_size
                        x_start = x * self.stride
                        x_end = x_start + self.filter_size
                        
                        region = X_padded[i, y_start:y_end, x_start:x_end, :]
                        
                        # Convolución con el filtro
                        output[i, y, x, f] = np.sum(region * self.weights[f]) + self.bias[f, 0]
        
        # Batch normalization (opcional)
        if self.use_batch_norm and training:
            output = self._batch_norm_forward(output)
        
        # Activación
        output_activated = self._activation_function(output)
        self.cache['Z'] = output
        self.cache['A'] = output_activated
        
        return output_activated
    
    def _batch_norm_forward(self, X):
        """Batch normalization forward pass"""
        batch_size = X.shape[0]
        
        # Calcular media y varianza del batch
        mean = np.mean(X, axis=(0, 1, 2), keepdims=True)
        var = np.var(X, axis=(0, 1, 2), keepdims=True)
        
        # Normalizar
        X_norm = (X - mean) / np.sqrt(var + self.epsilon)
        
        # Escalar y desplazar
        out = self.gamma * X_norm + self.beta
        
        # Guardar para backpropagation
        self.cache['bn_mean'] = mean
        self.cache['bn_var'] = var
        self.cache['bn_X_norm'] = X_norm
        
        # Actualizar running statistics
        self.running_mean = 0.9 * self.running_mean + 0.1 * mean
        self.running_var = 0.9 * self.running_var + 0.1 * var
        
        return out
    
    def _batch_norm_backward(self, grad_output):
        """Batch normalization backward pass"""
        X_norm = self.cache['bn_X_norm']
        mean = self.cache['bn_mean']
        var = self.cache['bn_var']
        
        batch_size = grad_output.shape[0]
        
        # Gradiente de gamma y beta
        dgamma = np.sum(grad_output * X_norm, axis=(0, 1, 2), keepdims=True)
        dbeta = np.sum(grad_output, axis=(0, 1, 2), keepdims=True)
        
        # Gradiente de X_normalizado
        dX_norm = grad_output * self.gamma
        
        # Gradiente de X
        dX = (1 / np.sqrt(var + self.epsilon)) * (
            dX_norm - 
            np.mean(dX_norm, axis=(0, 1, 2), keepdims=True) - 
            np.mean(dX_norm * X_norm, axis=(0, 1, 2), keepdims=True) * X_norm
        )
        
        self.cache['dgamma'] = dgamma
        self.cache['dbeta'] = dbeta
        
        return dX
    
    def backward(self, grad_output):
        """
        Backward pass de convolución
        
        Args:
            grad_output: Gradiente de la salida
        
        Returns:
            grad_input: Gradiente de la entrada
        """
        X_padded = self.cache['X_padded']
        Z = self.cache['Z']
        batch_size, out_h, out_w, _ = grad_output.shape
        
        # Gradiente a través de la activación
        dA = grad_output * self._activation_derivative(Z)
        
        # Batch normalization backward (si se usa)
        if self.use_batch_norm:
            dA = self._batch_norm_backward(dA)
        
        # Inicializar gradientes
        dW = np.zeros_like(self.weights)
        db = np.zeros_like(self.bias)
        dX_padded = np.zeros_like(X_padded)
        
        # Backward convolution
        for i in range(batch_size):
            for f in range(self.num_filters):
                for y in range(out_h):
                    for x in range(out_w):
                        y_start = y * self.stride
                        y_end = y_start + self.filter_size
                        x_start = x * self.stride
                        x_end = x_start + self.filter_size
                        
                        region = X_padded[i, y_start:y_end, x_start:x_end, :]
                        
                        # Gradiente de pesos
                        dW[f] += region * dA[i, y, x, f]
                        
                        # Gradiente de bias
                        db[f] += dA[i, y, x, f]
                        
                        # Gradiente de entrada
                        dX_padded[i, y_start:y_end, x_start:x_end, :] += (
                            dA[i, y, x, f] * self.weights[f]
                        )
        
        # Remover padding del gradiente de entrada
        if self.padding > 0:
            dX = dX_padded[:, self.padding:-self.padding, 
                          self.padding:-self.padding, :]
        else:
            dX = dX_padded
        
        # Guardar gradientes
        self.cache['dW'] = dW
        self.cache['db'] = db
        
        return dX
    
    def update_parameters(self, learning_rate):
        """Actualiza parámetros con SGD"""
        self.weights -= learning_rate * self.cache['dW']
        self.bias -= learning_rate * self.cache['db'].reshape(self.bias.shape)
        
        if self.use_batch_norm:
            self.gamma -= learning_rate * self.cache['dgamma']
            self.beta -= learning_rate * self.cache['dbeta']

class MaxPool2DLayer:
    """
    Capa de Max Pooling 2D
    """
    
    def __init__(self, pool_size, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.cache = {}
    
    def forward(self, X):
        """
        Forward pass de max pooling
        
        Args:
            X: (batch_size, height, width, channels)
        
        Returns:
            output: (batch_size, out_h, out_w, channels)
        """
        batch_size, h, w, c = X.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, out_h, out_w, c))
        max_indices = np.zeros((batch_size, out_h, out_w, c, 2), dtype=int)
        
        for i in range(batch_size):
            for y in range(out_h):
                for x in range(out_w):
                    y_start = y * self.stride
                    y_end = y_start + self.pool_size
                    x_start = x * self.stride
                    x_end = x_start + self.pool_size
                    
                    region = X[i, y_start:y_end, x_start:x_end, :]
                    
                    for ch in range(c):
                        max_val = np.max(region[:, :, ch])
                        output[i, y, x, ch] = max_val
                        
                        # Guardar índices para backpropagation
                        idx = np.where(region[:, :, ch] == max_val)
                        max_indices[i, y, x, ch] = [y_start + idx[0][0], 
                                                     x_start + idx[1][0]]
        
        self.cache['X'] = X
        self.cache['max_indices'] = max_indices
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass de max pooling
        """
        X = self.cache['X']
        max_indices = self.cache['max_indices']
        batch_size, h, w, c = X.shape
        
        grad_input = np.zeros_like(X)
        
        for i in range(batch_size):
            for y in range(grad_output.shape[1]):
                for x in range(grad_output.shape[2]):
                    for ch in range(c):
                        y_idx, x_idx = max_indices[i, y, x, ch]
                        grad_input[i, y_idx, x_idx, ch] += grad_output[i, y, x, ch]
        
        return grad_input

# Demostración de convolución
def demo_convolution():
    """Demostración visual de la operación de convolución"""
    
    # Crear imagen de prueba
    img = np.zeros((32, 32, 1))
    # Dibujar un cuadrado
    img[8:24, 8:24, 0] = 1
    
    # Crear filtro de detección de bordes
    edge_filter = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ]).reshape(3, 3, 1, 1)
    
    # Crear capa convolucional
    conv_layer = Conv2DLayer(1, 3, stride=1, padding=1, activation='linear')
    conv_layer.weights = edge_filter
    conv_layer.bias = np.zeros((1, 1))
    
    # Aplicar convolución
    img_batch = img.reshape(1, 32, 32, 1)
    output = conv_layer.forward(img_batch, training=False)
    
    # Visualizar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(img.squeeze(), cmap='gray')
    ax1.set_title('Imagen Original')
    ax1.axis('off')
    
    ax2.imshow(output.squeeze(), cmap='gray')
    ax2.set_title('Filtro de Detección de Bordes')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

demo_convolution()