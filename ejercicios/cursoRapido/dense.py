import numpy as np

class DenseLayer:
    """Implementación educativa de una capa densa"""
    
    def __init__(self, n_inputs, n_neurons):
        # Inicialización Xavier/Glorot
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2/n_inputs)
        self.bias = np.zeros(n_neurons)
        
    def forward(self, inputs):
        """
        Forward pass: y = inputs @ weights + bias
        
        Args:
            inputs: (batch_size, n_inputs)
        Returns:
            outputs: (batch_size, n_neurons)
        """
        self.inputs = inputs
        self.outputs = inputs @ self.weights + self.bias
        return self.outputs
    
    def backward(self, grad_output):
        """
        Backward pass: calcula gradientes
        
        Args:
            grad_output: (batch_size, n_neurons)
        Returns:
            grad_input: (batch_size, n_inputs)
        """
        # Gradiente de pesos
        self.grad_weights = self.inputs.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)
        
        # Gradiente de entrada para propagar hacia atrás
        grad_input = grad_output @ self.weights.T
        return grad_input
    
    def update(self, learning_rate):
        """Actualiza parámetros con SGD"""
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias

# Ejemplo de uso
layer = DenseLayer(3, 2)
x = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
output = layer.forward(x)
print(f"Input:\n{x}")
print(f"Output:\n{output}")