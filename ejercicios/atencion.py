import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

class AttentionMechanism:
    """
    Implementación educativa del mecanismo de atención
    """
    
    def __init__(self, d_model=64, num_heads=4):
        """
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de cabezas de atención
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Inicializar pesos
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Atención de producto punto escalada
        
        Args:
            Q: Query (batch, seq_len, d_k)
            K: Key (batch, seq_len, d_k)
            V: Value (batch, seq_len, d_k)
            mask: Máscara para padding
        
        Returns:
            output: Salida de atención
            attention_weights: Pesos de atención
        """
        # Calcular scores
        # Soportar tensores con multi-cabezas: Q,K,V tienen forma
        # (batch, num_heads, seq_len, d_k)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        # Aplicar máscara si existe. Acepta máscaras 2D (batch, seq_len)
        # o ya en forma broadcastable (batch, 1, 1, seq_len)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, np.newaxis, np.newaxis, :]
            scores = np.where(mask == 0, -1e9, scores)
        
        # Softmax
        attention_weights = softmax(scores, axis=-1)
        
        # Aplicar atención a los valores
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def split_heads(self, x, batch_size):
        """
        Divide la entrada en múltiples cabezas
        """
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def combine_heads(self, x, batch_size):
        """
        Combina las cabezas de atención
        """
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, -1, self.d_model)
    
    def forward(self, x, mask=None):
        """
        Forward pass de atención multi-cabeza
        
        Args:
            x: Entrada (batch, seq_len, d_model)
            mask: Máscara de atención
        
        Returns:
            output: Salida (batch, seq_len, d_model)
            attention_weights: Pesos de atención
        """
        batch_size = x.shape[0]
        
        # Proyecciones lineales
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # Dividir en cabezas
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # Atención escalada de producto punto
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combinar cabezas
        combined = self.combine_heads(attention_output, batch_size)
        
        # Proyección final
        output = np.matmul(combined, self.W_o)
        
        return output, attention_weights

# Demostración de atención
def demo_attention():
    """Demostración visual del mecanismo de atención"""
    
    # Crear secuencia de ejemplo
    seq_length = 10
    d_model = 16
    batch_size = 1
    
    # Datos sintéticos
    x = np.random.randn(batch_size, seq_length, d_model)
    
    # Crear atención
    attention = AttentionMechanism(d_model=d_model, num_heads=2)
    output, weights = attention.forward(x)
    
    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Entrada
    axes[0].imshow(x[0], aspect='auto', cmap='viridis')
    axes[0].set_title('Entrada')
    axes[0].set_xlabel('Dimensión')
    axes[0].set_ylabel('Posición')
    
    # 2. Pesos de atención (promedio de cabezas)
    avg_weights = np.mean(weights[0], axis=0)
    axes[1].imshow(avg_weights, aspect='auto', cmap='hot')
    axes[1].set_title('Pesos de Atención')
    axes[1].set_xlabel('Posición (Key)')
    axes[1].set_ylabel('Posición (Query)')
    
    # 3. Salida
    axes[2].imshow(output[0], aspect='auto', cmap='viridis')
    axes[2].set_title('Salida')
    axes[2].set_xlabel('Dimensión')
    axes[2].set_ylabel('Posición')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Forma de entrada: {x.shape}")
    print(f"Forma de salida: {output.shape}")
    print(f"Forma de pesos de atención: {weights.shape}")

demo_attention()