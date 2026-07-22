import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class VisualizationTools:
    """Herramientas avanzadas de visualización para IA"""
    
    def __init__(self):
        np.random.seed(42)
        self.X = np.random.randn(200, 2)
        self.y = np.random.randint(0, 2, 200)
        self.model_logs = {
            'accuracy': np.cumsum(np.random.randn(100) * 0.02 + 0.02) + 0.5,
            'loss': np.cumsum(np.random.randn(100) * 0.05 + 0.02) + 0.5
        }
    
    def plot_decision_boundary(self, model=None, X=None, y=None):
        """Visualización de fronteras de decisión"""
        if X is None:
            X = self.X
            y = self.y
        
        # Crear malla para la frontera de decisión
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Datos originales
        axes[0].scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.6, label='Clase 0')
        axes[0].scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.6, label='Clase 1')
        axes[0].set_title('Datos Originales')
        axes[0].set_xlabel('Característica 1')
        axes[0].set_ylabel('Característica 2')
        axes[0].legend()
        
        # Plot 2: Frontera de decisión (si hay modelo)
        if model is not None:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            axes[1].contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
            axes[1].scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.6, edgecolors='black')
            axes[1].scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.6, edgecolors='black')
            axes[1].set_title('Frontera de Decisión')
            axes[1].set_xlabel('Característica 1')
            axes[1].set_ylabel('Característica 2')
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self):
        """Visualización del historial de entrenamiento"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        axes[0].plot(self.model_logs['accuracy'], linewidth=2)
        axes[0].fill_between(range(len(self.model_logs['accuracy'])), 
                             0, self.model_logs['accuracy'], alpha=0.2)
        axes[0].set_title('Precisión de Entrenamiento')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(self.model_logs['loss'], color='red', linewidth=2)
        axes[1].fill_between(range(len(self.model_logs['loss'])), 
                             0, self.model_logs['loss'], alpha=0.2, color='red')
        axes[1].set_title('Pérdida de Entrenamiento')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def interactive_3d_plot(self, X=None):
        """Gráfico 3D interactivo con Plotly"""
        if X is None:
            X = np.random.randn(100, 3)
        
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Datos 3D', 'Proyección 2D'))
        
        # 3D Scatter
        fig.add_trace(
            go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2],
                        mode='markers',
                        marker=dict(size=5, opacity=0.8)),
            row=1, col=1
        )
        
        # 2D Projection
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1],
                      mode='markers',
                      marker=dict(size=8, opacity=0.7)),
            row=1, col=2
        )
        
        fig.update_layout(height=500, showlegend=False)
        fig.show()

# Ejemplo de uso
viz = VisualizationTools()
viz.plot_training_history()

# Crear y plotear frontera de decisión con un modelo simple
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(viz.X, viz.y)
viz.plot_decision_boundary(model)