import numpy as np
# Dataset de salud pública (simulado)
class HealthDataAnalyzer:
    """Análisis de datos de salud con álgebra lineal"""
    
    def __init__(self):
        # Datos de pacientes: [edad, peso, presión_sistólica, presión_diastólica, glucosa]
        self.patients = np.array([
            [45, 70, 120, 80, 95],
            [50, 85, 140, 90, 120],
            [35, 65, 110, 75, 85],
            [60, 90, 150, 95, 140],
            [40, 75, 130, 85, 110],
            [55, 80, 135, 88, 125],
            [30, 60, 105, 70, 80],
            [65, 95, 160, 100, 150]
        ])
        
        self.feature_names = ['Edad', 'Peso', 'Presión Sistólica', 'Presión Diastólica', 'Glucosa']
    
    def calculate_covariance(self):
        """Calcula matriz de covarianza"""
        centered = self.patients - np.mean(self.patients, axis=0)
        return (centered.T @ centered) / (len(self.patients) - 1)
    
    def pca_analysis(self, n_components=2):
        """Análisis de componentes principales para identificar factores de riesgo"""
        # Centrar datos
        centered = self.patients - np.mean(self.patients, axis=0)
        
        # SVD para PCA
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # Proyectar a n componentes
        pca_result = centered @ Vt[:n_components].T
        
        # Valores propios (varianza explicada)
        variance_explained = (s[:n_components] ** 2) / (len(self.patients) - 1)
        
        return pca_result, variance_explained, Vt[:n_components]
    
    def identify_risk_factors(self):
        """Identifica factores de riesgo principales"""
        cov = self.calculate_covariance()
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Ordenar por importancia
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        print("\n=== ANÁLISIS DE FACTORES DE RIESGO ===")
        print("\nComponentes principales:")
        for i in range(3):
            print(f"\nPC{i+1} (Varianza: {eigenvalues[i]:.3f}):")
            for j, name in enumerate(self.feature_names):
                print(f"  {name}: {eigenvectors[j, i]:.3f}")
        
        return eigenvectors[:, :3]

# Análisis práctico
analyzer = HealthDataAnalyzer()
risk_factors = analyzer.identify_risk_factors()

# Visualización
import matplotlib.pyplot as plt

pca_result, variance, components = analyzer.pca_analysis()
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
plt.xlabel(f'PC1 (Varianza: {variance[0]:.2f})')
plt.ylabel(f'PC2 (Varianza: {variance[1]:.2f})')
plt.title('PCA de Datos de Salud: Clustering de Pacientes')
plt.grid(True, alpha=0.3)
plt.show()