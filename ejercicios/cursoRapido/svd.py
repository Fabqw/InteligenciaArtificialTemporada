import numpy as np

# Descomposición SVD
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

U, s, Vt = np.linalg.svd(A, full_matrices=False)

print(f"Matriz original A:\n{A}")
print(f"\nValores singulares: {s}")
print(f"\nU (vectores singulares izquierdos):\n{U}")
print(f"\nVᵀ (vectores singulares derechos):\n{Vt}")

# Aplicación 1: Reducción de dimensionalidad (PCA)
k = 2  # Reducir a 2 dimensiones
A_reduced = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
print(f"\nAproximación rank-{k}:\n{A_reduced}")

# Aplicación 2: Recomendación (Filtrado colaborativo)
def collaborative_filtering(ratings_matrix, k=50):
    """
    Sistema de recomendación con SVD
    
    Args:
        ratings_matrix: Matriz usuarios × items
        k: Número de factores latentes
    Returns:
        predicted_ratings: Matriz de ratings predichos
    """
    U, s, Vt = np.linalg.svd(ratings_matrix, full_matrices=False)
    s_k = np.diag(s[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    
    return U_k @ s_k @ Vt_k

# Ejemplo: Sistema de recomendación de películas
# Matriz ficticia de 5 usuarios × 4 películas
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4]
])

predicted = collaborative_filtering(ratings, k=3)
print(f"\nRatings originales:\n{ratings}")
print(f"\nRatings predichos (k=3):\n{np.round(predicted, 2)}")