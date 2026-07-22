import numpy as np

A = np.array([[1, 2, 3], 
              [3, 4, 5], 
              [5, 6, 7]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [0, 2, 1]])

print(f"A + B =\n{A + B}")
print(f"A - B =\n{A - B}")
print(f"2 * A =\n{2 * A}")
print(f"A * B (element-wise) =\n{A * B}")
print(f"A @ B (producto matricial) =\n{A @ B}")
print(f"Transpuesta de A =\n{A.T}")
print(f"Determinante de A: {np.linalg.det(B):.2f}")
print(f"Inversa de A =\n{np.linalg.inv(B)}")