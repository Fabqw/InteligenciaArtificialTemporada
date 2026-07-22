import numpy as np

class Matrix:
    def __init__(self, data):
        self.data = np.array(data)
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0

    def __repr__(self):
        return f"Matrix({self.data})"

class Vector:
    def __init__(self, components):
        self.components = np.array(components)
        self.dimension = len(components)

    def __add__(self, other):
        if self.dimension != other.dimension:
            raise ValueError("No se pueden sumar vectores de diferentes dimensiones.")
        return Vector(self.components + other.components)
    
    def __sub__(self, other):
        if self.dimension != other.dimension:
            raise ValueError("No se pueden restar vectores de diferentes dimensiones.")
        return Vector(self.components - other.components)
    
    def scalar_multiply(self, scalar):
        return Vector(self.components * scalar)
    
    def norm(self, p=2):
        return np.linalg.norm(self.components, ord=p)
    
    def dot(self, other):
        if self.dimension != other.dimension:
            raise ValueError("No se puede calcular el producto punto de vectores de diferentes dimensiones.")
        return np.dot(self.components, other.components)
    
    def normalize(self):
        norm = self.norm()
        if norm == 0:
            raise ValueError("No se puede normalizar un vector nulo.")
        return Vector(self.components / norm)
    
    def __repr__(self):
        return f"Vector({self.components})"
    

v1 = Vector([3, 4])
v2 = Vector([1, 2])
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v1 + v2 = {v1 + v2}")
print(f"v1 · v2 = {v1.dot(v2)}")
print(f"||v1|| = {v1.norm():.2f}")
print(f"v1 normalizado = {v1.normalize()}")

p1 = Vector([1, 2, 3])
p2 = Vector([4, 5, 6])

distancia = (p1 - p2).norm()
print(f"Distancia: {distancia:.2f}")