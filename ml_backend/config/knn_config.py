"""
Configuración para el modelo K-Nearest Neighbors (KNN).
Define los hiperparámetros óptimos para el entrenamiento del modelo.
"""

# Hiperparámetros para KNN
knn_params = {
    "n_neighbors": 5,           # Número de vecinos a usar
    "weights": "uniform",       # Función de peso usada en la predicción (uniform, distance)
    "algorithm": "auto",        # Algoritmo usado para calcular los vecinos más cercanos
    "leaf_size": 30,            # Tamaño de hoja pasado a BallTree o KDTree
    "p": 2                      # Parámetro de potencia para la métrica de Minkowski (2 para distancia euclidiana)
}
