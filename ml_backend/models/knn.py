"""
Implementación del modelo K-Nearest Neighbors (KNN).
Utiliza la clase base ModelHandler para entrenamiento y evaluación.
"""

from sklearn.neighbors import KNeighborsClassifier
from ..config.knn_config import knn_params
from .model_handler import ModelHandler

class KNNModel(ModelHandler):
    def __init__(self):
        """
        Inicializa el modelo KNN con los hiperparámetros definidos.
        """
        super().__init__(KNeighborsClassifier, knn_params)
