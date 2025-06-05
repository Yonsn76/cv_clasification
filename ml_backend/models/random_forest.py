"""
Implementación del modelo Random Forest.
Utiliza la clase base ModelHandler para entrenamiento y evaluación.
"""

from sklearn.ensemble import RandomForestClassifier
from ..config.random_forest_config import random_forest_params
from .model_handler import ModelHandler

class RandomForestModel(ModelHandler):
    def __init__(self):
        """
        Inicializa el modelo Random Forest con los hiperparámetros definidos.
        """
        super().__init__(RandomForestClassifier, random_forest_params)
