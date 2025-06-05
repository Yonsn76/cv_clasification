"""
Implementación del modelo Gradient Boosting.
Utiliza la clase base ModelHandler para entrenamiento y evaluación.
"""

from sklearn.ensemble import GradientBoostingClassifier
from ..config.gradient_boosting_config import gradient_boosting_params
from .model_handler import ModelHandler

class GradientBoostingModel(ModelHandler):
    def __init__(self):
        """
        Inicializa el modelo Gradient Boosting con los hiperparámetros definidos.
        """
        super().__init__(GradientBoostingClassifier, gradient_boosting_params)
