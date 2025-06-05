"""
Implementación del modelo Logistic Regression.
Utiliza la clase base ModelHandler para entrenamiento y evaluación.
"""

from sklearn.linear_model import LogisticRegression
from ..config.logistic_regression_config import logistic_regression_params
from .model_handler import ModelHandler

class LogisticRegressionModel(ModelHandler):
    def __init__(self):
        """
        Inicializa el modelo Logistic Regression con los hiperparámetros definidos.
        """
        super().__init__(LogisticRegression, logistic_regression_params)
