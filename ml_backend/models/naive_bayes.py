"""
Implementación del modelo Naive Bayes.
Utiliza la clase base ModelHandler para entrenamiento y evaluación.
"""

from sklearn.naive_bayes import GaussianNB
from ..config.naive_bayes_config import naive_bayes_params
from .model_handler import ModelHandler

class NaiveBayesModel(ModelHandler):
    def __init__(self):
        """
        Inicializa el modelo Naive Bayes con los hiperparámetros definidos.
        """
        super().__init__(GaussianNB, naive_bayes_params)
