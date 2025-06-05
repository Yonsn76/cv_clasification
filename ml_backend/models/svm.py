"""
Implementación del modelo Support Vector Machine (SVM).
Utiliza la clase base ModelHandler para entrenamiento y evaluación.
"""

from sklearn.svm import SVC
from ..config.svm_config import svm_params
from .model_handler import ModelHandler

class SVMModel(ModelHandler):
    def __init__(self):
        """
        Inicializa el modelo SVM con los hiperparámetros definidos.
        """
        super().__init__(SVC, svm_params)
