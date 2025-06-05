"""
Clase base para manejar modelos de machine learning.
Proporciona métodos comunes para entrenamiento, predicción y evaluación.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import pandas as pd

class ModelHandler:
    def __init__(self, model, params):
        """
        Inicializa el manejador del modelo.
        
        Args:
            model: Instancia del modelo de scikit-learn
            params (dict): Diccionario de hiperparámetros para el modelo
        """
        self.model = model(**params)
        self.metrics = {}

    def train(self, X_train, y_train):
        """
        Entrena el modelo con los datos proporcionados.
        
        Args:
            X_train: Datos de entrenamiento (características)
            y_train: Etiquetas de entrenamiento
        """
        self.model.fit(X_train, y_train)
        print("Entrenamiento completado.")

    def predict(self, X):
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            X: Datos para predecir (características)
            
        Returns:
            Predicciones del modelo
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo con los datos de prueba.
        
        Args:
            X_test: Datos de prueba (características)
            y_test: Etiquetas de prueba
            
        Returns:
            Diccionario con métricas de evaluación
        """
        predictions = self.predict(X_test)
        self.metrics['accuracy'] = accuracy_score(y_test, predictions)
        self.metrics['precision'] = precision_score(y_test, predictions, average='weighted', zero_division=0)
        self.metrics['recall'] = recall_score(y_test, predictions, average='weighted', zero_division=0)
        self.metrics['f1'] = f1_score(y_test, predictions, average='weighted', zero_division=0)
        self.metrics['classification_report'] = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        return self.metrics

    def get_metrics(self):
        """
        Obtiene las métricas de evaluación del modelo.
        
        Returns:
            Diccionario con las métricas calculadas
        """
        return self.metrics
