"""
Funciones para validación de datos y modelos.
Incluye métodos para dividir datos en conjuntos de entrenamiento y prueba, y para validación cruzada.
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Args:
        X: Datos de características
        y: Etiquetas
        test_size (float): Proporción de datos para el conjunto de prueba
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        tuple: Conjuntos de entrenamiento y prueba (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def cross_validation(model, X, y, cv=5, scoring='accuracy'):
    """
    Realiza validación cruzada en el modelo.
    
    Args:
        model: Modelo de machine learning
        X: Datos de características
        y: Etiquetas
        cv (int): Número de pliegues para la validación cruzada
        scoring (str): Métrica de evaluación
        
    Returns:
        dict: Resultados de la validación cruzada
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'scores': scores.tolist()
    }
