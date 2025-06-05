"""
Funciones para calcular métricas de evaluación de modelos.
Incluye métricas comunes para clasificación que pueden integrarse con el frontend.
"""

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np

def calculate_confusion_matrix(y_true, y_pred):
    """
    Calcula la matriz de confusión para las predicciones.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        
    Returns:
        array: Matriz de confusión
    """
    return confusion_matrix(y_true, y_pred).tolist()

def calculate_roc_curve(y_true, y_scores):
    """
    Calcula la curva ROC y el área bajo la curva (AUC).
    
    Args:
        y_true: Etiquetas verdaderas
        y_scores: Puntuaciones de probabilidad predichas
        
    Returns:
        dict: FPR, TPR y AUC
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': roc_auc
    }

def calculate_precision_recall_curve(y_true, y_scores):
    """
    Calcula la curva de precisión-recall.
    
    Args:
        y_true: Etiquetas verdaderas
        y_scores: Puntuaciones de probabilidad predichas
        
    Returns:
        dict: Precisión, Recall y umbrales
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'thresholds': thresholds.tolist()
    }
