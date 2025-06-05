"""
Modelos de Machine Learning y Deep Learning
"""

from .cv_classifier import CVClassifier

# Importar Deep Learning si está disponible
try:
    from .deep_learning_classifier import DeepLearningClassifier
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

__all__ = ['CVClassifier']

if DEEP_LEARNING_AVAILABLE:
    __all__.append('DeepLearningClassifier')
