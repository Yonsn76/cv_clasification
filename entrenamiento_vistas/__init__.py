# __init__.py
"""
Paquete de vistas para entrenamiento de modelos de IA
Contiene las interfaces para Machine Learning y Deep Learning
"""

from .vista_ml_entrenamiento import VistaMLEntrenamiento
from .vista_dl_entrenamiento import VistaDLEntrenamiento

__all__ = ['VistaMLEntrenamiento', 'VistaDLEntrenamiento']
