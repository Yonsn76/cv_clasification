"""
Configuración para el modelo Gradient Boosting.
Define los hiperparámetros óptimos para el entrenamiento del modelo.
"""

# Hiperparámetros para Gradient Boosting
gradient_boosting_params = {
    "n_estimators": 100,        # Número de árboles de decisión a construir
    "learning_rate": 0.1,       # Tasa de aprendizaje
    "max_depth": 3,             # Profundidad máxima de los árboles individuales
    "min_samples_split": 2,     # Número mínimo de muestras requeridas para dividir un nodo
    "min_samples_leaf": 1,      # Número mínimo de muestras requeridas en cada hoja
    "subsample": 1.0,           # Fracción de muestras a usar para ajustar los árboles
    "random_state": 42          # Semilla para reproducibilidad
}
