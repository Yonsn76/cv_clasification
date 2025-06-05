"""
Configuración para el modelo Logistic Regression.
Define los hiperparámetros óptimos para el entrenamiento del modelo.
"""

# Hiperparámetros para Logistic Regression
logistic_regression_params = {
    "C": 1.0,                   # Inverso de la fuerza de regularización
    "penalty": "l2",            # Tipo de penalización (l1, l2, elasticnet, none)
    "solver": "lbfgs",          # Algoritmo a usar para la optimización
    "max_iter": 100,            # Número máximo de iteraciones
    "tol": 1e-4,                # Tolerancia para el criterio de parada
    "random_state": 42          # Semilla para reproducibilidad
}
