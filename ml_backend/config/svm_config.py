"""
Configuración para el modelo Support Vector Machine (SVM).
Define los hiperparámetros óptimos para el entrenamiento del modelo.
"""

# Hiperparámetros para SVM
svm_params = {
    "C": 1.0,                   # Parámetro de regularización
    "kernel": "rbf",            # Tipo de kernel (radial basis function)
    "gamma": "scale",           # Coeficiente del kernel para 'rbf', 'poly' y 'sigmoid'
    "tol": 1e-3,                # Tolerancia para el criterio de parada
    "max_iter": -1,             # Límite de iteraciones (-1 para sin límite)
    "random_state": 42          # Semilla para reproducibilidad
}
