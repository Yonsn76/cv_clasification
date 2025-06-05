"""
Configuración para el modelo Random Forest.
Define los hiperparámetros óptimos para el entrenamiento del modelo.
"""

# Hiperparámetros para Random Forest
random_forest_params = {
    "n_estimators": 100,        # Número de árboles en el bosque
    "max_depth": None,          # Profundidad máxima de los árboles (None para crecimiento completo)
    "min_samples_split": 2,     # Número mínimo de muestras requeridas para dividir un nodo
    "min_samples_leaf": 1,      # Número mínimo de muestras requeridas en cada hoja
    "max_features": "auto",     # Número de características a considerar para la mejor división
    "bootstrap": True,          # Muestreo con reemplazo
    "random_state": 42          # Semilla para reproducibilidad
}
