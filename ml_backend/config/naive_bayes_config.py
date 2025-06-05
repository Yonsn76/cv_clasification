"""
Configuración para el modelo Naive Bayes.
Define los hiperparámetros óptimos para el entrenamiento del modelo.
"""

# Hiperparámetros para Naive Bayes (GaussianNB)
naive_bayes_params = {
    "var_smoothing": 1e-9       # Porción de la varianza más grande de todos los features que se agrega a las varianzas para estabilidad numérica
}
