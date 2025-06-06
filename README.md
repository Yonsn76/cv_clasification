# Sistema de Clasificación de CVs

Este proyecto es una aplicación de escritorio para clasificar currículums mediante técnicas de Machine Learning y Deep Learning. La interfaz está desarrollada con PyQt6 y se complementa con APIs en Flask para funcionalidades web.

## Requisitos

- Python 3.8 o superior
- Las dependencias listadas en `requirements.txt`

- Dependencias de desarrollo en `requirements-dev.txt` (opcional)
## Instalación
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # opcional
```

## Ejecución

Para iniciar la interfaz principal:

```bash
python main_gui.py
```

También se incluyen scripts bajo `page/` para exponer una API web basada en Flask.

## Estructura del proyecto

- `main_gui.py` – ventana principal de la aplicación.
- `entrenamiento_vistas/` – vistas para entrenar modelos.
- `models/` – lógica de entrenamiento y gestión de modelos ML/DL.
- `page/` – scripts y recursos para la API y páginas auxiliares.
- `docs/` – documentación HTML extendida.
La ruta de la base de datos puede configurarse con la variable de entorno `DB_PATH`.

## Contribución

Se aceptan mejoras mediante *pull requests*. Por favor instale las dependencias y pruebe los cambios antes de enviarlos.

