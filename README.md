# Sistema de Clasificación de CVs

Este proyecto es una aplicación de escritorio para clasificar currículums mediante técnicas de Machine Learning y Deep Learning. La interfaz está desarrollada con PyQt6 y se complementa con APIs en Flask para funcionalidades web.

## Requisitos

- Python 3.8 o superior
- Las dependencias listadas en `requirements.txt`

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

Para iniciar la interfaz principal:

```bash
python main_gui.PY
```

También se incluyen scripts bajo `page/` para exponer una API web basada en Flask.
La API utiliza por defecto el puerto `5000`, pero puede personalizarse estableciendo la
variable de entorno `API_PORT` antes de ejecutarla.

## Estructura del proyecto

- `main_gui.PY` – ventana principal de la aplicación.
- `entrenamiento_vistas/` – vistas para entrenar modelos.
- `models/` – lógica de entrenamiento y gestión de modelos ML/DL.
- `page/` – scripts y recursos para la API y páginas auxiliares.
- `docs/` – documentación HTML extendida.

## Contribución

Se aceptan mejoras mediante *pull requests*. Por favor instale las dependencias y pruebe los cambios antes de enviarlos.

