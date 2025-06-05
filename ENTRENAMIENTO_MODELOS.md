# Entrenamiento de Modelos de IA - ClasificaTalento PRO

## Descripción

Se ha implementado una nueva interfaz para el entrenamiento de modelos de inteligencia artificial en la sección "Entrenamiento" de ClasificaTalento PRO. Esta interfaz permite a los usuarios elegir entre dos tipos de modelos para el análisis y clasificación de CVs.

## Características Implementadas

### 🧠 Interfaz Principal
- **Título**: "Entrenamiento de Modelos de IA"
- **Diseño**: Dos tarjetas lado a lado con indicadores circulares
- **Estilo**: Interfaz moderna con tema oscuro consistente con la aplicación

### 🤖 Opciones de Entrenamiento

#### 1. Machine Learning (ML)
- **Indicador**: Círculo azul con texto "ML"
- **Estado**: "LISTO"
- **Botón Principal**: "Entrenar" (azul)
- **Descripción**: "Algoritmos tradicionales de aprendizaje automático para clasificación de CVs"
- **Funcionalidad**: Al hacer clic, simula el inicio del proceso de entrenamiento

#### 2. Deep Learning (DL)
- **Indicador**: Círculo rojo con texto "DL"
- **Estado**: "DISPONIBLE"
- **Botón Principal**: "Configurar" (rojo)
- **Descripción**: "Redes neuronales profundas para análisis avanzado de perfiles profesionales"
- **Funcionalidad**: Al hacer clic, simula el proceso de configuración

### 🎨 Componentes Visuales

#### CircularIndicator
- Widget personalizado que muestra el progreso del entrenamiento
- Colores diferenciados por tipo de modelo
- Animación de progreso circular
- Texto central identificativo

#### TrainingOptionCard
- Tarjetas interactivas con hover effects
- Indicadores de estado dinámicos
- Botones de acción principales
- Botones de configuración secundarios
- Descripciones informativas

## Estructura de Archivos

### Archivos Modificados
- `vistas_contenido.py`: Vista principal con navegación entre interfaces
- `main_gui.py`: Integración con la aplicación principal (sin cambios necesarios)

### Nuevos Archivos Creados
- `entrenamiento_vistas/__init__.py`: Paquete de vistas de entrenamiento
- `entrenamiento_vistas/vista_ml_entrenamiento.py`: Interfaz completa para Machine Learning
- `entrenamiento_vistas/vista_dl_entrenamiento.py`: Interfaz completa para Deep Learning

### Nuevas Clases Implementadas
1. **ModelImageIndicator**: Widget que muestra las imágenes ML.png y DL.png
2. **TrainingOptionCard**: Tarjeta de opción de entrenamiento con botones "Entrenar"
3. **VistaMejorar**: Vista principal con QStackedWidget para navegación
4. **VistaMLEntrenamiento**: Interfaz completa para configurar y entrenar modelos ML
5. **VistaDLEntrenamiento**: Interfaz completa para configurar y entrenar modelos DL

## Funcionalidades Interactivas

### Feedback Visual
- **Hover Effects**: Las tarjetas cambian de color al pasar el mouse
- **Estados Dinámicos**: Los textos y botones cambian según el estado
- **Progreso Visual**: Los indicadores circulares muestran el progreso

### Simulación de Procesos
- **ML Training**: Cambia estado a "ENTRENANDO..." y muestra progreso
- **DL Configuration**: Cambia estado a "CONFIGURANDO..." y muestra progreso

## Uso

### Vista Principal
1. **Navegar**: Ir a la sección "Entrenamiento" en la barra lateral
2. **Seleccionar**: Elegir entre Machine Learning o Deep Learning
3. **Entrenar**: Hacer clic en el botón "Entrenar" de cualquier opción
4. **Observar**: Se abrirá la ventana de entrenamiento correspondiente

### Vista de Entrenamiento Machine Learning
1. **Configurar Algoritmo**: Seleccionar entre Random Forest, SVM, Logistic Regression, etc.
2. **Ajustar Parámetros**: Configurar número de estimadores, profundidad máxima, etc.
3. **Configurar Datos**: Ajustar la división entrenamiento/prueba
4. **Iniciar**: Hacer clic en "🚀 Iniciar Entrenamiento"
5. **Monitorear**: Ver el progreso y logs en tiempo real
6. **Volver**: Usar el botón "← Volver" para regresar a la vista principal

### Vista de Entrenamiento Deep Learning
1. **Configurar Arquitectura**: Seleccionar tipo de red neuronal (CNN, RNN, LSTM, etc.)
2. **Ajustar Hiperparámetros**: Learning rate, batch size, épocas, dropout, etc.
3. **Configurar Entrenamiento**: Early stopping, data augmentation, batch normalization
4. **Iniciar**: Hacer clic en "🚀 Iniciar Entrenamiento"
5. **Monitorear**: Ver métricas en tiempo real (loss, accuracy, val_loss, val_accuracy)
6. **Volver**: Usar el botón "← Volver" para regresar a la vista principal

## Extensibilidad

La implementación está diseñada para ser fácilmente extensible:

- **Nuevos Tipos de Modelos**: Agregar más opciones modificando las clases existentes
- **Procesos Reales**: Reemplazar las simulaciones con lógica real de entrenamiento
- **Configuraciones Avanzadas**: Expandir los botones de configuración
- **Métricas de Progreso**: Integrar métricas reales de entrenamiento

## Estilo y Tema

La interfaz mantiene consistencia con el tema oscuro de la aplicación:
- **Colores Primarios**: Azul (#3498DB) para ML, Rojo (#E74C3C) para DL
- **Fondo**: Gris oscuro (#34495E) para las tarjetas
- **Texto**: Blanco y grises claros para buena legibilidad
- **Efectos**: Bordes y sombras sutiles para profundidad

## Próximos Pasos

1. **Integración con Backend**: Conectar con sistemas reales de ML/DL
2. **Configuraciones Avanzadas**: Implementar ventanas de configuración detalladas
3. **Métricas en Tiempo Real**: Mostrar métricas de entrenamiento en vivo
4. **Historial de Entrenamientos**: Guardar y mostrar entrenamientos anteriores
5. **Exportación de Modelos**: Funcionalidad para exportar modelos entrenados
