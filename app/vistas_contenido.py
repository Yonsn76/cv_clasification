# vistas_contenido.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QStackedWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QColor
import os
from src.config.settings import Settings

# Vistas de entrenamiento simplificadas
class VistaMLEntrenamiento(QWidget):
    volver_solicitado = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Contenido_VistaMLEntrenamiento") # Nombre de objeto

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(20)

        # Header con botón volver
        header_layout = QHBoxLayout()

        self.btn_volver = QPushButton("← Volver")
        self.btn_volver.setObjectName("Contenido_ML_BackButton") # Nombre de objeto
        self.btn_volver.setFixedSize(100, 35)
        self.btn_volver.clicked.connect(self.volver_solicitado.emit)
        header_layout.addWidget(self.btn_volver)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # Título
        title_label = QLabel("🤖 Entrenamiento Machine Learning")
        title_label.setObjectName("Contenido_ML_TitleLabel") # Nombre de objeto
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Contenido
        content_label = QLabel("Aquí irá la interfaz completa de Machine Learning")
        content_label.setObjectName("Contenido_ML_ContentLabel") # Nombre de objeto
        content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(content_label)

        layout.addStretch()

class VistaDLEntrenamiento(QWidget):
    volver_solicitado = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Contenido_VistaDLEntrenamiento") # Nombre de objeto

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(20)

        # Header con botón volver
        header_layout = QHBoxLayout()

        self.btn_volver = QPushButton("← Volver")
        self.btn_volver.setObjectName("Contenido_DL_BackButton") # Nombre de objeto
        self.btn_volver.setFixedSize(100, 35)
        self.btn_volver.clicked.connect(self.volver_solicitado.emit)
        header_layout.addWidget(self.btn_volver)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # Título
        title_label = QLabel("🧠 Entrenamiento Deep Learning")
        title_label.setObjectName("Contenido_DL_TitleLabel") # Nombre de objeto
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Contenido
        content_label = QLabel("Aquí irá la interfaz completa de Deep Learning")
        content_label.setObjectName("Contenido_DL_ContentLabel") # Nombre de objeto
        content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(content_label)

        layout.addStretch()

class ModelImageIndicator(QLabel):
    """Widget que muestra la imagen del modelo (ML o DL)"""
    def __init__(self, indicator_type="ml", parent=None):
        super().__init__(parent)
        self.indicator_type = indicator_type  # "ml" o "dl"
        self.setObjectName(f"ModelImageIndicator_{indicator_type.upper()}") # Nombre de objeto dinámico
        self.setMinimumSize(100, 100)
        self.setMaximumSize(150, 150)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(True)

        # Cargar la imagen correspondiente
        self.load_image()

    def load_image(self):
        """Carga la imagen correspondiente al tipo de modelo"""
        # print(f"Cargando imagen para tipo: {self.indicator_type}")
        base_path = Settings.BASE_DIR
        icons_dir = os.path.join(base_path, "icons_png")

        if self.indicator_type == "ml":
            image_path = os.path.join(icons_dir, "ML.png")
        else:  # deep learning
            image_path = os.path.join(icons_dir, "DL.png")

        # print(f"Buscando imagen en: {image_path}")
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            # Escalar la imagen manteniendo la proporción
            scaled_pixmap = pixmap.scaled(130, 130, Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
            # print(f"✅ Imagen {self.indicator_type.upper()} cargada exitosamente: {image_path}")
        else:
            # Fallback: mostrar texto si no se encuentra la imagen
            self.setText(f"{self.indicator_type.upper()}")
            self.setObjectName(f"ModelImageIndicator_Fallback_{indicator_type.upper()}") # Nombre de objeto para fallback
            # print(f"❌ Advertencia: No se encontró la imagen {image_path}")

    def set_progress(self, value):
        """Método de compatibilidad - no hace nada ya que usamos imagen estática"""
        pass

class TrainingOptionCard(QFrame):
    """Tarjeta de opción de entrenamiento"""
    option_clicked = pyqtSignal(str)  # Emite el tipo de entrenamiento

    def __init__(self, option_type="ml", parent=None):
        super().__init__(parent)
        self.option_type = option_type
        self.setObjectName(f"TrainingOptionCard_{option_type.upper()}") # Nombre de objeto dinámico
        self.setMinimumSize(300, 350)
        self.setMaximumSize(400, 450)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # print(f"🔧 Creando tarjeta de entrenamiento: {option_type.upper()}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 30, 25, 25)
        layout.setSpacing(25)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Indicador de imagen
        self.indicator = ModelImageIndicator(option_type)
        layout.addWidget(self.indicator, 0, Qt.AlignmentFlag.AlignCenter)

        # Espaciado después de la imagen
        layout.addSpacing(25)

        # Botón principal - ambos dicen "Entrenar"
        button_text = "Entrenar"

        self.main_button = QPushButton(button_text)
        self.main_button.setObjectName(f"TrainingCard_MainButton_{option_type.upper()}") # Nombre de objeto
        self.main_button.setMinimumHeight(45)
        self.main_button.setMaximumHeight(50)
        self.main_button.clicked.connect(lambda: self.option_clicked.emit(self.option_type))
        layout.addWidget(self.main_button)

        # Espaciado entre botones
        layout.addSpacing(15)

        # Botón de configuración (opcional)
        self.config_button = QPushButton("Configurar")
        self.config_button.setObjectName(f"TrainingCard_ConfigButton_{option_type.upper()}") # Nombre de objeto
        self.config_button.setMinimumHeight(35)
        self.config_button.setMaximumHeight(40)
        layout.addWidget(self.config_button)

        # Espaciado antes de la descripción
        layout.addSpacing(15)

        # Descripción en la parte inferior
        if option_type == "ml":
            desc_text = "Algoritmos tradicionales de aprendizaje automático para clasificación de CVs."
        else:
            desc_text = "Redes neuronales profundas para análisis avanzado de perfiles profesionales."

        self.description_label = QLabel(desc_text)
        self.description_label.setObjectName(f"TrainingCard_DescriptionLabel_{option_type.upper()}") # Nombre de objeto
        self.description_label.setWordWrap(True)
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.description_label)

        # Stretch al final para empujar todo hacia arriba
        layout.addStretch(1)

class VistaMejorar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VistaMejorar")

        # Layout principal
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Crear el stacked widget para cambiar entre vistas
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setObjectName("Mejorar_StackedWidget")

        # Página 0: Vista principal de selección
        self.create_main_view()

        # Página 1: Vista de entrenamiento ML
        self.vista_ml = VistaMLEntrenamiento() # Usa la clase definida arriba en este archivo
        self.vista_ml.volver_solicitado.connect(self.volver_a_principal)
        self.stacked_widget.addWidget(self.vista_ml)

        # Página 2: Vista de entrenamiento DL
        self.vista_dl = VistaDLEntrenamiento() # Usa la clase definida arriba en este archivo
        self.vista_dl.volver_solicitado.connect(self.volver_a_principal)
        self.stacked_widget.addWidget(self.vista_dl)

        self.main_layout.addWidget(self.stacked_widget)

        # Mostrar la vista principal por defecto
        self.stacked_widget.setCurrentIndex(0)

    def create_main_view(self):
        """Crea la vista principal de selección"""
        main_view = QWidget()
        main_view.setObjectName("Mejorar_MainViewWidget")
        layout = QVBoxLayout(main_view)
        layout.setContentsMargins(25, 20, 25, 20)
        layout.setSpacing(25)

        # Título principal
        title_label = QLabel("🧠 Entrenamiento de Modelos de IA 🧠")
        title_label.setObjectName("Mejorar_MainTitleLabel") # Nombre de objeto
        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Subtítulo
        subtitle_label = QLabel("Selecciona el tipo de modelo que deseas entrenar")
        subtitle_label.setObjectName("Mejorar_SubtitleLabel") # Nombre de objeto
        subtitle_font = QFont()
        subtitle_font.setPointSize(14)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle_label)

        # Contenedor para las tarjetas
        cards_container = QWidget()
        cards_container.setObjectName("Mejorar_CardsContainer")
        cards_layout = QHBoxLayout(cards_container)
        cards_layout.setContentsMargins(20, 0, 20, 0)
        cards_layout.setSpacing(50)
        cards_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Tarjeta Machine Learning
        self.ml_card = TrainingOptionCard("ml")
        self.ml_card.option_clicked.connect(self.handle_training_option)
        cards_layout.addWidget(self.ml_card)

        # Tarjeta Deep Learning
        self.dl_card = TrainingOptionCard("dl")
        self.dl_card.option_clicked.connect(self.handle_training_option)
        cards_layout.addWidget(self.dl_card)

        layout.addWidget(cards_container)
        layout.addStretch(1)

        # Descripción general en la parte inferior
        description_label = QLabel(
            "Los algoritmos de Machine Learning son ideales para clasificación básica de CVs, "
            "mientras que Deep Learning ofrece análisis más profundo y detallado de perfiles profesionales."
        )
        description_label.setObjectName("Mejorar_GeneralDescriptionLabel") # Nombre de objeto
        description_label.setWordWrap(True)
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description_label)

        # Agregar la vista principal al stacked widget
        self.stacked_widget.addWidget(main_view)

    def handle_training_option(self, option_type):
        """Maneja la selección de una opción de entrenamiento"""
        if option_type == "ml":
            # print("Abriendo vista de entrenamiento Machine Learning...")
            self.stacked_widget.setCurrentIndex(1)  # Vista ML (índice de cuando se añadió)
        elif option_type == "dl":
            # print("Abriendo vista de entrenamiento Deep Learning...")
            self.stacked_widget.setCurrentIndex(2)  # Vista DL (índice de cuando se añadió)

    def volver_a_principal(self):
        """Vuelve a la vista principal de selección"""
        # print("Volviendo a la vista principal...")
        self.stacked_widget.setCurrentIndex(0) # El main_view es el primer widget añadido

# Clases de compatibilidad para las otras vistas
class VistaHerramientas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VistaHerramientas") # Nombre de objeto
        layout = QVBoxLayout(self)
        label = QLabel("🛠️ Contenido de Herramientas (Próximamente) 🛠️")
        label.setObjectName("Herramientas_Label") # Nombre de objeto
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = label.font(); font.setPointSize(16); label.setFont(font)
        layout.addWidget(label)
        # self.setStyleSheet("background-color: #2a3b4c; color: #D0D0D0;") # Eliminado

class VistaCentroAccion(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VistaCentroAccion") # Nombre de objeto
        layout = QVBoxLayout(self)
        label = QLabel("🔔 Contenido del Centro de Acción (Próximamente) 🔔")
        label.setObjectName("CentroAccion_Label") # Nombre de objeto
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = label.font(); font.setPointSize(16); label.setFont(font)
        layout.addWidget(label)
        # self.setStyleSheet("background-color: #3c2a4c; color: #D0D0D0;") # Eliminado
