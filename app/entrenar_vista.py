# vistas_contenido.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QSizePolicy, QStackedWidget,
                             QScrollArea)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QPixmap, QColor
import os
from src.config.settings import Settings
from entrenamiento_vistas.vista_ml_entrenamiento import VistaMLEntrenamiento
from entrenamiento_vistas.vista_dl_entrenamiento import VistaDLEntrenamiento

class ModelImageIndicator(QLabel):
    """Widget que muestra la imagen del modelo (ML o DL)"""
    def __init__(self, indicator_type="ml", parent=None):
        super().__init__(parent)
        self.indicator_type = indicator_type
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)

        self._original_pixmap = None
        self._aspect_ratio = 1.0
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        self.load_image()

    def load_image(self):
        base_path = Settings.BASE_DIR
        icons_dir = os.path.join(base_path, "icons_png")

        if self.indicator_type == "ml":
            image_path = os.path.join(icons_dir, "ML.png")
        else:
            image_path = os.path.join(icons_dir, "DL.png")

        if os.path.exists(image_path):
            self._original_pixmap = QPixmap(image_path)
            if not self._original_pixmap.isNull() and self._original_pixmap.width() > 0:
                self._aspect_ratio = self._original_pixmap.height() / float(self._original_pixmap.width())
            else:
                self._original_pixmap = None
                self._aspect_ratio = 1.0
            # print(f"Imagen original cargada: {image_path}, Aspect Ratio: {self._aspect_ratio}")
        else:
            self._original_pixmap = None
            self._aspect_ratio = 1.0
            # print(f"Advertencia: No se encontró la imagen {image_path}")

        if not self._original_pixmap:
             self.setText(f"{self.indicator_type.upper()}")
             self.setObjectName("CardModelIndicator") # Añadir objectName
        
        self.updateGeometry()

    def hasHeightForWidth(self):
        return self._original_pixmap is not None and not self._original_pixmap.isNull()

    def heightForWidth(self, width):
        if self.hasHeightForWidth():
            return int(width * self._aspect_ratio)
        return super().heightForWidth(width)

    def sizeHint(self):
        if self.hasHeightForWidth():
            base_width = 120 # Aumentado un poco para un mejor tamaño inicial
            if self._original_pixmap.width() < base_width and self._original_pixmap.width() > 0 : # Evitar base_width = 0
                base_width = self._original_pixmap.width()
            elif self._original_pixmap.width() == 0 and self._original_pixmap.height() > 0: # Caso imagen solo altura (raro)
                 return QSize(int(self._original_pixmap.height() / self._aspect_ratio if self._aspect_ratio != 0 else 50), self._original_pixmap.height())

            return QSize(base_width, int(base_width * self._aspect_ratio))
        elif self.text():
            return super().sizeHint()
        return QSize(80, 80) # Aumentado un poco

    def update_pixmap(self):
        if self.hasHeightForWidth():
            scaled_pixmap = self._original_pixmap.scaled(self.size(),
                                                         Qt.AspectRatioMode.KeepAspectRatio,
                                                         Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.hasHeightForWidth():
            self.update_pixmap()

    def set_progress(self, value):
        pass

class TrainingOptionCard(QFrame):
    option_clicked = pyqtSignal(str)

    def __init__(self, option_type="ml", parent=None):
        super().__init__(parent)
        self.option_type = option_type
        self.setObjectName("TrainingOptionCard")
        # Cambiado: La tarjeta ahora prefiere su sizeHint horizontalmente,
        # y se expande verticalmente si hay espacio.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        self.setMinimumWidth(280) # Ancho mínimo
        self.setMinimumHeight(330) # Reducida la altura mínima de la tarjeta

        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(10) # Reducido el espaciado

        self.indicator = ModelImageIndicator(option_type)
        layout.addWidget(self.indicator, 1, Qt.AlignmentFlag.AlignCenter)

        if option_type == "ml":
            button_text = "Entrenar"
            button_color = "#3498DB"
        else:
            button_text = "Entrenar"
            button_color = "#E74C3C"

        self.main_button = QPushButton(button_text)
        self.main_button.setObjectName("CardMainButton") # Añadir objectName
        self.main_button.setProperty("optionType", self.option_type) # Añadir propiedad dinámica
        self.main_button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.main_button.clicked.connect(lambda: self.option_clicked.emit(self.option_type))
        layout.addWidget(self.main_button, 0, Qt.AlignmentFlag.AlignCenter)

        self.config_button = QPushButton("Más Opciones") # Texto de ejemplo
        self.config_button.setObjectName("CardConfigButton") # Añadir objectName
        self.config_button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.config_button, 0, Qt.AlignmentFlag.AlignCenter)

        if option_type == "ml":
            desc_text = "Algoritmos tradicionales para clasificación de CVs y análisis predictivo."
        else:
            desc_text = "Redes neuronales profundas para análisis avanzado y generación de perfiles."

        self.description_label = QLabel(desc_text)
        self.description_label.setObjectName("CardDescriptionLabel") # Añadir objectName
        self.description_label.setWordWrap(True)
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        layout.addWidget(self.description_label, 0, Qt.AlignmentFlag.AlignCenter)

class seleccion(QWidget):
    def __init__(self, parent_window=None, parent=None):
        super().__init__(parent)
        self.main_window_ref = parent_window
        # self.setObjectName("VistaMejorar") # Eliminar objectName innecesario
        
        # Crear el QStackedWidget
        self.stack = QStackedWidget()

        # Crear la vista principal de selección y sus layouts alternativos para las tarjetas
        self.vista_seleccion = QWidget()
        self.vista_seleccion.setObjectName("VistaSeleccionContenido")
        
        self.cards_container = QWidget() # Contenedor para las tarjetas
        self.ml_card = TrainingOptionCard("ml")
        self.dl_card = TrainingOptionCard("dl")

        self.cards_layout_h = QHBoxLayout()
        self.cards_layout_h.setContentsMargins(10, 0, 10, 0)
        self.cards_layout_h.setSpacing(20) # Reducido el espaciado
        self.cards_layout_h.addStretch(1)
        self.cards_layout_h.addWidget(self.ml_card)
        self.cards_layout_h.addWidget(self.dl_card)
        self.cards_layout_h.addStretch(1)

        self.cards_layout_v = QVBoxLayout()
        self.cards_layout_v.setContentsMargins(10, 0, 10, 0)
        self.cards_layout_v.setSpacing(25) # Espaciado vertical entre tarjetas
        self.cards_layout_v.setAlignment(Qt.AlignmentFlag.AlignCenter) # Centrar tarjetas horizontalmente
        # Los widgets se añadirán dinámicamente

        self.current_cards_layout_is_horizontal = True
        self.cards_container.setLayout(self.cards_layout_h) # Layout inicial

        self.setup_vista_seleccion(self.vista_seleccion) # Pasar vista_seleccion para configurar su contenido

        # Inicializar las vistas de entrenamiento
        self.vista_ml = VistaMLEntrenamiento(self.main_window_ref) # Pasar la referencia de MainWindow
        self.vista_dl = VistaDLEntrenamiento(self.main_window_ref) # Pasar la referencia de MainWindow
        
        # Conectar señales de volver
        self.vista_ml.volver_solicitado.connect(lambda: self.stack.setCurrentIndex(0))
        self.vista_dl.volver_solicitado.connect(lambda: self.stack.setCurrentIndex(0))
        
        # La vista_seleccion se añade directamente, sin QScrollArea
        # self.scroll_area_seleccion = QScrollArea()
        # self.scroll_area_seleccion.setWidget(self.vista_seleccion)
        # self.scroll_area_seleccion.setWidgetResizable(True)
        # self.scroll_area_seleccion.setFrameShape(QFrame.Shape.NoFrame)

        # Envolver las vistas de entrenamiento ML y DL en QScrollArea
        scroll_ml = QScrollArea()
        scroll_ml.setWidget(self.vista_ml)
        scroll_ml.setWidgetResizable(True)
        scroll_ml.setFrameShape(QFrame.Shape.NoFrame)

        scroll_dl = QScrollArea()
        scroll_dl.setWidget(self.vista_dl)
        scroll_dl.setWidgetResizable(True)
        scroll_dl.setFrameShape(QFrame.Shape.NoFrame)
        
        # Agregar widgets/vistas al stack
        self.stack.addWidget(self.vista_seleccion)        # índice 0
        self.stack.addWidget(scroll_ml)                   # índice 1
        self.stack.addWidget(scroll_dl)                   # índice 2
        
        # Layout principal
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.stack)
        
        # Establecer un tamaño mínimo y preferido para la ventana
        self.setMinimumSize(800, 600) # Ajustar según sea necesario
        # self.resize(1000, 800) # El tamaño inicial puede ser manejado por la ventana principal

    def setup_vista_seleccion(self, vista_para_setup):
        """Configura la vista de selección"""
        layout = QVBoxLayout(vista_para_setup) # Usar el widget pasado
        layout.setContentsMargins(15, 5, 15, 15) # Reducido margen superior
        layout.setSpacing(5) # Reducido espaciado general

        title_label = QLabel("🧠 Entrenamiento de Modelos de IA 🧠")
        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setObjectName("TituloEntrenamiento")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        subtitle_label = QLabel("Selecciona el tipo de modelo que deseas entrenar o configurar")
        subtitle_font = QFont()
        subtitle_font.setPointSize(14)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setObjectName("SubtituloEntrenamiento")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle_label)
        layout.addSpacing(10) # Espacio adicional después del subtítulo

        # Conectar señales de las tarjetas (ya creadas en __init__)
        self.ml_card.option_clicked.connect(self.handle_training_option)
        self.dl_card.option_clicked.connect(self.handle_training_option)

        # Añadir el cards_container (que ya tiene el layout horizontal por defecto)
        layout.addWidget(self.cards_container)
        layout.addStretch(1)

        description_label = QLabel(
            "Los algoritmos de Machine Learning son ideales para clasificación básica de CVs, "
            "mientras que Deep Learning ofrece análisis más profundo y detallado de perfiles profesionales. "
            "Selecciona una opción para continuar."
        )
        description_label.setWordWrap(True)
        description_label.setObjectName("InstruccionEntrenamiento") # Añadir objectName
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description_label)

    def handle_training_option(self, option_type):
        if option_type == "ml":
            print("Iniciando entrenamiento de Machine Learning...")
            self.stack.setCurrentIndex(1)  # Cambiar a la vista ML
        elif option_type == "dl":
            print("Iniciando entrenamiento de Deep Learning...")
            self.stack.setCurrentIndex(2)  # Cambiar a la vista DL

    def start_ml_training(self):
        self.ml_card.main_button.setText("Entrenando...")
        self.ml_card.main_button.setEnabled(False)

    def start_dl_configuration(self):
        self.dl_card.main_button.setText("Entrenando...")
        self.dl_card.main_button.setEnabled(False)

class VistaHerramientas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("🛠️ Contenido de Herramientas (Próximamente) 🛠️")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = label.font(); font.setPointSize(16); label.setFont(font)
        layout.addWidget(label)

class VistaCentroAccion(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("🔔 Contenido del Centro de Acción (Próximamente) 🔔")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = label.font(); font.setPointSize(16); label.setFont(font)
        layout.addWidget(label)
