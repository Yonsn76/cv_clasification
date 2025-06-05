# vistas_contenido.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QSizePolicy, QStackedWidget,
                             QScrollArea)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QPixmap, QColor
import os
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
        base_path = os.path.dirname(os.path.abspath(__file__))
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
             self.setStyleSheet("""
                QLabel {
                    background-color: #34495E;
                    color: white;
                    border-radius: 10px;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 5px;
                }
            """)
        
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
        # Cambiado: La tarjeta ahora prefiere su sizeHint, no se expande horizontalmente por defecto.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        # Podríamos establecer un ancho mínimo para que la tarjeta no sea demasiado estrecha
        self.setMinimumWidth(280) # Ajusta este valor según necesites

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QFrame#TrainingOptionCard {
                background-color: #34495E;
                border-radius: 15px;
                border: 2px solid #2C3E50;
                padding: 20px; /* Aumentado un poco el padding para mejor estética */
            }
            QFrame#TrainingOptionCard:hover {
                border: 2px solid #3498DB;
                background-color: #3C5A78;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(15) # Aumentado un poco el espaciado

        self.indicator = ModelImageIndicator(option_type)
        layout.addWidget(self.indicator, 1, Qt.AlignmentFlag.AlignCenter)

        if option_type == "ml":
            button_text = "Entrenar"
            button_color = "#3498DB"
        else:
            button_text = "Entrenar"
            button_color = "#E74C3C"

        self.main_button = QPushButton(button_text)
        self.main_button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.main_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {button_color};
                color: white;
                border: none;
                border-radius: 20px; /* Un poco más redondeado */
                font-size: 15px; /* Un poco más grande */
                font-weight: bold;
                padding: 10px 20px; /* Padding ajustado */
            }}
            QPushButton:hover {{
                background-color: {QColor(button_color).lighter(120).name()};
            }}
            QPushButton:pressed {{
                background-color: {QColor(button_color).darker(120).name()};
            }}
        """)
        self.main_button.clicked.connect(lambda: self.option_clicked.emit(self.option_type))
        layout.addWidget(self.main_button, 0, Qt.AlignmentFlag.AlignCenter)

        self.config_button = QPushButton("Más Opciones") # Texto de ejemplo
        self.config_button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.config_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #7F8C8D;
                border: 1px solid #7F8C8D;
                border-radius: 18px; /* Un poco más redondeado */
                font-size: 13px; /* Un poco más grande */
                padding: 8px 18px; /* Padding ajustado */
            }
            QPushButton:hover {
                color: #BDC3C7;
                border: 1px solid #BDC3C7;
            }
        """)
        layout.addWidget(self.config_button, 0, Qt.AlignmentFlag.AlignCenter)

        if option_type == "ml":
            desc_text = "Algoritmos tradicionales para clasificación de CVs y análisis predictivo."
        else:
            desc_text = "Redes neuronales profundas para análisis avanzado y generación de perfiles."

        self.description_label = QLabel(desc_text)
        self.description_label.setWordWrap(True)
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        self.description_label.setStyleSheet("""
            color: #95A5A6;
            font-size: 12px; /* Un poco más grande */
            margin: 10px 0px; /* Margen vertical */
            padding: 8px;
        """)
        layout.addWidget(self.description_label, 0, Qt.AlignmentFlag.AlignCenter)

class seleccion(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VistaMejorar")
        
        # Crear el QStackedWidget
        self.stack = QStackedWidget()
        
        # Crear la vista principal de selección
        self.vista_seleccion = QWidget()
        self.setup_vista_seleccion()
        
        # Inicializar las vistas de entrenamiento
        self.vista_ml = VistaMLEntrenamiento()
        self.vista_dl = VistaDLEntrenamiento()
        
        # Conectar señales de volver
        self.vista_ml.volver_solicitado.connect(lambda: self.stack.setCurrentIndex(0))
        self.vista_dl.volver_solicitado.connect(lambda: self.stack.setCurrentIndex(0))
        
        # Crear scroll areas para cada vista
        scroll_seleccion = QScrollArea()
        scroll_seleccion.setWidget(self.vista_seleccion)
        scroll_seleccion.setWidgetResizable(True)
        scroll_seleccion.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_seleccion.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_seleccion.setFrameShape(QFrame.Shape.NoFrame)

        scroll_ml = QScrollArea()
        scroll_ml.setWidget(self.vista_ml)
        scroll_ml.setWidgetResizable(True)
        scroll_ml.setFrameShape(QFrame.Shape.NoFrame)

        scroll_dl = QScrollArea()
        scroll_dl.setWidget(self.vista_dl)
        scroll_dl.setWidgetResizable(True)
        scroll_dl.setFrameShape(QFrame.Shape.NoFrame)
        
        # Agregar scroll areas al stack
        self.stack.addWidget(scroll_seleccion)  # índice 0
        self.stack.addWidget(scroll_ml)         # índice 1
        self.stack.addWidget(scroll_dl)         # índice 2
        
        # Layout principal
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.stack)
        
        # Establecer un tamaño mínimo y preferido para la ventana
        self.setMinimumSize(800, 600)
        self.resize(1000, 800)

    def setup_vista_seleccion(self):
        """Configura la vista de selección"""
        layout = QVBoxLayout(self.vista_seleccion)
        layout.setContentsMargins(25, 20, 25, 20)
        layout.setSpacing(25)

        title_label = QLabel("🧠 Entrenamiento de Modelos de IA 🧠")
        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #E0E0E0; margin-bottom: 20px;")
        layout.addWidget(title_label)

        subtitle_label = QLabel("Selecciona el tipo de modelo que deseas entrenar o configurar")
        subtitle_font = QFont()
        subtitle_font.setPointSize(14)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: #BDC3C7; margin-bottom: 30px;")
        layout.addWidget(subtitle_label)

        cards_container = QWidget()
        cards_layout = QHBoxLayout(cards_container)
        cards_layout.setContentsMargins(10, 0, 10, 0)
        cards_layout.setSpacing(40)
        
        cards_layout.addStretch(1)

        self.ml_card = TrainingOptionCard("ml")
        self.ml_card.option_clicked.connect(self.handle_training_option)
        cards_layout.addWidget(self.ml_card)

        self.dl_card = TrainingOptionCard("dl")
        self.dl_card.option_clicked.connect(self.handle_training_option)
        cards_layout.addWidget(self.dl_card)
        
        cards_layout.addStretch(1)

        layout.addWidget(cards_container)
        layout.addStretch(1)

        description_label = QLabel(
            "Los algoritmos de Machine Learning son ideales para clasificación básica de CVs, "
            "mientras que Deep Learning ofrece análisis más profundo y detallado de perfiles profesionales. "
            "Selecciona una opción para continuar."
        )
        description_label.setWordWrap(True)
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_label.setStyleSheet("""
            color: #95A5A6;
            font-size: 12px;
            margin: 20px 40px;
            padding: 15px;
            background-color: rgba(52, 73, 94, 0.3);
            border-radius: 8px;
        """)
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
        self.setStyleSheet("background-color: #2a3b4c; color: #D0D0D0;")

class VistaCentroAccion(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("🔔 Contenido del Centro de Acción (Próximamente) 🔔")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = label.font(); font.setPointSize(16); label.setFont(font)
        layout.addWidget(label)
        self.setStyleSheet("background-color: #3c2a4c; color: #D0D0D0;")