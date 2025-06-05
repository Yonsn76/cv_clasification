# vistas_contenido.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QStackedWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QColor
import os

# Vistas de entrenamiento simplificadas
class VistaMLEntrenamiento(QWidget):
    volver_solicitado = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VistaMLEntrenamiento")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(20)

        # Header con botón volver
        header_layout = QHBoxLayout()

        self.btn_volver = QPushButton("← Volver")
        self.btn_volver.setFixedSize(100, 35)
        self.btn_volver.setStyleSheet("""
            QPushButton {
                background-color: #7F8C8D;
                color: white;
                border: none;
                border-radius: 17px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #95A5A6;
            }
        """)
        self.btn_volver.clicked.connect(self.volver_solicitado.emit)
        header_layout.addWidget(self.btn_volver)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # Título
        title_label = QLabel("🤖 Entrenamiento Machine Learning")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #E0E0E0; margin: 20px;")
        layout.addWidget(title_label)

        # Contenido
        content_label = QLabel("Aquí irá la interfaz completa de Machine Learning")
        content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_label.setStyleSheet("color: #BDC3C7; font-size: 14px; margin: 50px;")
        layout.addWidget(content_label)

        layout.addStretch()

class VistaDLEntrenamiento(QWidget):
    volver_solicitado = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VistaDLEntrenamiento")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(20)

        # Header con botón volver
        header_layout = QHBoxLayout()

        self.btn_volver = QPushButton("← Volver")
        self.btn_volver.setFixedSize(100, 35)
        self.btn_volver.setStyleSheet("""
            QPushButton {
                background-color: #7F8C8D;
                color: white;
                border: none;
                border-radius: 17px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #95A5A6;
            }
        """)
        self.btn_volver.clicked.connect(self.volver_solicitado.emit)
        header_layout.addWidget(self.btn_volver)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # Título
        title_label = QLabel("🧠 Entrenamiento Deep Learning")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #E0E0E0; margin: 20px;")
        layout.addWidget(title_label)

        # Contenido
        content_label = QLabel("Aquí irá la interfaz completa de Deep Learning")
        content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_label.setStyleSheet("color: #BDC3C7; font-size: 14px; margin: 50px;")
        layout.addWidget(content_label)

        layout.addStretch()

class ModelImageIndicator(QLabel):
    """Widget que muestra la imagen del modelo (ML o DL)"""
    def __init__(self, indicator_type="ml", parent=None):
        super().__init__(parent)
        self.indicator_type = indicator_type  # "ml" o "dl"
        self.setMinimumSize(100, 100)
        self.setMaximumSize(150, 150)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(True)
        
        # Cargar la imagen correspondiente
        self.load_image()
        
    def load_image(self):
        """Carga la imagen correspondiente al tipo de modelo"""
        print(f"Cargando imagen para tipo: {self.indicator_type}")
        base_path = os.path.dirname(os.path.abspath(__file__))
        icons_dir = os.path.join(base_path, "icons_png")

        if self.indicator_type == "ml":
            image_path = os.path.join(icons_dir, "ML.png")
        else:  # deep learning
            image_path = os.path.join(icons_dir, "DL.png")

        print(f"Buscando imagen en: {image_path}")
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            # Escalar la imagen manteniendo la proporción
            scaled_pixmap = pixmap.scaled(130, 130, Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
            print(f"✅ Imagen {self.indicator_type.upper()} cargada exitosamente: {image_path}")
        else:
            # Fallback: mostrar texto si no se encuentra la imagen
            self.setText(f"{self.indicator_type.upper()}")
            self.setStyleSheet("""
                QLabel {
                    background-color: #34495E;
                    color: white;
                    border-radius: 60px;
                    font-size: 16px;
                    font-weight: bold;
                }
            """)
            print(f"❌ Advertencia: No se encontró la imagen {image_path}")
    
    def set_progress(self, value):
        """Método de compatibilidad - no hace nada ya que usamos imagen estática"""
        # Este método se mantiene para compatibilidad con el código existente
        # pero no hace nada ya que ahora usamos imágenes estáticas
        pass

class TrainingOptionCard(QFrame):
    """Tarjeta de opción de entrenamiento"""
    option_clicked = pyqtSignal(str)  # Emite el tipo de entrenamiento
    
    def __init__(self, option_type="ml", parent=None):
        super().__init__(parent)
        self.option_type = option_type
        self.setObjectName("TrainingOptionCard")
        self.setMinimumSize(300, 350)
        self.setMaximumSize(400, 450)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        print(f"🔧 Creando tarjeta de entrenamiento: {option_type.upper()}")
        
        # Configurar estilo de la tarjeta
        self.setStyleSheet("""
            QFrame#TrainingOptionCard {
                background-color: #34495E;
                border-radius: 15px;
                border: 2px solid #2C3E50;
                padding: 20px;
            }
            QFrame#TrainingOptionCard:hover {
                border: 2px solid #3498DB;
                background-color: #3C5A78;
            }
        """)
        
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
        if option_type == "ml":
            button_color = "#3498DB"
        else:
            button_color = "#E74C3C"
            
        self.main_button = QPushButton(button_text)
        self.main_button.setMinimumHeight(45)
        self.main_button.setMaximumHeight(50)
        self.main_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {button_color};
                color: white;
                border: none;
                border-radius: 22px;
                font-size: 15px;
                font-weight: bold;
                padding: 12px 25px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: {QColor(button_color).lighter(120).name()};
            }}
            QPushButton:pressed {{
                background-color: {QColor(button_color).darker(120).name()};
            }}
        """)
        self.main_button.clicked.connect(lambda: self.option_clicked.emit(self.option_type))
        layout.addWidget(self.main_button)
        
        # Espaciado entre botones
        layout.addSpacing(15)
        
        # Botón de configuración (opcional)
        self.config_button = QPushButton("Configurar")
        self.config_button.setMinimumHeight(35)
        self.config_button.setMaximumHeight(40)
        self.config_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #7F8C8D;
                border: 1px solid #7F8C8D;
                border-radius: 18px;
                font-size: 13px;
                padding: 10px 20px;
                min-width: 100px;
            }
            QPushButton:hover {
                color: #BDC3C7;
                border: 1px solid #BDC3C7;
            }
        """)
        layout.addWidget(self.config_button)
        
        # Espaciado antes de la descripción
        layout.addSpacing(15)
        
        # Descripción en la parte inferior
        if option_type == "ml":
            desc_text = "Algoritmos tradicionales de aprendizaje automático para clasificación de CVs."
        else:
            desc_text = "Redes neuronales profundas para análisis avanzado de perfiles profesionales."
            
        self.description_label = QLabel(desc_text)
        self.description_label.setWordWrap(True)
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description_label.setStyleSheet("""
            color: #95A5A6;
            font-size: 12px;
            line-height: 1.4;
            margin: 10px 5px;
            padding: 12px;
        """)
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
        
        # Página 0: Vista principal de selección
        self.create_main_view()
        
        # Página 1: Vista de entrenamiento ML
        self.vista_ml = VistaMLEntrenamiento()
        self.vista_ml.volver_solicitado.connect(self.volver_a_principal)
        self.stacked_widget.addWidget(self.vista_ml)
        
        # Página 2: Vista de entrenamiento DL
        self.vista_dl = VistaDLEntrenamiento()
        self.vista_dl.volver_solicitado.connect(self.volver_a_principal)
        self.stacked_widget.addWidget(self.vista_dl)
        
        self.main_layout.addWidget(self.stacked_widget)
        
        # Mostrar la vista principal por defecto
        self.stacked_widget.setCurrentIndex(0)

    def create_main_view(self):
        """Crea la vista principal de selección"""
        main_view = QWidget()
        layout = QVBoxLayout(main_view)
        layout.setContentsMargins(25, 20, 25, 20)
        layout.setSpacing(25)

        # Título principal
        title_label = QLabel("🧠 Entrenamiento de Modelos de IA 🧠")
        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #E0E0E0; margin-bottom: 20px;")
        layout.addWidget(title_label)

        # Subtítulo
        subtitle_label = QLabel("Selecciona el tipo de modelo que deseas entrenar")
        subtitle_font = QFont()
        subtitle_font.setPointSize(14)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: #BDC3C7; margin-bottom: 30px;")
        layout.addWidget(subtitle_label)

        # Contenedor para las tarjetas
        cards_container = QWidget()
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
        
        # Agregar la vista principal al stacked widget
        self.stacked_widget.addWidget(main_view)

    def handle_training_option(self, option_type):
        """Maneja la selección de una opción de entrenamiento"""
        if option_type == "ml":
            print("Abriendo vista de entrenamiento Machine Learning...")
            self.stacked_widget.setCurrentIndex(1)  # Vista ML
        elif option_type == "dl":
            print("Abriendo vista de entrenamiento Deep Learning...")
            self.stacked_widget.setCurrentIndex(2)  # Vista DL
    
    def volver_a_principal(self):
        """Vuelve a la vista principal de selección"""
        print("Volviendo a la vista principal...")
        self.stacked_widget.setCurrentIndex(0)

# Clases de compatibilidad para las otras vistas
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
