# vista_herramientas.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QScrollArea)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class VistaHerramientas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VistaHerramientas") # Para QSS si es necesario

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20) # Márgenes generales
        main_layout.setSpacing(15) # Espacio entre título y scroll area

        title_label = QLabel("🤖 Gestión de Modelos de IA 🤖")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Puedes aplicar estilo directamente o mediante la hoja de estilos principal
        title_label.setStyleSheet("color: #E0E0E0; margin-bottom: 10px;") 
        main_layout.addWidget(title_label)

        # Contenedor para las herramientas con scroll
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        # Aplicar estilo al QScrollArea directamente o mediante QSS global
        scroll_area.setObjectName("HerramientasScrollArea") 
        scroll_area.setStyleSheet("""
            #HerramientasScrollArea { 
                border: none; 
                background-color: transparent; 
            }
            QScrollBar:vertical {
                border: 1px solid #2A3137;
                background: #2A3137;
                width: 12px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #4A5568;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
            }
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        
        tools_widget_container = QWidget() # Widget interno para el scroll area
        tools_layout = QVBoxLayout(tools_widget_container) # Layout para las tarjetas de herramientas
        tools_layout.setSpacing(15) # Espacio entre cada herramienta
        tools_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Las herramientas se alinean arriba

        models_data = [
            {"icon": "🤖", "name": "Random Forest CV", "desc": "Modelo de Machine Learning entrenado para clasificación de CVs con 94.2% de precisión.", "action": "Ver Detalles", "status": "Activo", "accuracy": "94.2%"},
            {"icon": "🧠", "name": "Neural Network Pro", "desc": "Red neuronal profunda especializada en análisis semántico de perfiles profesionales.", "action": "Configurar", "status": "Entrenando", "accuracy": "96.8%"},
            {"icon": "📊", "name": "SVM Classifier", "desc": "Support Vector Machine optimizado para categorización rápida de candidatos.", "action": "Activar", "status": "Inactivo", "accuracy": "91.5%"},
            {"icon": "🔍", "name": "BERT Analyzer", "desc": "Modelo transformer para análisis avanzado de texto y extracción de habilidades.", "action": "Entrenar", "status": "Disponible", "accuracy": "97.3%"},
            {"icon": "⚡", "name": "Quick Classifier", "desc": "Modelo ligero para clasificación rápida en tiempo real de grandes volúmenes.", "action": "Optimizar", "status": "Activo", "accuracy": "89.7%"}
        ]

        for model_info in models_data:
            tool_frame = QFrame()
            tool_frame.setObjectName("ToolFrame") # Para aplicar estilo QSS
            # El estilo QSS para ToolFrame se aplicará desde la hoja de estilos principal
            # o puedes definirlo aquí si esta vista no usará la global.
            # Por ahora, asumimos que se define en apply_stylesheet de MainWindow.
            
            frame_layout = QHBoxLayout(tool_frame)
            frame_layout.setSpacing(15) # Espacio interno en la tarjeta de herramienta

            icon_label = QLabel(model_info["icon"])
            icon_font = QFont(); icon_font.setPointSize(28)
            icon_label.setFont(icon_font)
            icon_label.setMinimumWidth(40)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            text_layout = QVBoxLayout()
            text_layout.setSpacing(5)

            # Nombre del modelo con estado
            name_status_layout = QHBoxLayout()
            name_label = QLabel(model_info["name"])
            name_font = QFont(); name_font.setPointSize(14); name_font.setBold(True)
            name_label.setFont(name_font)

            # Estado del modelo
            status_label = QLabel(f"• {model_info['status']}")
            status_font = QFont(); status_font.setPointSize(10)
            status_label.setFont(status_font)

            # Color según estado
            status_colors = {
                "Activo": "#27AE60",
                "Entrenando": "#F39C12",
                "Inactivo": "#7F8C8D",
                "Disponible": "#3498DB"
            }
            status_color = status_colors.get(model_info["status"], "#BDC3C7")
            status_label.setStyleSheet(f"color: {status_color}; font-weight: bold;")

            name_status_layout.addWidget(name_label)
            name_status_layout.addStretch()
            name_status_layout.addWidget(status_label)

            # Precisión
            accuracy_label = QLabel(f"Precisión: {model_info['accuracy']}")
            accuracy_label.setStyleSheet("color: #E74C3C; font-weight: bold; font-size: 11pt;")

            desc_label = QLabel(model_info["desc"])
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #BDC3C7; font-size: 11pt;")

            text_layout.addLayout(name_status_layout)
            text_layout.addWidget(accuracy_label)
            text_layout.addWidget(desc_label)
            text_layout.addStretch()

            action_button = QPushButton(model_info["action"])
            action_button.setObjectName("ToolActionButton") # Para estilo QSS específico
            action_button.setMinimumWidth(130) # Ancho mínimo del botón
            action_button.setFixedHeight(35)  # Altura fija del botón
            # El estilo QSS para ToolActionButton se aplicará desde la hoja de estilos principal.

            frame_layout.addWidget(icon_label, 0) 
            frame_layout.addLayout(text_layout, 1) 
            frame_layout.addWidget(action_button, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)

            tools_layout.addWidget(tool_frame)
        
        tools_layout.addStretch(1) # Si hay pocas herramientas, este stretch las empuja hacia arriba.
        scroll_area.setWidget(tools_widget_container) # Importante: establecer el widget con el layout
        main_layout.addWidget(scroll_area) # Añadir el QScrollArea al layout principal de la vista