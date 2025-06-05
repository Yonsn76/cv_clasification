# vista_centro_accion.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QScrollArea)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor # QColor para los iconos

class VistaCentroAccion(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VistaCentroAccion") # Para QSS si es necesario

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20) # Márgenes generales
        main_layout.setSpacing(15) # Espacio entre título y scroll area

        title_label = QLabel("📄 Clasificación de CVs 📄")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Puedes aplicar estilo directamente o mediante la hoja de estilos principal
        title_label.setStyleSheet("color: #E0E0E0; margin-bottom: 10px;")
        main_layout.addWidget(title_label)

        # Contenedor para las notificaciones con scroll
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        # Aplicar estilo al QScrollArea directamente o mediante QSS global
        scroll_area.setObjectName("AccionScrollArea")
        scroll_area.setStyleSheet("""
            #AccionScrollArea { 
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
                height: 0px; /* Oculta las flechas de línea */
            }
            /* Para ocultar completamente las flechas si es necesario (aunque lo anterior suele bastar)
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                background: none;
                border: none;
                height: 0px;
            }
            */
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        
        actions_widget_container = QWidget()
        actions_layout = QVBoxLayout(actions_widget_container)
        actions_layout.setSpacing(20)
        actions_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Sección de carga de CVs
        upload_section = self.create_upload_section()
        actions_layout.addWidget(upload_section)

        # Sección de clasificación en tiempo real
        classification_section = self.create_classification_section()
        actions_layout.addWidget(classification_section)

        # Sección de resultados
        results_section = self.create_results_section()
        actions_layout.addWidget(results_section)

        actions_layout.addStretch(1)
        scroll_area.setWidget(actions_widget_container)
        main_layout.addWidget(scroll_area)

    def create_upload_section(self):
        """Crea la sección de carga de CVs"""
        section = QFrame()
        section.setObjectName("ToolFrame")
        section.setStyleSheet("""
            QFrame#ToolFrame {
                background-color: #34495E;
                border-radius: 10px;
                border: 2px solid #3498DB;
                padding: 20px;
                margin: 10px;
            }
        """)

        layout = QVBoxLayout(section)
        layout.setSpacing(15)

        # Título
        title = QLabel("📁 Cargar CVs para Clasificar")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #3498DB; margin-bottom: 10px;")
        layout.addWidget(title)

        # Área de arrastrar y soltar
        drop_area = QFrame()
        drop_area.setFixedHeight(120)
        drop_area.setStyleSheet("""
            QFrame {
                border: 2px dashed #7F8C8D;
                border-radius: 8px;
                background-color: #2C3E50;
            }
        """)

        drop_layout = QVBoxLayout(drop_area)
        drop_icon = QLabel("📄")
        drop_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_icon.setStyleSheet("font-size: 32px; color: #7F8C8D;")

        drop_text = QLabel("Arrastra archivos PDF aquí o haz clic para seleccionar")
        drop_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_text.setStyleSheet("color: #BDC3C7; font-size: 12px;")

        drop_layout.addWidget(drop_icon)
        drop_layout.addWidget(drop_text)
        layout.addWidget(drop_area)

        # Botones
        buttons_layout = QHBoxLayout()

        select_btn = QPushButton("📂 Seleccionar Archivos")
        select_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)

        clear_btn = QPushButton("🗑️ Limpiar")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #7F8C8D;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #95A5A6;
            }
        """)

        buttons_layout.addWidget(select_btn)
        buttons_layout.addWidget(clear_btn)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)

        return section

    def create_classification_section(self):
        """Crea la sección de clasificación en tiempo real"""
        section = QFrame()
        section.setObjectName("ToolFrame")
        section.setStyleSheet("""
            QFrame#ToolFrame {
                background-color: #34495E;
                border-radius: 10px;
                border: 2px solid #E74C3C;
                padding: 20px;
                margin: 10px;
            }
        """)

        layout = QVBoxLayout(section)
        layout.setSpacing(15)

        # Título
        title = QLabel("⚡ Clasificación en Tiempo Real")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #E74C3C; margin-bottom: 10px;")
        layout.addWidget(title)

        # Estado y modelo
        info_layout = QHBoxLayout()

        status_label = QLabel("Estado: Listo")
        status_label.setStyleSheet("color: #27AE60; font-weight: bold; font-size: 13px;")

        model_label = QLabel("Modelo: Random Forest CV (94.2%)")
        model_label.setStyleSheet("color: #3498DB; font-weight: bold; font-size: 13px;")

        info_layout.addWidget(status_label)
        info_layout.addStretch()
        info_layout.addWidget(model_label)
        layout.addLayout(info_layout)

        # Botón de clasificar
        classify_btn = QPushButton("🚀 Iniciar Clasificación")
        classify_btn.setFixedHeight(45)
        classify_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                border-radius: 22px;
                padding: 12px 25px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        layout.addWidget(classify_btn)

        return section

    def create_results_section(self):
        """Crea la sección de resultados"""
        section = QFrame()
        section.setObjectName("ToolFrame")
        section.setStyleSheet("""
            QFrame#ToolFrame {
                background-color: #34495E;
                border-radius: 10px;
                border: 2px solid #27AE60;
                padding: 20px;
                margin: 10px;
            }
        """)

        layout = QVBoxLayout(section)
        layout.setSpacing(15)

        # Título
        title = QLabel("📊 Resultados de Clasificación")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #27AE60; margin-bottom: 10px;")
        layout.addWidget(title)

        # Estadísticas en tarjetas
        stats_layout = QHBoxLayout()

        # Crear tarjetas de estadísticas
        stats_data = [
            {"number": "0", "label": "CVs Procesados", "color": "#3498DB"},
            {"number": "0", "label": "Clasificados", "color": "#27AE60"},
            {"number": "0%", "label": "Precisión", "color": "#E74C3C"}
        ]

        for stat in stats_data:
            stat_frame = QFrame()
            stat_frame.setStyleSheet("""
                QFrame {
                    background-color: #2C3E50;
                    border-radius: 8px;
                    padding: 15px;
                }
            """)
            stat_layout = QVBoxLayout(stat_frame)

            number_label = QLabel(stat["number"])
            number_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            number_label.setStyleSheet(f"color: {stat['color']}; font-size: 24px; font-weight: bold;")

            text_label = QLabel(stat["label"])
            text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            text_label.setStyleSheet("color: #BDC3C7; font-size: 11px;")

            stat_layout.addWidget(number_label)
            stat_layout.addWidget(text_label)
            stats_layout.addWidget(stat_frame)

        layout.addLayout(stats_layout)

        # Botones de acción
        actions_layout = QHBoxLayout()

        export_btn = QPushButton("📤 Exportar")
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #F39C12;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E67E22;
            }
        """)

        view_btn = QPushButton("👁️ Ver Detalles")
        view_btn.setStyleSheet("""
            QPushButton {
                background-color: #9B59B6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8E44AD;
            }
        """)

        actions_layout.addWidget(export_btn)
        actions_layout.addWidget(view_btn)
        actions_layout.addStretch()
        layout.addLayout(actions_layout)

        return section

    def create_classification_section(self):
        """Crea la sección de clasificación en tiempo real"""
        section = QFrame()
        section.setObjectName("ToolFrame")
        section.setStyleSheet("""
            QFrame#ToolFrame {
                background-color: #34495E;
                border-radius: 10px;
                border: 2px solid #E74C3C;
                padding: 20px;
                margin: 10px;
            }
        """)

        layout = QVBoxLayout(section)
        layout.setSpacing(15)

        # Título
        title = QLabel("⚡ Clasificación en Tiempo Real")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #E74C3C; margin-bottom: 10px;")
        layout.addWidget(title)

        # Estado de clasificación
        status_layout = QHBoxLayout()
        status_label = QLabel("Estado:")
        status_label.setStyleSheet("color: #BDC3C7; font-size: 13px;")

        status_value = QLabel("Listo para clasificar")
        status_value.setStyleSheet("color: #27AE60; font-weight: bold; font-size: 13px;")

        status_layout.addWidget(status_label)
        status_layout.addWidget(status_value)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # Modelo activo
        model_layout = QHBoxLayout()
        model_label = QLabel("Modelo activo:")
        model_label.setStyleSheet("color: #BDC3C7; font-size: 13px;")

        model_value = QLabel("Random Forest CV (94.2%)")
        model_value.setStyleSheet("color: #3498DB; font-weight: bold; font-size: 13px;")

        model_layout.addWidget(model_label)
        model_layout.addWidget(model_value)
        model_layout.addStretch()
        layout.addLayout(model_layout)

        # Botón de clasificar
        classify_btn = QPushButton("🚀 Iniciar Clasificación")
        classify_btn.setFixedHeight(45)
        classify_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                border-radius: 22px;
                padding: 12px 25px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        layout.addWidget(classify_btn)

        return section

    def create_results_section(self):
        """Crea la sección de resultados"""
        section = QFrame()
        section.setObjectName("ToolFrame")
        section.setStyleSheet("""
            QFrame#ToolFrame {
                background-color: #34495E;
                border-radius: 10px;
                border: 2px solid #27AE60;
                padding: 20px;
                margin: 10px;
            }
        """)

        layout = QVBoxLayout(section)
        layout.setSpacing(15)

        # Título
        title = QLabel("📊 Resultados de Clasificación")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #27AE60; margin-bottom: 10px;")
        layout.addWidget(title)

        # Estadísticas rápidas
        stats_layout = QHBoxLayout()

        # Total procesados
        total_frame = QFrame()
        total_frame.setStyleSheet("""
            QFrame {
                background-color: #2C3E50;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        total_layout = QVBoxLayout(total_frame)
        total_number = QLabel("0")
        total_number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        total_number.setStyleSheet("color: #3498DB; font-size: 24px; font-weight: bold;")
        total_label = QLabel("CVs Procesados")
        total_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        total_label.setStyleSheet("color: #BDC3C7; font-size: 11px;")
        total_layout.addWidget(total_number)
        total_layout.addWidget(total_label)

        # Clasificados
        classified_frame = QFrame()
        classified_frame.setStyleSheet(total_frame.styleSheet())
        classified_layout = QVBoxLayout(classified_frame)
        classified_number = QLabel("0")
        classified_number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        classified_number.setStyleSheet("color: #27AE60; font-size: 24px; font-weight: bold;")
        classified_label = QLabel("Clasificados")
        classified_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        classified_label.setStyleSheet("color: #BDC3C7; font-size: 11px;")
        classified_layout.addWidget(classified_number)
        classified_layout.addWidget(classified_label)

        # Precisión promedio
        accuracy_frame = QFrame()
        accuracy_frame.setStyleSheet(total_frame.styleSheet())
        accuracy_layout = QVBoxLayout(accuracy_frame)
        accuracy_number = QLabel("0%")
        accuracy_number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        accuracy_number.setStyleSheet("color: #E74C3C; font-size: 24px; font-weight: bold;")
        accuracy_label = QLabel("Precisión")
        accuracy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        accuracy_label.setStyleSheet("color: #BDC3C7; font-size: 11px;")
        accuracy_layout.addWidget(accuracy_number)
        accuracy_layout.addWidget(accuracy_label)

        stats_layout.addWidget(total_frame)
        stats_layout.addWidget(classified_frame)
        stats_layout.addWidget(accuracy_frame)
        layout.addLayout(stats_layout)

        # Botones de acción
        actions_layout = QHBoxLayout()

        export_btn = QPushButton("📤 Exportar Resultados")
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #F39C12;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E67E22;
            }
        """)

        view_btn = QPushButton("👁️ Ver Detalles")
        view_btn.setStyleSheet("""
            QPushButton {
                background-color: #9B59B6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8E44AD;
            }
        """)

        actions_layout.addWidget(export_btn)
        actions_layout.addWidget(view_btn)
        actions_layout.addStretch()
        layout.addLayout(actions_layout)

        return section