# vista_ml_entrenamiento.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QProgressBar, QTextEdit,
                              QComboBox, QSpinBox, QCheckBox, QGroupBox,
                              QGridLayout, QSlider, QLineEdit, QListWidget, QFileDialog, QMessageBox) # QMessageBox añadido
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QColor
import os

class VistaMLEntrenamiento(QWidget):
    """Vista para configurar y entrenar modelos de Machine Learning"""
    
    # Señales para comunicación con la vista principal
    entrenamiento_iniciado = pyqtSignal()
    entrenamiento_completado = pyqtSignal()
    volver_solicitado = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VistaMLEntrenamiento")
        self.progreso_actual = 0
        self.timer_entrenamiento = None
        
        # Referencias a widgets que necesitan actualización de colores
        self.widgets_to_update = {}
        
        self.init_ui()
        
    def init_ui(self):
        """Inicializa la interfaz de usuario"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(20)
        
        # Header con título y botón volver
        header_layout = QHBoxLayout()
        
        # Botón volver
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
        
        # Título
        title_label = QLabel("🤖 Entrenamiento Machine Learning")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Guardar referencia para actualizar el tema
        self.title_label_ml = title_label 
        header_layout.addWidget(self.title_label_ml)
        
        header_layout.addStretch()
        # No es necesario un QLabel espaciador si el título está centrado y los stretches lo manejan.
        # Considerar si el botón volver desequilibra. Si es así, se puede añadir un widget invisible del mismo ancho que el botón volver al final.
        # Por ahora, lo dejamos así, el stretch debería funcionar.
        # header_layout.addWidget(QLabel()) 
        
        # Aplicar color inicial del tema
        main_window = self.parent() # Ahora el padre es MainWindow
        if main_window and hasattr(main_window, 'color_text_light') and hasattr(main_window, 'color_text_medium') and hasattr(main_window, 'color_central_bg'):
            self.title_label_ml.setStyleSheet(f"color: {main_window.color_text_light}; margin-bottom: 10px;")
            self.btn_volver.setStyleSheet(f"""
                QPushButton {{
                    background-color: {main_window.color_text_medium if main_window.color_text_medium else '#7F8C8D'}; /* Fallback */
                    color: {main_window.color_central_bg if main_window.color_central_bg else 'white'}; /* Texto que contraste con el fondo del botón */
                    border: none;
                    border-radius: 17px;
                    font-size: 13px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {QColor(main_window.color_text_medium if main_window.color_text_medium else '#7F8C8D').lighter(110).name()};
                }}
            """)
        else:
            print("VistaMLEntrenamiento: No se pudo obtener la ventana principal o atributos de color para aplicar el tema inicial en init_ui.")
        
        layout.addLayout(header_layout)
        
        # Contenido principal en scroll
        main_content = QWidget()
        content_layout = QVBoxLayout(main_content)
        content_layout.setSpacing(25)

        # Sección de Configuración de Datos y Profesión
        self.create_data_profession_config_section(content_layout)

        # Sección para Nombre del Modelo
        self.create_model_name_section(content_layout)
        
        # Sección de configuración del algoritmo
        self.create_algorithm_section(content_layout)
        
        # Sección de parámetros
        self.create_parameters_section(content_layout)
        
        # Sección de datos
        self.create_data_section(content_layout)
        
        # Sección de progreso
        self.create_progress_section(content_layout)
        
        # Botones de acción
        self.create_action_buttons(content_layout)
        
        layout.addWidget(main_content)
        
        
    def create_model_name_section(self, parent_layout):
        """Crea la sección para ingresar el nombre del modelo"""
        group = QGroupBox("2. Nombre del Modelo a Entrenar")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
                border: 2px solid #8E44AD; /* Un color nuevo para esta sección */
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QHBoxLayout(group)
        layout.setSpacing(10)

        label_model_name = QLabel("Nombre del Modelo:")
        label_model_name.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        layout.addWidget(label_model_name)

        self.line_edit_model_name_ml = QLineEdit()
        self.line_edit_model_name_ml.setPlaceholderText("Ej: MiModeloClasificacionCV, ModeloPruebaRH")
        self.widgets_to_update['line_edit_model_name_ml'] = self.line_edit_model_name_ml # Guardar referencia
        self.line_edit_model_name_ml.setStyleSheet("""
            QLineEdit {
                background-color: #34495E;
                color: white;
                border: 1px solid #8E44AD;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        layout.addWidget(self.line_edit_model_name_ml)
        self.widgets_to_update['group_model_name'] = group # Guardar referencia del QGroupBox
        parent_layout.addWidget(group)

    def create_algorithm_section(self, parent_layout):
        """Crea la sección de selección de algoritmo"""
        group = QGroupBox("3. Algoritmo de Machine Learning") # Numeración actualizada
        self.widgets_to_update['group_algorithm'] = group # Guardar referencia del QGroupBox
        group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
                border: 2px solid #3498DB;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        
        # Selector de algoritmo
        algo_layout = QHBoxLayout()
        algo_label = QLabel("Algoritmo:")
        algo_label.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        self.widgets_to_update['algo_label_ml'] = algo_label # Guardar referencia
        
        self.combo_algoritmo = QComboBox()
        self.widgets_to_update['combo_algoritmo_ml'] = self.combo_algoritmo # Guardar referencia
        self.combo_algoritmo.addItems([
            "Random Forest",
            "Support Vector Machine (SVM)",
            "Logistic Regression",
            "Gradient Boosting",
            "K-Nearest Neighbors (KNN)",
            "Naive Bayes"
        ])
        self.combo_algoritmo.setStyleSheet("""
            QComboBox {
                background-color: #34495E;
                color: white;
                border: 1px solid #3498DB;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
            }
            QComboBox QListView {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #3498DB;
                padding: 4px; 
                outline: 0px; 
            }
            QComboBox QListView::item { 
                background-color: transparent; 
                color: #ECF0F1;
                min-height: 25px; 
                padding: 0px 5px; 
            }
            QComboBox QListView::item:selected {
                background-color: #4A6B8A;
                color: #ECF0F1;
            }
            QComboBox QListView::item:hover {
                background-color: #557CAC;
                color: #ECF0F1;
            }
        """)
        
        
        algo_layout.addWidget(algo_label)
        algo_layout.addWidget(self.combo_algoritmo)
        algo_layout.addStretch()
        
        layout.addLayout(algo_layout)
        
        # Descripción del algoritmo
        self.desc_algoritmo = QLabel("Random Forest: Ensemble de árboles de decisión, robusto y eficaz para clasificación.")
        self.desc_algoritmo.setWordWrap(True)
        self.widgets_to_update['desc_algoritmo_ml'] = self.desc_algoritmo # Guardar referencia
        self.desc_algoritmo.setStyleSheet("""
            color: #95A5A6;
            font-size: 12px;
            padding: 10px;
            background-color: rgba(52, 73, 94, 0.3);
            border-radius: 5px;
        """)
        layout.addWidget(self.desc_algoritmo)
        
        # Conectar cambio de algoritmo
        self.combo_algoritmo.currentTextChanged.connect(self.actualizar_descripcion_algoritmo)
        
        
        parent_layout.addWidget(group)

    def create_data_profession_config_section(self, parent_layout):
        """Crea la sección de configuración de datos y profesión"""
        group = QGroupBox("1. Configuración de Datos y Profesión")
        self.widgets_to_update['group_data_profession'] = group # Guardar referencia del QGroupBox
        group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
                border: 2px solid #1ABC9C; /* Color distintivo para esta sección */
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QGridLayout(group) # Corregido: usar 'group' en lugar de 'self.group_data_profession' que aún no está definido
        layout.setSpacing(15)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        # Profesión Objetivo
        self.label_profession = QLabel("Profesión Objetivo:")
        # Obtener main_window de forma segura
        main_window = self.window()
        medium_text_color = "#BDC3C7" # Valor por defecto
        if main_window and hasattr(main_window, 'color_text_medium'):
            medium_text_color = main_window.color_text_medium
        
        self.label_profession.setStyleSheet(f"color: {medium_text_color}; font-size: 13px;")
        self.widgets_to_update['label_profession_ml'] = self.label_profession # Guardar referencia
        layout.addWidget(self.label_profession, 0, 0)
        
        self.line_edit_profession_ml = QLineEdit()
        self.line_edit_profession_ml.setPlaceholderText("Ej: Ingeniero de Software")
        self.widgets_to_update['line_edit_profession_ml'] = self.line_edit_profession_ml # Guardar referencia
        self.line_edit_profession_ml.setStyleSheet("""
            QLineEdit {
                background-color: #34495E;
                color: white;
                border: 1px solid #1ABC9C;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        layout.addWidget(self.line_edit_profession_ml, 0, 1, 1, 2)

        # Carpeta de Datos
        self.label_data_folder = QLabel("Carpeta de Datos:")
        self.label_data_folder.setStyleSheet(f"color: {medium_text_color}; font-size: 13px;") # Esto necesitará ser actualizado en update_theme_colors
        self.widgets_to_update['label_data_folder_ml'] = self.label_data_folder # Guardar referencia
        layout.addWidget(self.label_data_folder, 1, 0)

        self.line_edit_selected_folder_ml = QLineEdit()
        self.line_edit_selected_folder_ml.setPlaceholderText("Seleccione una carpeta...")
        self.line_edit_selected_folder_ml.setReadOnly(True)
        self.widgets_to_update['line_edit_selected_folder_ml'] = self.line_edit_selected_folder_ml # Guardar referencia
        self.line_edit_selected_folder_ml.setStyleSheet("""
            QLineEdit {
                background-color: #2C3E50; /* Un poco más oscuro para indicar solo lectura */
                color: #BDC3C7;
                border: 1px solid #1ABC9C;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        layout.addWidget(self.line_edit_selected_folder_ml, 1, 1)

        self.btn_select_folder_ml = QPushButton("Seleccionar Carpeta...")
        self.widgets_to_update['btn_select_folder_ml'] = self.btn_select_folder_ml # Guardar referencia
        self.btn_select_folder_ml.setStyleSheet("""
            QPushButton { background-color: #3498DB; color: white; border: none; border-radius: 5px; padding: 8px 12px; font-size: 12px; font-weight: bold; }
            QPushButton:hover { background-color: #2980B9; }
        """)
        self.btn_select_folder_ml.clicked.connect(self._handle_select_folder_ml)
        layout.addWidget(self.btn_select_folder_ml, 1, 2)
        
        # Botón para agregar la asociación
        self.btn_add_profession_data_ml = QPushButton("Asociar Profesión y Carpeta")
        self.widgets_to_update['btn_add_profession_data_ml'] = self.btn_add_profession_data_ml # Guardar referencia
        self.btn_add_profession_data_ml.setStyleSheet("""
            QPushButton { background-color: #2ECC71; color: white; border: none; border-radius: 5px; padding: 10px 15px; font-size: 13px; font-weight: bold; }
            QPushButton:hover { background-color: #27AE60; }
        """)
        self.btn_add_profession_data_ml.clicked.connect(self._handle_add_profession_data_ml)
        layout.addWidget(self.btn_add_profession_data_ml, 2, 1, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)


        # Lista de Datos Asociados
        self.label_data_sources = QLabel("Datos Asociados (Profesión - Carpeta):")
        self.label_data_sources.setStyleSheet(f"color: {medium_text_color}; font-size: 13px; margin-top: 10px;") # Esto necesitará ser actualizado en update_theme_colors
        self.widgets_to_update['label_data_sources_ml'] = self.label_data_sources # Guardar referencia
        layout.addWidget(self.label_data_sources, 3, 0, 1, 3)

        self.list_widget_data_sources_ml = QListWidget()
        self.widgets_to_update['list_widget_data_sources_ml'] = self.list_widget_data_sources_ml # Guardar referencia
        self.list_widget_data_sources_ml.setStyleSheet("""
            QListWidget {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #1ABC9C;
                border-radius: 5px;
                padding: 5px;
                font-size: 12px;
                min-height: 100px; /* Aumentar altura mínima */
            }
            QListWidget::item {
                padding: 4px; /* Más padding para mejor lectura */
            }
            QListWidget::item:selected {
                background-color: #1ABC9C;
                color: #2C3E50;
            }
        """)
        layout.addWidget(self.list_widget_data_sources_ml, 4, 0, 1, 3) # Ocupa 3 columnas

        # Botón para quitar seleccionado
        self.btn_remove_source_ml = QPushButton("Quitar Seleccionado")
        self.widgets_to_update['btn_remove_source_ml'] = self.btn_remove_source_ml # Guardar referencia
        self.btn_remove_source_ml.setStyleSheet("""
            QPushButton { background-color: #E74C3C; color: white; border: none; border-radius: 5px; padding: 8px 12px; font-size: 12px; font-weight: bold; }
            QPushButton:hover { background-color: #C0392B; }
        """)
        self.btn_remove_source_ml.clicked.connect(self._handle_remove_source_ml)
        layout.addWidget(self.btn_remove_source_ml, 5, 2, alignment=Qt.AlignmentFlag.AlignRight) # Alineado a la derecha
        
        parent_layout.addWidget(group)
        
        
    def create_parameters_section(self, parent_layout):
        """Crea la sección de parámetros"""
        group = QGroupBox("4. Parámetros de Entrenamiento") # Numeración actualizada
        self.widgets_to_update['group_parameters'] = group # Guardar referencia del QGroupBox
        group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
                border: 2px solid #E74C3C;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QGridLayout(group)
        layout.setSpacing(15)
        
        # Número de estimadores
        label_estimadores = QLabel("Número de estimadores:")
        self.widgets_to_update['label_estimadores_ml'] = label_estimadores # Guardar referencia
        layout.addWidget(label_estimadores, 0, 0)
        self.spin_estimadores = QSpinBox()
        self.spin_estimadores.setRange(10, 1000)
        self.spin_estimadores.setValue(100)
        self.widgets_to_update['spin_estimadores_ml'] = self.spin_estimadores # Guardar referencia
        self.spin_estimadores.setStyleSheet("""
            QSpinBox {
                background-color: #34495E;
                color: white;
                border: 1px solid #E74C3C;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.spin_estimadores, 0, 1)
        
        # Profundidad máxima
        label_profundidad = QLabel("Profundidad máxima:")
        self.widgets_to_update['label_profundidad_ml'] = label_profundidad # Guardar referencia
        layout.addWidget(label_profundidad, 1, 0)
        self.spin_profundidad = QSpinBox()
        self.spin_profundidad.setRange(1, 50)
        self.spin_profundidad.setValue(10)
        self.widgets_to_update['spin_profundidad_ml'] = self.spin_profundidad # Guardar referencia
        self.spin_profundidad.setStyleSheet(self.spin_estimadores.styleSheet()) # El estilo se actualizará en update_theme_colors
        layout.addWidget(self.spin_profundidad, 1, 1)
        
        # Validación cruzada
        self.check_validacion = QCheckBox("Usar validación cruzada")
        self.check_validacion.setChecked(True)
        self.widgets_to_update['check_validacion_ml'] = self.check_validacion # Guardar referencia
        self.check_validacion.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        layout.addWidget(self.check_validacion, 2, 0, 1, 2)
        
        # Aplicar estilos a las etiquetas (se hará en update_theme_colors)
        # for i in range(layout.rowCount()):
        #     item = layout.itemAtPosition(i, 0)
        #     if item and item.widget() and isinstance(item.widget(), QLabel):
        #         item.widget().setStyleSheet("color: #BDC3C7; font-size: 13px;")
        
        parent_layout.addWidget(group)
        
        
    def create_data_section(self, parent_layout):
        """Crea la sección de configuración de datos"""
        group = QGroupBox("5. Configuración de Datos Adicional") # Numeración actualizada y nombre
        self.widgets_to_update['group_data_additional'] = group # Guardar referencia del QGroupBox
        group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
                border: 2px solid #F39C12;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        
        # División de datos
        split_layout = QHBoxLayout()
        split_label = QLabel("División entrenamiento/prueba:")
        split_label.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        self.widgets_to_update['split_label_ml'] = split_label # Guardar referencia
        
        self.slider_split = QSlider(Qt.Orientation.Horizontal)
        self.widgets_to_update['slider_split_ml'] = self.slider_split # Guardar referencia
        self.slider_split.setRange(60, 90)
        self.slider_split.setValue(80)
        self.slider_split.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #34495E;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #F39C12;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)
        
        self.label_split = QLabel("80%")
        self.label_split.setStyleSheet("color: #F39C12; font-weight: bold;")
        self.widgets_to_update['label_split_value_ml'] = self.label_split # Guardar referencia
        self.slider_split.valueChanged.connect(lambda v: self.label_split.setText(f"{v}%"))
        
        split_layout.addWidget(split_label)
        split_layout.addWidget(self.slider_split)
        split_layout.addWidget(self.label_split)
        
        layout.addLayout(split_layout)
        
        parent_layout.addWidget(group)
        
        
    def create_progress_section(self, parent_layout):
        """Crea la sección de progreso"""
        group = QGroupBox("6. Progreso del Entrenamiento") # Numeración actualizada
        self.widgets_to_update['group_progress'] = group # Guardar referencia del QGroupBox
        group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
                border: 2px solid #27AE60;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.widgets_to_update['progress_bar_ml'] = self.progress_bar # Guardar referencia
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #27AE60;
                border-radius: 8px;
                text-align: center;
                color: white;
                font-weight: bold;
                background-color: #2C3E50;
            }
            QProgressBar::chunk {
                background-color: #27AE60;
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Log de entrenamiento
        self.log_entrenamiento = QTextEdit()
        self.log_entrenamiento.setMaximumHeight(120)
        self.log_entrenamiento.setReadOnly(True)
        self.log_entrenamiento.setPlaceholderText("Los logs del entrenamiento aparecerán aquí...")
        self.widgets_to_update['log_entrenamiento_ml'] = self.log_entrenamiento # Guardar referencia
        self.log_entrenamiento.setStyleSheet("""
            QTextEdit {
                background-color: #2C3E50;
                color: #BDC3C7;
                border: 1px solid #27AE60;
                border-radius: 5px;
                padding: 8px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.log_entrenamiento)
        
        parent_layout.addWidget(group)
        
    def create_action_buttons(self, parent_layout):
        """Crea los botones de acción"""
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(20)
        
        # Botón iniciar entrenamiento
        self.btn_iniciar = QPushButton("🚀 Iniciar Entrenamiento")
        self.btn_iniciar.setFixedHeight(50)
        self.widgets_to_update['btn_iniciar_ml'] = self.btn_iniciar # Guardar referencia
        self.btn_iniciar.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 16px;
                font-weight: bold;
                padding: 15px 30px;
            }
            QPushButton:hover {
                background-color: #2ECC71;
            }
            QPushButton:pressed {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #7F8C8D;
            }
        """)
        self.btn_iniciar.clicked.connect(self.iniciar_entrenamiento)
        
        # Botón detener
        self.btn_detener = QPushButton("⏹ Detener")
        self.btn_detener.setFixedHeight(50)
        self.btn_detener.setEnabled(False)
        self.widgets_to_update['btn_detener_ml'] = self.btn_detener # Guardar referencia
        self.btn_detener.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 16px;
                font-weight: bold;
                padding: 15px 30px;
            }
            QPushButton:hover {
                background-color: #EC7063;
            }
            QPushButton:pressed {
                background-color: #C0392B;
            }
            QPushButton:disabled {
                background-color: #7F8C8D;
            }
        """)
        self.btn_detener.clicked.connect(self.detener_entrenamiento)
        
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.btn_iniciar)
        buttons_layout.addWidget(self.btn_detener)
        buttons_layout.addStretch()
        
        parent_layout.addLayout(buttons_layout)
        
    def actualizar_descripcion_algoritmo(self, algoritmo):
        """Actualiza la descripción según el algoritmo seleccionado"""
        descripciones = {
            "Random Forest": "Random Forest: Ensemble de árboles de decisión, robusto y eficaz para clasificación.",
            "Support Vector Machine (SVM)": "SVM: Encuentra el hiperplano óptimo para separar clases, efectivo en espacios de alta dimensión.",
            "Logistic Regression": "Regresión Logística: Modelo lineal para clasificación binaria y multiclase, interpretable y rápido.",
            "Gradient Boosting": "Gradient Boosting: Combina modelos débiles secuencialmente, muy potente pero puede sobreajustar.",
            "K-Nearest Neighbors (KNN)": "KNN: Clasifica basándose en los k vecinos más cercanos, simple pero sensible a la escala.",
            "Naive Bayes": "Naive Bayes: Basado en probabilidades con independencia condicional, rápido y efectivo con datos categóricos."
        }
        self.desc_algoritmo.setText(descripciones.get(algoritmo, "Descripción no disponible."))
        
    def iniciar_entrenamiento(self):
        """Inicia el proceso de entrenamiento simulado"""
        self.btn_iniciar.setEnabled(False)
        self.btn_detener.setEnabled(True)
        self.progreso_actual = 0
        self.progress_bar.setValue(0)
        
        # Limpiar log
        self.log_entrenamiento.clear()
        self.log_entrenamiento.append("🔄 Iniciando entrenamiento...")
        self.log_entrenamiento.append(f"📊 Algoritmo: {self.combo_algoritmo.currentText()}")
        self.log_entrenamiento.append(f"⚙️ Estimadores: {self.spin_estimadores.value()}")
        self.log_entrenamiento.append(f"📏 Profundidad máxima: {self.spin_profundidad.value()}")
        self.log_entrenamiento.append(f"📈 División datos: {self.slider_split.value()}% entrenamiento")
        self.log_entrenamiento.append("=" * 50)
        
        # Emitir señal
        self.entrenamiento_iniciado.emit()
        
        # Iniciar timer para simular progreso
        self.timer_entrenamiento = QTimer()
        self.timer_entrenamiento.timeout.connect(self.actualizar_progreso)
        self.timer_entrenamiento.start(200)  # Actualizar cada 200ms
        
    def actualizar_progreso(self):
        """Actualiza el progreso del entrenamiento"""
        import random
        
        # Incrementar progreso
        incremento = random.randint(1, 3)
        self.progreso_actual += incremento
        
        if self.progreso_actual >= 100:
            self.progreso_actual = 100
            self.finalizar_entrenamiento()
        
        self.progress_bar.setValue(self.progreso_actual)
        
        # Agregar logs simulados
        if self.progreso_actual % 10 == 0:
            accuracy = round(0.75 + (self.progreso_actual / 100) * 0.20, 3)
            self.log_entrenamiento.append(f"📊 Época {self.progreso_actual//10}: Accuracy = {accuracy}")
        
    def finalizar_entrenamiento(self):
        """Finaliza el entrenamiento"""
        if self.timer_entrenamiento:
            self.timer_entrenamiento.stop()
            
        self.btn_iniciar.setEnabled(True)
        self.btn_detener.setEnabled(False)
        
        self.log_entrenamiento.append("=" * 50)
        self.log_entrenamiento.append("✅ ¡Entrenamiento completado exitosamente!")
        self.log_entrenamiento.append("📈 Accuracy final: 0.952")
        self.log_entrenamiento.append("🎯 Precisión: 0.948")
        self.log_entrenamiento.append("🔄 Recall: 0.956")
        self.log_entrenamiento.append("💾 Modelo guardado correctamente")
        
        # Emitir señal
        self.entrenamiento_completado.emit()
        
    def detener_entrenamiento(self):
        """Detiene el entrenamiento"""
        if self.timer_entrenamiento:
            self.timer_entrenamiento.stop()
            
        self.btn_iniciar.setEnabled(True)
        self.btn_detener.setEnabled(False)
        
        self.log_entrenamiento.append("⏹ Entrenamiento detenido por el usuario")

    # --- Métodos para manejo de fuentes de datos por profesión ---
    def _handle_select_folder_ml(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de Datos para la Profesión")
        if folder_path:
            self.line_edit_selected_folder_ml.setText(folder_path)
            print(f"Carpeta seleccionada (ML): {folder_path}")
        else:
            print("Selección de carpeta cancelada (ML).")

    def _handle_add_profession_data_ml(self):
        profesion = self.line_edit_profession_ml.text().strip()
        ruta_carpeta = self.line_edit_selected_folder_ml.text().strip()

        if not profesion:
            QMessageBox.warning(self, "Campo Vacío", "Por favor, ingrese una profesión.")
            self.line_edit_profession_ml.setFocus()
            return
        
        if not ruta_carpeta:
            QMessageBox.warning(self, "Campo Vacío", "Por favor, seleccione una carpeta de datos.")
            self.btn_select_folder_ml.setFocus() # Opcional: dar foco al botón de seleccionar
            return

        # Contar archivos en la carpeta
        num_archivos = 0
        try:
            if os.path.isdir(ruta_carpeta):
                num_archivos = len([f for f in os.listdir(ruta_carpeta) if os.path.isfile(os.path.join(ruta_carpeta, f))])
        except Exception as e:
            print(f"Error al contar archivos en {ruta_carpeta}: {e}")
            QMessageBox.warning(self, "Error Carpeta", f"No se pudo acceder o contar archivos en la carpeta:\n{ruta_carpeta}\n\nError: {e}")
            # Podríamos decidir no agregar la carpeta si no se pueden contar los archivos, o agregarla con (Archivos: Error)
            # Por ahora, la agregaremos indicando el error o 0 archivos si no se pudo leer.
            # Opcionalmente, podríamos retornar aquí si es crítico no poder leer la carpeta.

        item_text = f"Profesión: {profesion}  |  Carpeta: {ruta_carpeta} (Archivos: {num_archivos})"
        
        # Verificar duplicados (considerando el nuevo formato con conteo de archivos)
        # Para una verificación robusta, podríamos parsear el texto o almacenar los datos de forma estructurada
        # Por simplicidad, si el texto exacto (incluyendo el conteo) ya existe, es un duplicado.
        # Una mejor aproximación sería verificar solo "Profesión" y "Carpeta" para duplicados lógicos.
        
        items_actuales_text = [self.list_widget_data_sources_ml.item(i).text() for i in range(self.list_widget_data_sources_ml.count())]
        
        # Chequeo de duplicado lógico (sin contar archivos, ya que el contenido puede cambiar)
        profesion_carpeta_text_base = f"Profesión: {profesion}  |  Carpeta: {ruta_carpeta}"
        for item_existente in items_actuales_text:
            if item_existente.startswith(profesion_carpeta_text_base):
                QMessageBox.information(self, "Duplicado", "Esta combinación de profesión y carpeta (independientemente del número de archivos) ya ha sido agregada.")
                return

        self.list_widget_data_sources_ml.addItem(item_text)
        print(f"Datos asociados (ML): {item_text}")

        # Limpiar campos para la siguiente entrada
        self.line_edit_profession_ml.clear()
        self.line_edit_selected_folder_ml.clear()
        self.line_edit_profession_ml.setFocus() # Foco en profesión para nueva entrada

    def _handle_remove_source_ml(self):
        selected_items = self.list_widget_data_sources_ml.selectedItems()
        if not selected_items:
            # QMessageBox.information(self, "Información", "Ningún elemento seleccionado para quitar.")
            print("Ningún elemento seleccionado para quitar (ML).") # Mantener log de consola
            return
        
        for item in selected_items:
            item_text = item.text()
            self.list_widget_data_sources_ml.takeItem(self.list_widget_data_sources_ml.row(item))
            print(f"Elemento quitado (ML): {item_text}")
