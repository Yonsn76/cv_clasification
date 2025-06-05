from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QProgressBar, QTextEdit,
                             QComboBox, QSpinBox, QCheckBox, QGroupBox,
                             QGridLayout, QSlider, QLineEdit, QListWidget, QFileDialog, QMessageBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QColor
import os
import pandas as pd
from ml_backend.models.random_forest import RandomForestModel
from ml_backend.models.svm import SVMModel
from ml_backend.models.logistic_regression import LogisticRegressionModel
from ml_backend.models.gradient_boosting import GradientBoostingModel
from ml_backend.models.knn import KNNModel
from ml_backend.models.naive_bayes import NaiveBayesModel
from ml_backend.utils.preprocessing import extract_text_from_pdf, optimize_data, scale_features
from ml_backend.utils.validation import split_data, cross_validation


class VistaMLEntrenamiento(QWidget):
    """Vista para configurar y entrenar modelos de Machine Learning"""
    
    entrenamiento_iniciado = pyqtSignal()
    entrenamiento_completado = pyqtSignal()
    volver_solicitado = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VistaMLEntrenamiento")
        self.progreso_actual = 0
        self.timer_entrenamiento = None
        
        self.widgets_to_update = {}
        self.param_widgets = {} # Diccionario para rastrear los widgets de entrada de parámetros
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(20)
        
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
        
        title_label = QLabel("🤖 Entrenamiento Machine Learning")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label_ml = title_label 
        header_layout.addWidget(self.title_label_ml)
        
        header_layout.addStretch()
        
        main_window = self.parent() 
        if main_window and hasattr(main_window, 'color_text_light') and hasattr(main_window, 'color_text_medium') and hasattr(main_window, 'color_central_bg'):
            self.title_label_ml.setStyleSheet(f"color: {main_window.color_text_light}; margin-bottom: 10px;")
            self.btn_volver.setStyleSheet(f"""
                QPushButton {{
                    background-color: {main_window.color_text_medium if main_window.color_text_medium else '#7F8C8D'}; 
                    color: {main_window.color_central_bg if main_window.color_central_bg else 'white'}; 
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
        
        main_content = QWidget()
        content_layout = QVBoxLayout(main_content)
        content_layout.setSpacing(25)

        self.create_data_profession_config_section(content_layout)
        self.create_model_name_section(content_layout)
        self.create_algorithm_section(content_layout)
        self.create_parameters_section(content_layout)
        self.create_data_section(content_layout)
        self.create_progress_section(content_layout)
        self.create_action_buttons(content_layout)
        
        layout.addWidget(main_content)
        
        if self.combo_algoritmo.count() > 0:
            self.actualizar_descripcion_algoritmo(self.combo_algoritmo.currentText())

    def create_model_name_section(self, parent_layout):
        group = QGroupBox("2. Nombre del Modelo a Entrenar")
        self.widgets_to_update['group_model_name'] = group
        group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
                border: 2px solid #8E44AD;
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
        self.widgets_to_update['line_edit_model_name_ml'] = self.line_edit_model_name_ml
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
        parent_layout.addWidget(group)

    def create_algorithm_section(self, parent_layout):
        group = QGroupBox("3. Algoritmo de Machine Learning")
        self.widgets_to_update['group_algorithm'] = group
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
        algo_layout = QHBoxLayout()
        algo_label = QLabel("Algoritmo:")
        algo_label.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        self.widgets_to_update['algo_label_ml'] = algo_label
        self.combo_algoritmo = QComboBox()
        self.widgets_to_update['combo_algoritmo_ml'] = self.combo_algoritmo
        self.combo_algoritmo.addItems([
            "Random Forest", "Support Vector Machine (SVM)", "Logistic Regression",
            "Gradient Boosting", "K-Nearest Neighbors (KNN)", "Naive Bayes"
        ])
        self.combo_algoritmo.setStyleSheet("""
            QComboBox { background-color: #34495E; color: white; border: 1px solid #3498DB; border-radius: 5px; padding: 5px; } 
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { image: url(iconos/arrow_down_light.png); width: 10px; height: 10px; }
            QComboBox QListView { background-color: #2C3E50; border: 1px solid #3498DB; color: white; }
            QComboBox QListView::item { padding: 5px; }
            QComboBox QListView::item:selected { background-color: #3498DB; }
            QComboBox QListView::item:hover { background-color: #2980B9; }
        """)
        algo_layout.addWidget(algo_label)
        algo_layout.addWidget(self.combo_algoritmo)
        algo_layout.addStretch()
        layout.addLayout(algo_layout)
        self.desc_algoritmo = QLabel("Descripción del algoritmo...")
        self.desc_algoritmo.setWordWrap(True)
        self.widgets_to_update['desc_algoritmo_ml'] = self.desc_algoritmo
        self.desc_algoritmo.setStyleSheet("""
            color: #95A5A6; font-size: 12px; padding: 10px;
            background-color: rgba(52, 73, 94, 0.3); border-radius: 5px;
        """)
        layout.addWidget(self.desc_algoritmo)
        self.combo_algoritmo.currentTextChanged.connect(self.actualizar_descripcion_algoritmo)
        parent_layout.addWidget(group)

    def create_data_profession_config_section(self, parent_layout):
        group = QGroupBox("1. Configuración de Datos y Profesión")
        self.widgets_to_update['group_data_profession'] = group
        group.setStyleSheet(""" QGroupBox { font-size: 14px; font-weight: bold; color: #E0E0E0; border: 2px solid #3498DB; border-radius: 8px; margin-top: 10px; padding-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; } """)
        layout = QGridLayout(group)
        layout.setSpacing(15)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        self.label_profession = QLabel("Profesión Objetivo:")
        medium_text_color = getattr(self.window(), 'color_text_medium', "#BDC3C7")
        self.label_profession.setStyleSheet(f"color: {medium_text_color}; font-size: 13px;")
        self.widgets_to_update['label_profession_ml'] = self.label_profession
        layout.addWidget(self.label_profession, 0, 0)
        self.line_edit_profession_ml = QLineEdit()
        self.line_edit_profession_ml.setPlaceholderText("Ej: Ingeniero de Software")
        self.widgets_to_update['line_edit_profession_ml'] = self.line_edit_profession_ml
        self.line_edit_profession_ml.setStyleSheet(""" QLineEdit { background-color: #34495E; color: white; border: 1px solid #3498DB; border-radius: 5px; padding: 8px; font-size: 13px; } """)
        layout.addWidget(self.line_edit_profession_ml, 0, 1, 1, 2)
        self.label_data_folder = QLabel("Carpeta de Datos:")
        self.label_data_folder.setStyleSheet(f"color: {medium_text_color}; font-size: 13px;")
        self.widgets_to_update['label_data_folder_ml'] = self.label_data_folder
        layout.addWidget(self.label_data_folder, 1, 0)
        self.line_edit_selected_folder_ml = QLineEdit()
        self.line_edit_selected_folder_ml.setPlaceholderText("Seleccione una carpeta...")
        self.line_edit_selected_folder_ml.setReadOnly(True)
        self.widgets_to_update['line_edit_selected_folder_ml'] = self.line_edit_selected_folder_ml
        self.line_edit_selected_folder_ml.setStyleSheet(""" QLineEdit { background-color: #34495E; color: white; border: 1px solid #3498DB; border-radius: 5px; padding: 8px; font-size: 13px; } """)
        layout.addWidget(self.line_edit_selected_folder_ml, 1, 1)
        self.btn_select_folder_ml = QPushButton("Seleccionar Carpeta...")
        self.widgets_to_update['btn_select_folder_ml'] = self.btn_select_folder_ml
        self.btn_select_folder_ml.setStyleSheet(""" QPushButton { background-color: #E74C3C; color: white; border: none; border-radius: 8px; font-size: 13px; padding: 8px; } QPushButton:hover { background-color: #c0392b; } """)
        self.btn_select_folder_ml.clicked.connect(self._handle_select_folder_ml)
        layout.addWidget(self.btn_select_folder_ml, 1, 2)
        self.btn_add_profession_data_ml = QPushButton("Asociar Profesión y Carpeta")
        self.widgets_to_update['btn_add_profession_data_ml'] = self.btn_add_profession_data_ml
        self.btn_add_profession_data_ml.setStyleSheet(""" QPushButton { background-color: #E74C3C; color: white; border: none; border-radius: 8px; font-size: 13px; padding: 8px; } QPushButton:hover { background-color: #c0392b; } """)
        self.btn_add_profession_data_ml.clicked.connect(self._handle_add_profession_data_ml)
        layout.addWidget(self.btn_add_profession_data_ml, 2, 1, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        self.label_data_sources = QLabel("Datos Asociados (Profesión - Carpeta):")
        self.label_data_sources.setStyleSheet(f"color: {medium_text_color}; font-size: 13px; margin-top: 10px;")
        self.widgets_to_update['label_data_sources_ml'] = self.label_data_sources
        layout.addWidget(self.label_data_sources, 3, 0, 1, 3)
        self.list_widget_data_sources_ml = QListWidget()
        self.widgets_to_update['list_widget_data_sources_ml'] = self.list_widget_data_sources_ml
        self.list_widget_data_sources_ml.setStyleSheet(""" QListWidget { background-color: #34495E; color: white; border: 1px solid #3498DB; border-radius: 5px; } QListWidget::item { padding: 5px; } QListWidget::item:selected { background-color: #3498DB; color: white; } """)
        layout.addWidget(self.list_widget_data_sources_ml, 4, 0, 1, 3)
        self.btn_remove_source_ml = QPushButton("Quitar Seleccionado")
        self.widgets_to_update['btn_remove_source_ml'] = self.btn_remove_source_ml
        self.btn_remove_source_ml.setStyleSheet(""" QPushButton { background-color: #C0392B; color: white; border: none; border-radius: 8px; font-size: 13px; padding: 8px; } QPushButton:hover { background-color: #A93226; } """)
        self.btn_remove_source_ml.clicked.connect(self._handle_remove_source_ml)
        layout.addWidget(self.btn_remove_source_ml, 5, 2, alignment=Qt.AlignmentFlag.AlignRight)
        parent_layout.addWidget(group)
        
    def create_parameters_section(self, parent_layout):
        group = QGroupBox("4. Parámetros de Entrenamiento")
        self.widgets_to_update['group_parameters'] = group
        group.setStyleSheet("""
            QGroupBox {
                font-size: 14px; font-weight: bold; color: #E0E0E0;
                border: 2px solid #E74C3C; border-radius: 8px;
                margin-top: 10px; padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px;
            }
        """)
        
        # Este layout contendrá los widgets de parámetros que cambian dinámicamente
        self.params_layout = QGridLayout()
        self.params_layout.setSpacing(15)
        
        # Usamos un QWidget como contenedor para el QGridLayout,
        # lo que facilita manejar su visibilidad o reemplazarlo si fuera necesario,
        # aunque aquí solo estamos limpiando el contenido del QGridLayout.
        params_container_widget = QWidget()
        params_container_widget.setLayout(self.params_layout)
        
        group_layout = QVBoxLayout(group) # Layout principal del QGroupBox
        group_layout.addWidget(params_container_widget) # Añade el contenedor de parámetros

        # Checkbox para validación cruzada (fuera del layout dinámico)
        self.check_validacion = QCheckBox("Usar validación cruzada")
        self.check_validacion.setChecked(True)
        self.widgets_to_update['check_validacion_ml'] = self.check_validacion
        group_layout.addWidget(self.check_validacion)

        parent_layout.addWidget(group)

    def update_parameters_ui(self, algoritmo):
        # Limpiar completamente el QGridLayout de parámetros anteriores
        while self.params_layout.count() > 0:
            item = self.params_layout.takeAt(0) # Extrae el QLayoutItem en el índice 0
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None) # Remueve el widget de la jerarquía de layouts
                    widget.deleteLater()   # Programa la eliminación del widget

        # Limpiar el diccionario que rastrea los widgets de entrada de parámetros
        self.param_widgets.clear()

        # Estilos para los nuevos widgets de parámetros
        spin_style = """
            QSpinBox, QDoubleSpinBox {
                background-color: #34495E; color: white; border: 1px solid #E74C3C;
                border-radius: 5px; padding: 5px;
            }
        """
        combo_style = """
            QComboBox {
                background-color: #34495E; color: white; border: 1px solid #E74C3C;
                border-radius: 5px; padding: 5px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QListView {
                background-color: #2C3E50; border: 1px solid #E74C3C; color: white;
            }
            QComboBox QListView::item { padding: 5px; }
            QComboBox QListView::item:selected { background-color: #E74C3C; }
        """
        label_style = "color: #BDC3C7; font-size: 13px;"
        current_row = 0 # Para organizar los nuevos widgets en el QGridLayout

        # Añadir nuevos widgets de parámetros según el algoritmo seleccionado
        if algoritmo == "Random Forest":
            label_estimadores = QLabel("Número de estimadores:")
            label_estimadores.setStyleSheet(label_style)
            spin_estimadores = QSpinBox()
            spin_estimadores.setRange(10, 1000); spin_estimadores.setValue(100)
            spin_estimadores.setStyleSheet(spin_style)
            self.params_layout.addWidget(label_estimadores, current_row, 0)
            self.params_layout.addWidget(spin_estimadores, current_row, 1)
            self.param_widgets['n_estimators'] = spin_estimadores # Rastrear el widget de entrada
            current_row += 1

            label_profundidad = QLabel("Profundidad máxima:")
            label_profundidad.setStyleSheet(label_style)
            spin_profundidad = QSpinBox()
            spin_profundidad.setRange(1, 50); spin_profundidad.setValue(10)
            spin_profundidad.setStyleSheet(spin_style)
            self.params_layout.addWidget(label_profundidad, current_row, 0)
            self.params_layout.addWidget(spin_profundidad, current_row, 1)
            self.param_widgets['max_depth'] = spin_profundidad # Rastrear el widget de entrada
            current_row += 1

        elif algoritmo == "Gradient Boosting":
            label_estimadores = QLabel("Número de estimadores:")
            label_estimadores.setStyleSheet(label_style)
            spin_estimadores = QSpinBox()
            spin_estimadores.setRange(10, 1000); spin_estimadores.setValue(100)
            spin_estimadores.setStyleSheet(spin_style)
            self.params_layout.addWidget(label_estimadores, current_row, 0)
            self.params_layout.addWidget(spin_estimadores, current_row, 1)
            self.param_widgets['n_estimators'] = spin_estimadores
            current_row += 1

            label_lr = QLabel("Tasa de aprendizaje:")
            label_lr.setStyleSheet(label_style)
            spin_lr = QDoubleSpinBox()
            spin_lr.setRange(0.01, 1.0); spin_lr.setValue(0.1); spin_lr.setSingleStep(0.01)
            spin_lr.setDecimals(3)
            spin_lr.setStyleSheet(spin_style)
            self.params_layout.addWidget(label_lr, current_row, 0)
            self.params_layout.addWidget(spin_lr, current_row, 1)
            self.param_widgets['learning_rate'] = spin_lr
            current_row += 1
            
            label_profundidad = QLabel("Profundidad máxima:")
            label_profundidad.setStyleSheet(label_style)
            spin_profundidad = QSpinBox()
            spin_profundidad.setRange(1, 20); spin_profundidad.setValue(3)
            spin_profundidad.setStyleSheet(spin_style)
            self.params_layout.addWidget(label_profundidad, current_row, 0)
            self.params_layout.addWidget(spin_profundidad, current_row, 1)
            self.param_widgets['max_depth'] = spin_profundidad
            current_row += 1

        elif algoritmo == "Support Vector Machine (SVM)":
            label_c = QLabel("Parámetro de regularización (C):")
            label_c.setStyleSheet(label_style)
            spin_c = QDoubleSpinBox()
            spin_c.setRange(0.001, 1000.0); spin_c.setValue(1.0); spin_c.setSingleStep(0.1)
            spin_c.setDecimals(3)
            spin_c.setStyleSheet(spin_style)
            self.params_layout.addWidget(label_c, current_row, 0)
            self.params_layout.addWidget(spin_c, current_row, 1)
            self.param_widgets['C'] = spin_c
            current_row += 1

            label_kernel = QLabel("Tipo de kernel:")
            label_kernel.setStyleSheet(label_style)
            combo_kernel = QComboBox()
            combo_kernel.addItems(['rbf', 'linear', 'poly', 'sigmoid'])
            combo_kernel.setCurrentText('rbf')
            combo_kernel.setStyleSheet(combo_style)
            self.params_layout.addWidget(label_kernel, current_row, 0)
            self.params_layout.addWidget(combo_kernel, current_row, 1)
            self.param_widgets['kernel'] = combo_kernel
            current_row += 1

        elif algoritmo == "Logistic Regression":
            label_c = QLabel("Inverso de regularización (C):")
            label_c.setStyleSheet(label_style)
            spin_c = QDoubleSpinBox()
            spin_c.setRange(0.001, 1000.0); spin_c.setValue(1.0); spin_c.setSingleStep(0.1)
            spin_c.setDecimals(3)
            spin_c.setStyleSheet(spin_style)
            self.params_layout.addWidget(label_c, current_row, 0)
            self.params_layout.addWidget(spin_c, current_row, 1)
            self.param_widgets['C'] = spin_c
            current_row += 1

            label_solver = QLabel("Algoritmo de optimización:")
            label_solver.setStyleSheet(label_style)
            combo_solver = QComboBox()
            combo_solver.addItems(['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'])
            combo_solver.setCurrentText('lbfgs')
            combo_solver.setStyleSheet(combo_style)
            self.params_layout.addWidget(label_solver, current_row, 0)
            self.params_layout.addWidget(combo_solver, current_row, 1)
            self.param_widgets['solver'] = combo_solver
            current_row += 1

        elif algoritmo == "K-Nearest Neighbors (KNN)":
            label_neighbors = QLabel("Número de vecinos (k):")
            label_neighbors.setStyleSheet(label_style)
            spin_neighbors = QSpinBox()
            spin_neighbors.setRange(1, 100); spin_neighbors.setValue(5)
            spin_neighbors.setStyleSheet(spin_style)
            self.params_layout.addWidget(label_neighbors, current_row, 0)
            self.params_layout.addWidget(spin_neighbors, current_row, 1)
            self.param_widgets['n_neighbors'] = spin_neighbors
            current_row += 1

            label_weights = QLabel("Ponderación de vecinos:")
            label_weights.setStyleSheet(label_style)
            combo_weights = QComboBox()
            combo_weights.addItems(['uniform', 'distance'])
            combo_weights.setCurrentText('uniform')
            combo_weights.setStyleSheet(combo_style)
            self.params_layout.addWidget(label_weights, current_row, 0)
            self.params_layout.addWidget(combo_weights, current_row, 1)
            self.param_widgets['weights'] = combo_weights
            current_row += 1

        elif algoritmo == "Naive Bayes":
            label_alpha = QLabel("Suavizado (alpha):")
            label_alpha.setStyleSheet(label_style)
            spin_alpha = QDoubleSpinBox()
            spin_alpha.setRange(0.0, 10.0); spin_alpha.setValue(1.0); spin_alpha.setSingleStep(0.1)
            spin_alpha.setStyleSheet(spin_style)
            self.params_layout.addWidget(label_alpha, current_row, 0)
            self.params_layout.addWidget(spin_alpha, current_row, 1)
            self.param_widgets['alpha'] = spin_alpha
            current_row += 1
            
        # Actualizar la geometría del QGroupBox que contiene los parámetros
        # para que se ajuste al nuevo contenido.
        if 'group_parameters' in self.widgets_to_update and self.widgets_to_update['group_parameters']:
             self.widgets_to_update['group_parameters'].updateGeometry()


    def create_data_section(self, parent_layout):
        group = QGroupBox("5. Configuración de Datos Adicional")
        self.widgets_to_update['group_data_additional'] = group
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        split_layout = QHBoxLayout()
        split_label = QLabel("División entrenamiento/prueba:")
        split_label.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        self.widgets_to_update['split_label_ml'] = split_label
        self.slider_split = QSlider(Qt.Orientation.Horizontal)
        self.widgets_to_update['slider_split_ml'] = self.slider_split
        self.slider_split.setRange(60, 90); self.slider_split.setValue(80)
        self.label_split = QLabel("80%")
        self.label_split.setStyleSheet("color: #F39C12; font-weight: bold;")
        self.widgets_to_update['label_split_value_ml'] = self.label_split
        self.slider_split.valueChanged.connect(lambda v: self.label_split.setText(f"{v}%"))
        split_layout.addWidget(split_label)
        split_layout.addWidget(self.slider_split)
        split_layout.addWidget(self.label_split)
        layout.addLayout(split_layout)
        parent_layout.addWidget(group)
        
    def create_progress_section(self, parent_layout):
        group = QGroupBox("6. Progreso del Entrenamiento")
        self.widgets_to_update['group_progress'] = group
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        self.widgets_to_update['progress_bar_ml'] = self.progress_bar
        layout.addWidget(self.progress_bar)
        self.log_entrenamiento = QTextEdit()
        self.log_entrenamiento.setMaximumHeight(120); self.log_entrenamiento.setReadOnly(True)
        self.log_entrenamiento.setPlaceholderText("Los logs del entrenamiento aparecerán aquí...")
        self.widgets_to_update['log_entrenamiento_ml'] = self.log_entrenamiento
        layout.addWidget(self.log_entrenamiento)
        parent_layout.addWidget(group)
        
    def create_action_buttons(self, parent_layout):
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(20)
        self.btn_iniciar = QPushButton("🚀 Iniciar Entrenamiento")
        self.btn_iniciar.setFixedHeight(50)
        self.widgets_to_update['btn_iniciar_ml'] = self.btn_iniciar
        self.btn_iniciar.clicked.connect(self.iniciar_entrenamiento)
        self.btn_detener = QPushButton("⏹ Detener")
        self.btn_detener.setFixedHeight(50); self.btn_detener.setEnabled(False)
        self.widgets_to_update['btn_detener_ml'] = self.btn_detener
        self.btn_detener.clicked.connect(self.detener_entrenamiento)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.btn_iniciar)
        buttons_layout.addWidget(self.btn_detener)
        buttons_layout.addStretch()
        parent_layout.addLayout(buttons_layout)
        
    def actualizar_descripcion_algoritmo(self, algoritmo):
        descripciones = {
            "Random Forest": "Random Forest: Ensemble de árboles de decisión, robusto y eficaz para clasificación.",
            "Support Vector Machine (SVM)": "SVM: Encuentra el hiperplano óptimo para separar clases, efectivo en espacios de alta dimensión.",
            "Logistic Regression": "Regresión Logística: Modelo lineal para clasificación binaria y multiclase, interpretable y rápido.",
            "Gradient Boosting": "Gradient Boosting: Combina modelos débiles secuencialmente, muy potente pero puede sobreajustar.",
            "K-Nearest Neighbors (KNN)": "KNN: Clasifica basándose en los k vecinos más cercanos, simple pero sensible a la escala.",
            "Naive Bayes": "Naive Bayes: Basado en probabilidades con independencia condicional, rápido y efectivo con datos categóricos."
        }
        self.desc_algoritmo.setText(descripciones.get(algoritmo, "Descripción no disponible."))
        self.update_parameters_ui(algoritmo)

    def get_model_parameters(self, algoritmo):
        params = {}

        if algoritmo == "Random Forest":
            if 'n_estimators' in self.param_widgets:
                params['n_estimators'] = self.param_widgets['n_estimators'].value()
            if 'max_depth' in self.param_widgets:
                params['max_depth'] = self.param_widgets['max_depth'].value()

        elif algoritmo == "Gradient Boosting":
            if 'n_estimators' in self.param_widgets:
                params['n_estimators'] = self.param_widgets['n_estimators'].value()
            if 'learning_rate' in self.param_widgets:
                params['learning_rate'] = self.param_widgets['learning_rate'].value()
            if 'max_depth' in self.param_widgets:
                params['max_depth'] = self.param_widgets['max_depth'].value()

        elif algoritmo == "Support Vector Machine (SVM)":
            if 'C' in self.param_widgets:
                params['C'] = self.param_widgets['C'].value()
            if 'kernel' in self.param_widgets:
                params['kernel'] = self.param_widgets['kernel'].currentText()

        elif algoritmo == "Logistic Regression":
            if 'C' in self.param_widgets:
                params['C'] = self.param_widgets['C'].value()
            if 'solver' in self.param_widgets:
                params['solver'] = self.param_widgets['solver'].currentText()

        elif algoritmo == "K-Nearest Neighbors (KNN)":
            if 'n_neighbors' in self.param_widgets:
                params['n_neighbors'] = self.param_widgets['n_neighbors'].value()
            if 'weights' in self.param_widgets:
                params['weights'] = self.param_widgets['weights'].currentText()

        elif algoritmo == "Naive Bayes":
            if 'alpha' in self.param_widgets:
                params['alpha'] = self.param_widgets['alpha'].value()
        
        return params
        
    def iniciar_entrenamiento(self):
        self.btn_iniciar.setEnabled(False)
        self.btn_detener.setEnabled(True)
        self.progreso_actual = 0
        self.progress_bar.setValue(0)
        self.log_entrenamiento.clear()
        self.log_entrenamiento.append("🔄 Iniciando entrenamiento...")
        algoritmo = self.combo_algoritmo.currentText()
        self.log_entrenamiento.append(f"📊 Algoritmo: {algoritmo}")
        
        params = self.get_model_parameters(algoritmo)
        for key, value in params.items():
            self.log_entrenamiento.append(f"⚙️ {key.replace('_', ' ').capitalize()}: {value}")
            
        self.log_entrenamiento.append(f"📈 División datos: {self.slider_split.value()}% entrenamiento")
        self.log_entrenamiento.append("=" * 50)
        self.entrenamiento_iniciado.emit()
        self.timer_entrenamiento = QTimer()
        self.timer_entrenamiento.timeout.connect(self.actualizar_progreso)
        self.timer_entrenamiento.start(200)
        self.entrenar_modelo()

    def entrenar_modelo(self):
        try:
            data_sources = [self.list_widget_data_sources_ml.item(i).text().split("Carpeta: ")[1].split(" (")[0]
                            for i in range(self.list_widget_data_sources_ml.count())]
            if not data_sources:
                self.log_entrenamiento.append("❌ Error: No hay datos asociados para entrenar.")
                self.detener_entrenamiento()
                return

            X, y = self.cargar_datos(data_sources)
            if X is None or y is None:
                self.log_entrenamiento.append("❌ Error: No se pudieron cargar los datos.")
                self.detener_entrenamiento()
                return

            test_size = 1 - (self.slider_split.value() / 100)
            X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=42)
            self.log_entrenamiento.append(f"📊 Datos divididos: {len(X_train)} entrenamiento, {len(X_test)} prueba")

            algoritmo = self.combo_algoritmo.currentText()
            modelo = self.seleccionar_modelo(algoritmo)
            if modelo is None:
                self.log_entrenamiento.append(f"❌ Error: Modelo {algoritmo} no disponible.")
                self.detener_entrenamiento()
                return

            params = self.get_model_parameters(algoritmo)

            self.log_entrenamiento.append(f"🚀 Entrenando modelo {algoritmo}...")
            modelo.train(X_train, y_train)

            self.log_entrenamiento.append("📈 Evaluando modelo...")
            metrics = modelo.evaluate(X_test, y_test)

            self.log_entrenamiento.append("=" * 50)
            self.log_entrenamiento.append("✅ ¡Entrenamiento completado exitosamente!")
            self.log_entrenamiento.append(f"📈 Accuracy final: {metrics.get('accuracy', 'N/A'):.3f}")
            self.log_entrenamiento.append(f"🎯 Precisión: {metrics.get('precision', 'N/A'):.3f}")
            self.log_entrenamiento.append(f"🔄 Recall: {metrics.get('recall', 'N/A'):.3f}")
            self.log_entrenamiento.append(f"📊 F1-Score: {metrics.get('f1', 'N/A'):.3f}")
            self.log_entrenamiento.append("💾 Modelo guardado correctamente (simulado)")

            self.entrenamiento_completado.emit()

            if self.timer_entrenamiento:
                self.timer_entrenamiento.stop()
            self.progress_bar.setValue(100)
            self.btn_iniciar.setEnabled(True)
            self.btn_detener.setEnabled(False)

        except Exception as e:
            self.log_entrenamiento.append(f"❌ Error durante el entrenamiento: {str(e)}")
            self.detener_entrenamiento()


    def cargar_datos(self, data_sources):
        try:
            data = []
            labels = []
            for folder in data_sources:
                profesion_label = "Desconocida"
                for i in range(self.list_widget_data_sources_ml.count()):
                    item_text = self.list_widget_data_sources_ml.item(i).text()
                    if folder in item_text:
                        try:
                            profesion_label = item_text.split("Profesión: ")[1].split("  |")[0].strip()
                            break
                        except IndexError:
                            profesion_label = os.path.basename(folder)
                            break
                else:
                    profesion_label = os.path.basename(folder)

                for file_name in os.listdir(folder):
                    file_path = os.path.join(folder, file_name)
                    if file_name.endswith('.pdf'):
                        text = extract_text_from_pdf(file_path)
                        if text:
                            data.append(text)
                            labels.append(profesion_label)

            if not data:
                self.log_entrenamiento.append("❌ No se encontraron datos en los archivos PDF procesados.")
                return None, None

            df_optimized = optimize_data(data, is_text=True)

            if isinstance(df_optimized, pd.DataFrame):
                X_numeric = df_optimized.values
            elif hasattr(df_optimized, 'toarray'):
                X_numeric = df_optimized.toarray()
            else:
                X_numeric = df_optimized

            if not X_numeric.size:
                self.log_entrenamiento.append("❌ Error: El preprocesamiento no generó características.")
                return None, None

            X_scaled = scale_features(X_numeric)

            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(labels)
            self.log_entrenamiento.append(f"ℹ️ Clases codificadas: {encoder.classes_} -> {list(range(len(encoder.classes_)))}")
            self.label_encoder = encoder

            return X_scaled, y_encoded
            
        except Exception as e:
            self.log_entrenamiento.append(f"❌ Error al cargar o preprocesar datos: {str(e)}")
            import traceback
            self.log_entrenamiento.append(f"Traceback: {traceback.format_exc()}")
            return None, None

    def seleccionar_modelo(self, algoritmo):
        current_params = self.get_model_parameters(algoritmo)

        modelos = {
            "Random Forest": RandomForestModel(**current_params),
            "Support Vector Machine (SVM)": SVMModel(**current_params),
            "Logistic Regression": LogisticRegressionModel(**current_params),
            "Gradient Boosting": GradientBoostingModel(**current_params),
            "K-Nearest Neighbors (KNN)": KNNModel(**current_params),
            "Naive Bayes": NaiveBayesModel(**current_params) 
        }
        return modelos.get(algoritmo)

    def _handle_select_folder_ml(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de Datos")
        if folder_path:
            self.line_edit_selected_folder_ml.setText(folder_path)
            self.log_entrenamiento.append(f"📁 Carpeta de datos seleccionada: {folder_path}")
        else:
            self.log_entrenamiento.append("⚠️ Selección de carpeta cancelada.")

    def _handle_add_profession_data_ml(self):
        profession = self.line_edit_profession_ml.text().strip()
        folder_path = self.line_edit_selected_folder_ml.text().strip()

        if not profession:
            QMessageBox.warning(self, "Campo Vacío", "Por favor, ingrese un nombre para la profesión.")
            return
        if not folder_path:
            QMessageBox.warning(self, "Campo Vacío", "Por favor, seleccione una carpeta de datos.")
            return

        for i in range(self.list_widget_data_sources_ml.count()):
            item_text = self.list_widget_data_sources_ml.item(i).text()
            if f"Profesión: {profession}" in item_text and f"Carpeta: {folder_path}" in item_text:
                QMessageBox.information(self, "Duplicado", "Esta combinación de profesión y carpeta ya ha sido agregada.")
                return

        item_text = f"Profesión: {profession}  |  Carpeta: {folder_path} (Archivos: {self._count_pdf_files(folder_path)})"
        self.list_widget_data_sources_ml.addItem(item_text)
        self.log_entrenamiento.append(f"➕ Datos asociados: {profession} -> {folder_path}")

    def _handle_remove_source_ml(self):
        selected_items = self.list_widget_data_sources_ml.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Nada Seleccionado", "Por favor, seleccione un ítem de la lista para quitar.")
            return
        for item in selected_items:
            self.log_entrenamiento.append(f"➖ Datos desasociados: {item.text()}")
            self.list_widget_data_sources_ml.takeItem(self.list_widget_data_sources_ml.row(item))

    def _count_pdf_files(self, folder_path):
        if not os.path.isdir(folder_path):
            return 0
        count = 0
        try:
            for fname in os.listdir(folder_path):
                if fname.lower().endswith('.pdf'):
                    count += 1
        except Exception as e:
            print(f"Error contando archivos PDF en {folder_path}: {e}")
            return 0
        return count
        
    def actualizar_progreso(self):
        if self.progreso_actual < 100:
            self.progreso_actual += 5 
            self.progress_bar.setValue(self.progreso_actual)
            if self.progreso_actual % 20 == 0 : 
                self.log_entrenamiento.append(f"⏳ Progreso: {self.progreso_actual}%...")
        else:
            if self.timer_entrenamiento:
                self.timer_entrenamiento.stop()

    def detener_entrenamiento(self):
        if self.timer_entrenamiento:
            self.timer_entrenamiento.stop()
        self.log_entrenamiento.append("🛑 Entrenamiento detenido por el usuario.")
        self.progress_bar.setValue(0) 
        self.progreso_actual = 0
        self.btn_iniciar.setEnabled(True)
        self.btn_detener.setEnabled(False)

    def aplicar_tema(self, theme_colors):
        try:
            bg_color = theme_colors.get('color_fondo_principal', '#2C3E50')
            text_light = theme_colors.get('color_text_light', '#ECF0F1')
            text_medium = theme_colors.get('color_text_medium', '#BDC3C7')
            text_dark = theme_colors.get('color_text_dark', '#95A5A6')
            accent_color = theme_colors.get('color_acento_primario', '#3498DB')
            secondary_accent_color = theme_colors.get('color_acento_secundario', '#E74C3C')
            button_bg = theme_colors.get('color_botones', '#3498DB')
            button_text = theme_colors.get('color_text_botones', '#FFFFFF')
            input_bg = theme_colors.get('color_input_bg', '#34495E')

            self.setStyleSheet(f"background-color: {bg_color}; color: {text_light};")

            if hasattr(self, 'title_label_ml'):
                self.title_label_ml.setStyleSheet(f"color: {text_light}; margin-bottom: 10px;")
            if hasattr(self, 'btn_volver'):
                self.btn_volver.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {text_medium}; 
                        color: {bg_color}; 
                        border: none; border-radius: 17px; font-size: 13px; font-weight: bold;
                    }}
                    QPushButton:hover {{ background-color: {QColor(text_medium).lighter(110).name()}; }}
                """)

            group_style = """
                QGroupBox {{ font-size: 14px; font-weight: bold; color: {text_light};
                             border: 2px solid {border_color}; border-radius: 8px;
                             margin-top: 10px; padding-top: 10px; }}
                QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }}
            """
            label_style = f"color: {text_medium}; font-size: 13px;"
            input_style = f"""
                background-color: {input_bg}; color: {text_light};
                border: 1px solid {accent_color}; border-radius: 5px; padding: 8px; font-size: 13px;
            """
            combo_list_view_style = f"""
                QComboBox QListView {{
                    background-color: {input_bg}; color: {text_light};
                    border: 1px solid {accent_color}; selection-background-color: {accent_color};
                }}
                QComboBox QListView::item {{ padding: 5px; }}
                QComboBox QListView::item:selected {{ background-color: {accent_color}; color: {button_text}; }}
                QComboBox QListView::item:hover {{ background-color: {QColor(accent_color).lighter(110).name()}; }}
            """
            
            if hasattr(self, 'widgets_to_update'):
                group_boxes_config = {
                    'group_data_profession': accent_color,
                    'group_model_name': secondary_accent_color,
                    'group_algorithm': accent_color,
                    'group_parameters': secondary_accent_color,
                    'group_data_additional': accent_color,
                    'group_progress': secondary_accent_color
                }
                for key, border_c in group_boxes_config.items():
                    if key in self.widgets_to_update and self.widgets_to_update[key]:
                        self.widgets_to_update[key].setStyleSheet(group_style.format(text_light=text_light, border_color=border_c))

                if 'label_profession_ml' in self.widgets_to_update: self.widgets_to_update['label_profession_ml'].setStyleSheet(label_style)
                if 'line_edit_profession_ml' in self.widgets_to_update: self.widgets_to_update['line_edit_profession_ml'].setStyleSheet(input_style)
                if 'label_data_folder_ml' in self.widgets_to_update: self.widgets_to_update['label_data_folder_ml'].setStyleSheet(label_style)
                if 'line_edit_selected_folder_ml' in self.widgets_to_update: self.widgets_to_update['line_edit_selected_folder_ml'].setStyleSheet(input_style)
                if 'label_data_sources_ml' in self.widgets_to_update: self.widgets_to_update['label_data_sources_ml'].setStyleSheet(label_style)
                
                if 'line_edit_model_name_ml' in self.widgets_to_update: self.widgets_to_update['line_edit_model_name_ml'].setStyleSheet(input_style.replace(accent_color, secondary_accent_color))

                if 'algo_label_ml' in self.widgets_to_update: self.widgets_to_update['algo_label_ml'].setStyleSheet(label_style)
                if 'desc_algoritmo_ml' in self.widgets_to_update: 
                    self.widgets_to_update['desc_algoritmo_ml'].setStyleSheet(f"""
                        color: {text_dark}; font-size: 12px; padding: 10px;
                        background-color: {QColor(input_bg).lighter(110).name()}; border-radius: 5px;
                    """)

                if 'split_label_ml' in self.widgets_to_update: self.widgets_to_update['split_label_ml'].setStyleSheet(label_style)
                if 'label_split_value_ml' in self.widgets_to_update: self.widgets_to_update['label_split_value_ml'].setStyleSheet(f"color: {theme_colors.get('color_especial_1', '#F39C12')}; font-weight: bold;")

                if 'combo_algoritmo_ml' in self.widgets_to_update:
                    self.widgets_to_update['combo_algoritmo_ml'].setStyleSheet(f"""
                        QComboBox {{ {input_style} padding-right: 20px; }}
                        QComboBox::drop-down {{ border: none; subcontrol-origin: padding; subcontrol-position: top right; width: 20px; }}
                        QComboBox::down-arrow {{ image: url(iconos/arrow_down_light.png); width: 12px; height: 12px; }}
                        {combo_list_view_style}
                    """)
                
                spin_param_style = f"""
                    QSpinBox, QDoubleSpinBox {{
                        background-color: {input_bg}; color: {text_light};
                        border: 1px solid {secondary_accent_color}; 
                        border-radius: 5px; padding: 5px;
                    }}
                """
                combo_param_style = f"""
                    QComboBox {{
                        background-color: {input_bg}; color: {text_light};
                        border: 1px solid {secondary_accent_color}; border-radius: 5px; padding: 5px;
                    }}
                    QComboBox::drop-down {{ border: none; }}
                    QComboBox QListView {{
                        background-color: {QColor(input_bg).darker(110).name()}; border: 1px solid {secondary_accent_color}; color: {text_light};
                    }}
                    QComboBox QListView::item {{ padding: 5px; }}
                    QComboBox QListView::item:selected {{ background-color: {secondary_accent_color}; }}
                """
                # Aplicar estilos a los widgets de parámetros que existen actualmente
                # Esto es importante si el tema se aplica DESPUÉS de que los parámetros iniciales se hayan mostrado
                for widget_param in self.param_widgets.values():
                    if isinstance(widget_param, (QSpinBox, QDoubleSpinBox)):
                        widget_param.setStyleSheet(spin_param_style)
                    elif isinstance(widget_param, QComboBox):
                        widget_param.setStyleSheet(combo_param_style)
                
                # También es necesario aplicar estilo a las etiquetas de los parámetros si se recrean
                # y no se guardan en self.param_widgets. Se puede hacer buscando en self.params_layout
                # o asegurar que los estilos base se apliquen al crear las etiquetas.
                # La lógica actual en update_parameters_ui ya aplica label_style a las nuevas etiquetas.


                if 'check_validacion_ml' in self.widgets_to_update: self.widgets_to_update['check_validacion_ml'].setStyleSheet(f"color: {text_medium}; font-size: 13px;")

                if 'slider_split_ml' in self.widgets_to_update:
                    self.widgets_to_update['slider_split_ml'].setStyleSheet(f"""
                        QSlider::groove:horizontal {{
                            border: 1px solid {accent_color}; background: {input_bg}; height: 8px; border-radius: 4px;
                        }}
                        QSlider::handle:horizontal {{
                            background: {accent_color}; border: 1px solid {accent_color};
                            width: 18px; margin: -5px 0; border-radius: 9px;
                        }}
                    """)

                if 'progress_bar_ml' in self.widgets_to_update:
                    self.widgets_to_update['progress_bar_ml'].setStyleSheet(f"""
                        QProgressBar {{
                            border: 1px solid {secondary_accent_color}; border-radius: 5px; text-align: center;
                            background-color: {input_bg}; color: {text_light};
                        }}
                        QProgressBar::chunk {{ background-color: {secondary_accent_color}; width: 10px; margin: 0.5px; }}
                    """)
                
                if 'log_entrenamiento_ml' in self.widgets_to_update:
                    self.widgets_to_update['log_entrenamiento_ml'].setStyleSheet(f"""
                        QTextEdit {{
                            background-color: {QColor(input_bg).darker(110).name()}; color: {text_medium};
                            border: 1px solid {secondary_accent_color}; border-radius: 5px; padding: 5px;
                        }}
                    """)

                if 'list_widget_data_sources_ml' in self.widgets_to_update:
                    self.widgets_to_update['list_widget_data_sources_ml'].setStyleSheet(f"""
                        QListWidget {{
                            background-color: {input_bg}; color: {text_medium};
                            border: 1px solid {accent_color}; border-radius: 5px;
                        }}
                        QListWidget::item {{ padding: 5px; }}
                        QListWidget::item:selected {{ background-color: {accent_color}; color: {button_text}; }}
                    """)

                button_main_style = f"""
                    QPushButton {{
                        background-color: {button_bg}; color: {button_text};
                        border: none; border-radius: 8px; font-size: 14px; font-weight: bold; padding: 10px;
                    }}
                    QPushButton:hover {{ background-color: {QColor(button_bg).lighter(110).name()}; }}
                    QPushButton:disabled {{ background-color: {text_dark}; color: {text_medium}; }}
                """
                button_secondary_style = f"""
                    QPushButton {{
                        background-color: {secondary_accent_color}; color: {button_text};
                        border: none; border-radius: 8px; font-size: 13px; padding: 8px;
                    }}
                    QPushButton:hover {{ background-color: {QColor(secondary_accent_color).lighter(110).name()}; }}
                """
                
                if 'btn_select_folder_ml' in self.widgets_to_update: self.widgets_to_update['btn_select_folder_ml'].setStyleSheet(button_secondary_style)
                if 'btn_add_profession_data_ml' in self.widgets_to_update: self.widgets_to_update['btn_add_profession_data_ml'].setStyleSheet(button_secondary_style)
                if 'btn_remove_source_ml' in self.widgets_to_update: self.widgets_to_update['btn_remove_source_ml'].setStyleSheet(button_secondary_style.replace(secondary_accent_color, theme_colors.get('color_peligro', '#C0392B')))


                if 'btn_iniciar_ml' in self.widgets_to_update: self.widgets_to_update['btn_iniciar_ml'].setStyleSheet(button_main_style)
                if 'btn_detener_ml' in self.widgets_to_update: self.widgets_to_update['btn_detener_ml'].setStyleSheet(button_main_style.replace(button_bg, secondary_accent_color))


        except Exception as e:
            print(f"Error aplicando tema en VistaMLEntrenamiento: {e}")
            import traceback
            print(traceback.format_exc())
