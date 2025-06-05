# vista_dl_entrenamiento.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QProgressBar, QTextEdit,
                             QComboBox, QSpinBox, QCheckBox, QGroupBox,
                             QGridLayout, QSlider, QDoubleSpinBox, QLineEdit, QListWidget, QFileDialog, QMessageBox) # Añadido QMessageBox
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap
import os

class VistaDLEntrenamiento(QWidget):
    """Vista para configurar y entrenar modelos de Deep Learning"""
    
    # Señales para comunicación con la vista principal
    entrenamiento_iniciado = pyqtSignal()
    entrenamiento_completado = pyqtSignal()
    volver_solicitado = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VistaDLEntrenamiento")
        self.progreso_actual = 0
        self.epoca_actual = 0
        self.timer_entrenamiento = None
        
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
        title_label = QLabel("🧠 Entrenamiento Deep Learning")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #E0E0E0; margin-bottom: 10px;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        header_layout.addWidget(QLabel())  # Espaciador para centrar el título
        
        layout.addLayout(header_layout)
        
        # Contenido principal
        main_content = QWidget()
        content_layout = QVBoxLayout(main_content)
        content_layout.setSpacing(25)

        # Sección de Configuración de Datos y Profesión
        self.create_data_profession_config_section(content_layout)

        # Sección para Nombre del Modelo
        self.create_model_name_section(content_layout)
        
        # Sección de arquitectura de red
        self.create_architecture_section(content_layout)
        
        # Sección de hiperparámetros
        self.create_hyperparameters_section(content_layout)
        
        # Sección de entrenamiento
        self.create_training_section(content_layout)
        
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
                border: 2px solid #8E44AD; /* Mismo color que en ML para consistencia */
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

        self.line_edit_model_name_dl = QLineEdit()
        self.line_edit_model_name_dl.setPlaceholderText("Ej: ModeloImagenesCNN, ModeloTextosLSTM")
        self.line_edit_model_name_dl.setStyleSheet("""
            QLineEdit {
                background-color: #34495E;
                color: white;
                border: 1px solid #8E44AD;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        layout.addWidget(self.line_edit_model_name_dl)
        parent_layout.addWidget(group)

    def create_architecture_section(self, parent_layout):
        """Crea la sección de arquitectura de red neuronal"""
        group = QGroupBox("3. Arquitectura de Red Neuronal") # Numeración actualizada
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
        
        # Tipo de red
        layout.addWidget(QLabel("Tipo de red:"), 0, 0)
        self.combo_red = QComboBox()
        self.combo_red.addItems([
            "Feedforward Neural Network",
            "Convolutional Neural Network (CNN)",
            "Recurrent Neural Network (RNN)",
            "Long Short-Term Memory (LSTM)",
            "Transformer",
            "Autoencoder"
        ])
        self.combo_red.setStyleSheet("""
            QComboBox {
                background-color: #34495E;
                color: white;
                border: 1px solid #E74C3C;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
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
        layout.addWidget(self.combo_red, 0, 1)
        
        # Número de capas ocultas
        layout.addWidget(QLabel("Capas ocultas:"), 1, 0)
        self.spin_capas = QSpinBox()
        self.spin_capas.setRange(1, 20)
        self.spin_capas.setValue(3)
        self.spin_capas.setStyleSheet("""
            QSpinBox {
                background-color: #34495E;
                color: white;
                border: 1px solid #E74C3C;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.spin_capas, 1, 1)
        
        # Neuronas por capa
        layout.addWidget(QLabel("Neuronas por capa:"), 2, 0)
        self.spin_neuronas = QSpinBox()
        self.spin_neuronas.setRange(16, 2048)
        self.spin_neuronas.setValue(128)
        self.spin_neuronas.setSingleStep(16)
        self.spin_neuronas.setStyleSheet(self.spin_capas.styleSheet())
        layout.addWidget(self.spin_neuronas, 2, 1)
        
        # Función de activación
        layout.addWidget(QLabel("Activación:"), 3, 0)
        self.combo_activacion = QComboBox()
        self.combo_activacion.addItems(["ReLU", "Sigmoid", "Tanh", "Leaky ReLU", "ELU", "Swish"])
        qss_combo_activacion_dl = """
            QComboBox {
                background-color: #34495E;
                color: white;
                border: 1px solid #E74C3C;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
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
        """
        self.combo_activacion.setStyleSheet(qss_combo_activacion_dl)
        layout.addWidget(self.combo_activacion, 3, 1)
        
        # Aplicar estilos a las etiquetas
        for i in range(layout.rowCount()):
            item = layout.itemAtPosition(i, 0)
            if item and item.widget() and isinstance(item.widget(), QLabel):
                item.widget().setStyleSheet("color: #BDC3C7; font-size: 13px;")
        
        
        parent_layout.addWidget(group)

    def create_data_profession_config_section(self, parent_layout):
        """Crea la sección de configuración de datos y profesión"""
        group = QGroupBox("1. Configuración de Datos y Profesión")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
                border: 2px solid #1ABC9C; /* Mismo color que en ML para consistencia */
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
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        # Profesión Objetivo
        label_profession = QLabel("Profesión Objetivo:")
        label_profession.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        layout.addWidget(label_profession, 0, 0)
        
        self.line_edit_profession_dl = QLineEdit()
        self.line_edit_profession_dl.setPlaceholderText("Ej: Científico de Datos")
        self.line_edit_profession_dl.setStyleSheet("""
            QLineEdit {
                background-color: #34495E;
                color: white;
                border: 1px solid #1ABC9C;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        layout.addWidget(self.line_edit_profession_dl, 0, 1, 1, 2)

        # Carpeta de Datos
        label_data_folder = QLabel("Carpeta de Datos:")
        label_data_folder.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        layout.addWidget(label_data_folder, 1, 0)

        self.line_edit_selected_folder_dl = QLineEdit()
        self.line_edit_selected_folder_dl.setPlaceholderText("Seleccione una carpeta...")
        self.line_edit_selected_folder_dl.setReadOnly(True)
        self.line_edit_selected_folder_dl.setStyleSheet("""
            QLineEdit {
                background-color: #2C3E50; 
                color: #BDC3C7;
                border: 1px solid #1ABC9C;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        layout.addWidget(self.line_edit_selected_folder_dl, 1, 1)

        self.btn_select_folder_dl = QPushButton("Seleccionar Carpeta...")
        self.btn_select_folder_dl.setStyleSheet("""
            QPushButton { background-color: #3498DB; color: white; border: none; border-radius: 5px; padding: 8px 12px; font-size: 12px; font-weight: bold; }
            QPushButton:hover { background-color: #2980B9; }
        """)
        self.btn_select_folder_dl.clicked.connect(self._handle_select_folder_dl)
        layout.addWidget(self.btn_select_folder_dl, 1, 2)
        
        # Botón para agregar la asociación
        self.btn_add_profession_data_dl = QPushButton("Asociar Profesión y Carpeta")
        self.btn_add_profession_data_dl.setStyleSheet("""
            QPushButton { background-color: #2ECC71; color: white; border: none; border-radius: 5px; padding: 10px 15px; font-size: 13px; font-weight: bold; }
            QPushButton:hover { background-color: #27AE60; }
        """)
        self.btn_add_profession_data_dl.clicked.connect(self._handle_add_profession_data_dl)
        layout.addWidget(self.btn_add_profession_data_dl, 2, 1, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)

        # Lista de Datos Asociados
        label_data_sources = QLabel("Datos Asociados (Profesión - Carpeta):")
        label_data_sources.setStyleSheet("color: #BDC3C7; font-size: 13px; margin-top: 10px;")
        layout.addWidget(label_data_sources, 3, 0, 1, 3)

        self.list_widget_data_sources_dl = QListWidget()
        self.list_widget_data_sources_dl.setStyleSheet("""
            QListWidget {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #1ABC9C;
                border-radius: 5px;
                padding: 5px;
                font-size: 12px;
                min-height: 100px;
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:selected {
                background-color: #1ABC9C;
                color: #2C3E50;
            }
        """)
        layout.addWidget(self.list_widget_data_sources_dl, 4, 0, 1, 3)

        # Botón para quitar seleccionado
        self.btn_remove_source_dl = QPushButton("Quitar Seleccionado")
        self.btn_remove_source_dl.setStyleSheet("""
            QPushButton { background-color: #E74C3C; color: white; border: none; border-radius: 5px; padding: 8px 12px; font-size: 12px; font-weight: bold; }
            QPushButton:hover { background-color: #C0392B; }
        """)
        self.btn_remove_source_dl.clicked.connect(self._handle_remove_source_dl)
        layout.addWidget(self.btn_remove_source_dl, 5, 2, alignment=Qt.AlignmentFlag.AlignRight)
        
        parent_layout.addWidget(group)
        
        
    def create_hyperparameters_section(self, parent_layout):
        """Crea la sección de hiperparámetros"""
        group = QGroupBox("4. Hiperparámetros de Entrenamiento") # Numeración actualizada
        group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
                border: 2px solid #9B59B6;
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
        
        # Learning rate
        layout.addWidget(QLabel("Learning Rate:"), 0, 0)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 1.0)
        self.spin_lr.setValue(0.001)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setSingleStep(0.0001)
        self.spin_lr.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #34495E;
                color: white;
                border: 1px solid #9B59B6;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.spin_lr, 0, 1)
        
        # Batch size
        layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(8, 512)
        self.spin_batch.setValue(32)
        self.spin_batch.setSingleStep(8)
        self.spin_batch.setStyleSheet("""
            QSpinBox {
                background-color: #34495E;
                color: white;
                border: 1px solid #9B59B6;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.spin_batch, 1, 1)
        
        # Épocas
        layout.addWidget(QLabel("Épocas:"), 2, 0)
        self.spin_epocas = QSpinBox()
        self.spin_epocas.setRange(10, 1000)
        self.spin_epocas.setValue(100)
        self.spin_epocas.setStyleSheet(self.spin_batch.styleSheet())
        layout.addWidget(self.spin_epocas, 2, 1)
        
        # Dropout
        layout.addWidget(QLabel("Dropout:"), 3, 0)
        self.spin_dropout = QDoubleSpinBox()
        self.spin_dropout.setRange(0.0, 0.9)
        self.spin_dropout.setValue(0.2)
        self.spin_dropout.setDecimals(2)
        self.spin_dropout.setSingleStep(0.1)
        self.spin_dropout.setStyleSheet(self.spin_lr.styleSheet())
        layout.addWidget(self.spin_dropout, 3, 1)
        
        # Optimizador
        layout.addWidget(QLabel("Optimizador:"), 4, 0)
        self.combo_optimizador = QComboBox()
        self.combo_optimizador.addItems(["Adam", "SGD", "RMSprop", "AdaGrad", "AdaDelta"])
        self.combo_optimizador.setStyleSheet("""
            QComboBox {
                background-color: #34495E;
                color: white;
                border: 1px solid #9B59B6;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
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
        layout.addWidget(self.combo_optimizador, 4, 1)
        
        # Aplicar estilos a las etiquetas
        for i in range(layout.rowCount()):
            item = layout.itemAtPosition(i, 0)
            if item and item.widget() and isinstance(item.widget(), QLabel):
                item.widget().setStyleSheet("color: #BDC3C7; font-size: 13px;")
        
        parent_layout.addWidget(group)
        
        
    def create_training_section(self, parent_layout):
        """Crea la sección de configuración de entrenamiento"""
        # El comentario original decía create_training_section, pero la función se llamaba create_training_config_section
        # Mantendré create_training_section para consistencia con init_ui
        group = QGroupBox("5. Configuración de Entrenamiento Adicional") # Numeración actualizada
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
        
        # Checkboxes de opciones
        options_layout = QGridLayout()
        
        self.check_early_stopping = QCheckBox("Early Stopping")
        self.check_early_stopping.setChecked(True)
        self.check_early_stopping.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        options_layout.addWidget(self.check_early_stopping, 0, 0)
        
        self.check_data_augmentation = QCheckBox("Data Augmentation")
        self.check_data_augmentation.setChecked(False)
        self.check_data_augmentation.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        options_layout.addWidget(self.check_data_augmentation, 0, 1)
        
        self.check_batch_norm = QCheckBox("Batch Normalization")
        self.check_batch_norm.setChecked(True)
        self.check_batch_norm.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        options_layout.addWidget(self.check_batch_norm, 1, 0)
        
        self.check_lr_scheduler = QCheckBox("Learning Rate Scheduler")
        self.check_lr_scheduler.setChecked(False)
        self.check_lr_scheduler.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        options_layout.addWidget(self.check_lr_scheduler, 1, 1)
        
        layout.addLayout(options_layout)
        
        parent_layout.addWidget(group)
        
        
    def create_progress_section(self, parent_layout):
        """Crea la sección de progreso"""
        # El comentario original decía create_progress_section, pero la función se llamaba create_progress_section_dl
        # Mantendré create_progress_section para consistencia con init_ui
        group = QGroupBox("6. Progreso del Entrenamiento") # Numeración actualizada
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
        
        # Información de época actual
        self.label_epoca = QLabel("Época: 0 / 0")
        self.label_epoca.setStyleSheet("color: #27AE60; font-size: 14px; font-weight: bold;")
        layout.addWidget(self.label_epoca)
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
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
        
        # Métricas en tiempo real
        metrics_layout = QHBoxLayout()
        
        self.label_loss = QLabel("Loss: --")
        self.label_loss.setStyleSheet("color: #E74C3C; font-size: 12px; font-weight: bold;")
        metrics_layout.addWidget(self.label_loss)
        
        self.label_accuracy = QLabel("Accuracy: --")
        self.label_accuracy.setStyleSheet("color: #27AE60; font-size: 12px; font-weight: bold;")
        metrics_layout.addWidget(self.label_accuracy)
        
        self.label_val_loss = QLabel("Val Loss: --")
        self.label_val_loss.setStyleSheet("color: #F39C12; font-size: 12px; font-weight: bold;")
        metrics_layout.addWidget(self.label_val_loss)
        
        self.label_val_accuracy = QLabel("Val Accuracy: --")
        self.label_val_accuracy.setStyleSheet("color: #3498DB; font-size: 12px; font-weight: bold;")
        metrics_layout.addWidget(self.label_val_accuracy)
        
        layout.addLayout(metrics_layout)
        
        # Log de entrenamiento
        self.log_entrenamiento = QTextEdit()
        self.log_entrenamiento.setMaximumHeight(120)
        self.log_entrenamiento.setReadOnly(True)
        self.log_entrenamiento.setPlaceholderText("Los logs del entrenamiento aparecerán aquí...")
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
        self.btn_iniciar.setStyleSheet("""
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
        self.btn_iniciar.clicked.connect(self.iniciar_entrenamiento)
        
        # Botón detener
        self.btn_detener = QPushButton("⏹ Detener")
        self.btn_detener.setFixedHeight(50)
        self.btn_detener.setEnabled(False)
        self.btn_detener.setStyleSheet("""
            QPushButton {
                background-color: #95A5A6;
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 16px;
                font-weight: bold;
                padding: 15px 30px;
            }
            QPushButton:hover {
                background-color: #BDC3C7;
            }
            QPushButton:pressed {
                background-color: #7F8C8D;
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
        
    def iniciar_entrenamiento(self):
        """Inicia el proceso de entrenamiento simulado"""
        self.btn_iniciar.setEnabled(False)
        self.btn_detener.setEnabled(True)
        self.progreso_actual = 0
        self.epoca_actual = 0
        self.progress_bar.setValue(0)
        
        # Limpiar log y métricas
        self.log_entrenamiento.clear()
        self.log_entrenamiento.append("🔄 Iniciando entrenamiento Deep Learning...")
        self.log_entrenamiento.append(f"🧠 Arquitectura: {self.combo_red.currentText()}")
        self.log_entrenamiento.append(f"🔢 Capas ocultas: {self.spin_capas.value()}")
        self.log_entrenamiento.append(f"⚡ Neuronas por capa: {self.spin_neuronas.value()}")
        self.log_entrenamiento.append(f"📊 Learning Rate: {self.spin_lr.value()}")
        self.log_entrenamiento.append(f"📦 Batch Size: {self.spin_batch.value()}")
        self.log_entrenamiento.append(f"🔄 Épocas: {self.spin_epocas.value()}")
        self.log_entrenamiento.append("=" * 50)
        
        # Actualizar label de época
        self.label_epoca.setText(f"Época: 0 / {self.spin_epocas.value()}")
        
        # Emitir señal
        self.entrenamiento_iniciado.emit()
        
        # Iniciar timer para simular progreso
        self.timer_entrenamiento = QTimer()
        self.timer_entrenamiento.timeout.connect(self.actualizar_progreso)
        self.timer_entrenamiento.start(300)  # Actualizar cada 300ms
        
    def actualizar_progreso(self):
        """Actualiza el progreso del entrenamiento"""
        import random
        
        # Calcular progreso basado en épocas
        epocas_totales = self.spin_epocas.value()
        progreso_por_epoca = 100 / epocas_totales
        
        # Incrementar progreso
        incremento = random.uniform(0.5, 2.0)
        self.progreso_actual += incremento
        
        # Calcular época actual
        nueva_epoca = int(self.progreso_actual / progreso_por_epoca) + 1
        
        if nueva_epoca > self.epoca_actual and nueva_epoca <= epocas_totales:
            self.epoca_actual = nueva_epoca
            
            # Simular métricas
            loss = round(random.uniform(0.1, 2.0) * (1 - self.progreso_actual/100), 4)
            accuracy = round(0.5 + (self.progreso_actual/100) * 0.45 + random.uniform(-0.05, 0.05), 4)
            val_loss = round(loss + random.uniform(-0.1, 0.2), 4)
            val_accuracy = round(accuracy + random.uniform(-0.1, 0.1), 4)
            
            # Actualizar métricas
            self.label_loss.setText(f"Loss: {loss}")
            self.label_accuracy.setText(f"Accuracy: {accuracy}")
            self.label_val_loss.setText(f"Val Loss: {val_loss}")
            self.label_val_accuracy.setText(f"Val Accuracy: {val_accuracy}")
            
            # Actualizar época
            self.label_epoca.setText(f"Época: {self.epoca_actual} / {epocas_totales}")
            
            # Agregar log
            self.log_entrenamiento.append(f"📊 Época {self.epoca_actual}: Loss={loss}, Acc={accuracy}, Val_Loss={val_loss}, Val_Acc={val_accuracy}")
        
        if self.progreso_actual >= 100:
            self.progreso_actual = 100
            self.finalizar_entrenamiento()
        
        self.progress_bar.setValue(int(self.progreso_actual))
        
    def finalizar_entrenamiento(self):
        """Finaliza el entrenamiento"""
        if self.timer_entrenamiento:
            self.timer_entrenamiento.stop()
            
        self.btn_iniciar.setEnabled(True)
        self.btn_detener.setEnabled(False)
        
        self.log_entrenamiento.append("=" * 50)
        self.log_entrenamiento.append("✅ ¡Entrenamiento Deep Learning completado!")
        self.log_entrenamiento.append("🎯 Accuracy final: 0.943")
        self.log_entrenamiento.append("📉 Loss final: 0.156")
        self.log_entrenamiento.append("🔍 Val Accuracy: 0.938")
        self.log_entrenamiento.append("📊 Val Loss: 0.162")
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

    # --- Métodos para manejo de fuentes de datos por profesión (DL) ---
    def _handle_select_folder_dl(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de Datos para la Profesión (DL)")
        if folder_path:
            self.line_edit_selected_folder_dl.setText(folder_path)
            print(f"Carpeta seleccionada (DL): {folder_path}")
        else:
            print("Selección de carpeta cancelada (DL).")

    def _handle_add_profession_data_dl(self):
        profesion = self.line_edit_profession_dl.text().strip()
        ruta_carpeta = self.line_edit_selected_folder_dl.text().strip()

        if not profesion:
            QMessageBox.warning(self, "Campo Vacío", "Por favor, ingrese una profesión.")
            self.line_edit_profession_dl.setFocus()
            return
        
        if not ruta_carpeta:
            QMessageBox.warning(self, "Campo Vacío", "Por favor, seleccione una carpeta de datos.")
            self.btn_select_folder_dl.setFocus()
            return

        # Contar archivos en la carpeta
        num_archivos = 0
        try:
            if os.path.isdir(ruta_carpeta):
                num_archivos = len([f for f in os.listdir(ruta_carpeta) if os.path.isfile(os.path.join(ruta_carpeta, f))])
        except Exception as e:
            print(f"Error al contar archivos en {ruta_carpeta}: {e}")
            QMessageBox.warning(self, "Error Carpeta", f"No se pudo acceder o contar archivos en la carpeta:\n{ruta_carpeta}\n\nError: {e}")

        item_text = f"Profesión: {profesion}  |  Carpeta: {ruta_carpeta} (Archivos: {num_archivos})"
        
        # Chequeo de duplicado lógico (sin contar archivos)
        items_actuales_text = [self.list_widget_data_sources_dl.item(i).text() for i in range(self.list_widget_data_sources_dl.count())]
        profesion_carpeta_text_base = f"Profesión: {profesion}  |  Carpeta: {ruta_carpeta}"
        for item_existente in items_actuales_text:
            if item_existente.startswith(profesion_carpeta_text_base):
                QMessageBox.information(self, "Duplicado", "Esta combinación de profesión y carpeta (independientemente del número de archivos) ya ha sido agregada.")
                return

        self.list_widget_data_sources_dl.addItem(item_text)
        print(f"Datos asociados (DL): {item_text}")

        self.line_edit_profession_dl.clear()
        self.line_edit_selected_folder_dl.clear()
        self.line_edit_profession_dl.setFocus()

    def _handle_remove_source_dl(self):
        selected_items = self.list_widget_data_sources_dl.selectedItems()
        if not selected_items:
            print("Ningún elemento seleccionado para quitar (DL).")
            return
        
        for item in selected_items:
            item_text = item.text()
            self.list_widget_data_sources_dl.takeItem(self.list_widget_data_sources_dl.row(item))
            print(f"Elemento quitado (DL): {item_text}")
