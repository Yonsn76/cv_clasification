# vista_ml_entrenamiento.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QProgressBar, QTextEdit,
                             QComboBox, QSpinBox, QCheckBox, QGroupBox,
                             QGridLayout, QSlider)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap
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
        title_label.setStyleSheet("color: #E0E0E0; margin-bottom: 10px;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        header_layout.addWidget(QLabel())  # Espaciador para centrar el título
        
        layout.addLayout(header_layout)
        
        # Contenido principal en scroll
        main_content = QWidget()
        content_layout = QVBoxLayout(main_content)
        content_layout.setSpacing(25)
        
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
        
    def create_algorithm_section(self, parent_layout):
        """Crea la sección de selección de algoritmo"""
        group = QGroupBox("Algoritmo de Machine Learning")
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
        
        self.combo_algoritmo = QComboBox()
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
        """)
        
        algo_layout.addWidget(algo_label)
        algo_layout.addWidget(self.combo_algoritmo)
        algo_layout.addStretch()
        
        layout.addLayout(algo_layout)
        
        # Descripción del algoritmo
        self.desc_algoritmo = QLabel("Random Forest: Ensemble de árboles de decisión, robusto y eficaz para clasificación.")
        self.desc_algoritmo.setWordWrap(True)
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
        
    def create_parameters_section(self, parent_layout):
        """Crea la sección de parámetros"""
        group = QGroupBox("Parámetros de Entrenamiento")
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
        layout.addWidget(QLabel("Número de estimadores:"), 0, 0)
        self.spin_estimadores = QSpinBox()
        self.spin_estimadores.setRange(10, 1000)
        self.spin_estimadores.setValue(100)
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
        layout.addWidget(QLabel("Profundidad máxima:"), 1, 0)
        self.spin_profundidad = QSpinBox()
        self.spin_profundidad.setRange(1, 50)
        self.spin_profundidad.setValue(10)
        self.spin_profundidad.setStyleSheet(self.spin_estimadores.styleSheet())
        layout.addWidget(self.spin_profundidad, 1, 1)
        
        # Validación cruzada
        self.check_validacion = QCheckBox("Usar validación cruzada")
        self.check_validacion.setChecked(True)
        self.check_validacion.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        layout.addWidget(self.check_validacion, 2, 0, 1, 2)
        
        # Aplicar estilos a las etiquetas
        for i in range(layout.rowCount()):
            item = layout.itemAtPosition(i, 0)
            if item and item.widget() and isinstance(item.widget(), QLabel):
                item.widget().setStyleSheet("color: #BDC3C7; font-size: 13px;")
        
        parent_layout.addWidget(group)
        
    def create_data_section(self, parent_layout):
        """Crea la sección de configuración de datos"""
        group = QGroupBox("Configuración de Datos")
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
        
        self.slider_split = QSlider(Qt.Orientation.Horizontal)
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
        self.slider_split.valueChanged.connect(lambda v: self.label_split.setText(f"{v}%"))
        
        split_layout.addWidget(split_label)
        split_layout.addWidget(self.slider_split)
        split_layout.addWidget(self.label_split)
        
        layout.addLayout(split_layout)
        
        parent_layout.addWidget(group)
        
    def create_progress_section(self, parent_layout):
        """Crea la sección de progreso"""
        group = QGroupBox("Progreso del Entrenamiento")
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
