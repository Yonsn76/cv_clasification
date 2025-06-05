from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QProgressBar, QTextEdit,
                             QComboBox, QGroupBox, QGridLayout, QLineEdit,
                             QListWidget, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont
import os
import PyPDF2
from models.cv_classifier import CVClassifier


class MLTrainingWorker(QThread):
    """Worker thread para entrenamiento ML en segundo plano"""
    progress_updated = pyqtSignal(int, str)  # progreso, mensaje
    training_completed = pyqtSignal(dict)  # resultados
    training_failed = pyqtSignal(str)  # error

    def __init__(self, profession_folders, model_name, model_type):
        super().__init__()
        self.profession_folders = profession_folders
        self.model_name = model_name
        self.model_type = model_type
        self.classifier = CVClassifier()

    def extract_text_from_pdf(self, pdf_path):
        """Extrae texto de un archivo PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"Error extrayendo texto de {pdf_path}: {e}")
            return ""

    def run(self):
        """Ejecuta el entrenamiento"""
        try:
            self.progress_updated.emit(10, "Preparando datos de entrenamiento...")

            # Preparar datos de CVs
            cv_data = []
            total_files = sum(len([f for f in os.listdir(folder) if f.lower().endswith('.pdf')])
                            for folder in self.profession_folders.values())

            processed_files = 0

            for profession, folder_path in self.profession_folders.items():
                self.progress_updated.emit(20 + (processed_files * 40 // total_files),
                                         f"Procesando CVs de {profession}...")

                for filename in os.listdir(folder_path):
                    if filename.lower().endswith('.pdf'):
                        file_path = os.path.join(folder_path, filename)
                        text = self.extract_text_from_pdf(file_path)

                        if text:
                            cv_data.append({
                                'text': text,
                                'profession': profession,
                                'filename': filename,
                                'status': 'success'
                            })
                        else:
                            cv_data.append({
                                'text': '',
                                'profession': profession,
                                'filename': filename,
                                'status': 'failed'
                            })

                        processed_files += 1

            self.progress_updated.emit(70, "Entrenando modelo de Machine Learning...")

            # Entrenar modelo
            results = self.classifier.train_model(cv_data, model_type=self.model_type)

            self.progress_updated.emit(90, "Guardando modelo...")

            # Guardar modelo
            save_success = self.classifier.save_model(self.model_name)

            if save_success:
                results['model_saved'] = True
                results['model_name'] = self.model_name
                self.progress_updated.emit(100, "Entrenamiento completado exitosamente!")
                self.training_completed.emit(results)
            else:
                self.training_failed.emit("Error guardando el modelo")

        except Exception as e:
            self.training_failed.emit(f"Error durante el entrenamiento: {str(e)}")


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
                    background-color: #95A5A6;
                }}
            """)
        else:
            print("VistaMLEntrenamiento: No se pudo obtener la ventana principal o atributos de color para aplicar el tema inicial en init_ui.")
        
        layout.addLayout(header_layout)
        
        main_content = QWidget()
        content_layout = QVBoxLayout(main_content)
        content_layout.setSpacing(25)

        self.create_profession_config(content_layout)
        self.create_training_config(content_layout)
        self.create_training_log(content_layout)
        
        layout.addWidget(main_content)
        
        # Inicializar estado
        self.profession_folders = {}
        self.selected_folder = None

    def create_profession_config(self, parent_layout):
        """Crea la sección de configuración de profesiones y datos"""
        group = QGroupBox("1. Configuración de Profesiones y Datos")
        self.widgets_to_update['group_profession'] = group
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

        # Sección para agregar profesiones
        add_layout = QGridLayout()
        add_layout.setSpacing(10)

        name_input = QLineEdit()
        name_input.setPlaceholderText("Ej: Ingeniero de Software, Agrónomo")
        name_input.setStyleSheet("""
            QLineEdit {
                background-color: #34495E;
                color: white;
                border: 1px solid #3498DB;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)

        btn_select_folder = QPushButton("📁 Seleccionar Carpeta de CVs")
        btn_select_folder.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)

        btn_add_profession = QPushButton("➕ Agregar Profesión")
        btn_add_profession.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        btn_add_profession.setEnabled(False)

        add_layout.addWidget(QLabel("Nombre de Profesión:"), 0, 0)
        add_layout.addWidget(name_input, 0, 1, 1, 2)
        add_layout.addWidget(btn_select_folder, 1, 1)
        add_layout.addWidget(btn_add_profession, 1, 2)
        layout.addLayout(add_layout)

        # Lista de profesiones
        list_widget = QListWidget()
        list_widget.setMaximumHeight(180)
        list_widget.setStyleSheet("""
            QListWidget {
                background-color: #34495E;
                color: white;
                border: 1px solid #3498DB;
                border-radius: 5px;
            }
            QListWidget::item {
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #3498DB;
                color: white;
            }
        """)
        layout.addWidget(QLabel("Profesiones y carpetas añadidas:"))
        layout.addWidget(list_widget)

        btn_clear_professions = QPushButton("🗑️ Limpiar Lista de Profesiones")
        btn_clear_professions.setStyleSheet("""
            QPushButton {
                background-color: #C0392B;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #A93226;
            }
        """)
        layout.addWidget(btn_clear_professions, 0, Qt.AlignmentFlag.AlignRight)

        # Guardar referencias
        self.profession_name_input = name_input
        self.btn_select_folder = btn_select_folder
        self.btn_add_profession = btn_add_profession
        self.profession_list = list_widget
        self.btn_clear_professions = btn_clear_professions

        # Conectar señales
        btn_select_folder.clicked.connect(self.select_profession_folder)
        btn_add_profession.clicked.connect(self.add_profession)
        btn_clear_professions.clicked.connect(self.clear_professions)
        name_input.textChanged.connect(lambda text: btn_add_profession.setEnabled(bool(text.strip()) and hasattr(self, 'selected_folder')))

        parent_layout.addWidget(group)

    def create_training_config(self, parent_layout):
        """Crea la sección de configuración del modelo y entrenamiento"""
        group = QGroupBox("2. Configuración del Modelo y Entrenamiento")
        self.widgets_to_update['group_training'] = group
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
        layout = QGridLayout(group)
        layout.setSpacing(10)

        # Nombre del modelo
        layout.addWidget(QLabel("Nombre del modelo:"), 0, 0)
        self.training_model_name_input = QLineEdit()
        self.training_model_name_input.setPlaceholderText("Ej: modelo_tecnologia_rf_2025")
        self.training_model_name_input.setStyleSheet("""
            QLineEdit {
                background-color: #34495E;
                color: white;
                border: 1px solid #8E44AD;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        layout.addWidget(self.training_model_name_input, 0, 1, 1, 2)

        # Tipo de algoritmo
        layout.addWidget(QLabel("Tipo de algoritmo (ML):"), 1, 0)
        self.model_type_combo = QComboBox()
        algorithms = [
            ("random_forest", "Random Forest (Recomendado, Equilibrado)"),
            ("logistic_regression", "Regresión Logística (Rápido, Lineal)"),
            ("svm", "Máquina de Vectores de Soporte (SVM)"),
            ("naive_bayes", "Naive Bayes (Simple, Bueno para texto)")
        ]
        for value, display_name in algorithms:
            self.model_type_combo.addItem(display_name, value)
        self.model_type_combo.setCurrentIndex(0)
        self.model_type_combo.setStyleSheet("""
            QComboBox {
                background-color: #34495E;
                color: white;
                border: 1px solid #8E44AD;
                border-radius: 5px;
                padding: 5px;
            }
            QComboBox QListView {
                background-color: #2C3E50;
                border: 1px solid #8E44AD;
                color: white;
            }
            QComboBox QListView::item {
                padding: 5px;
            }
            QComboBox QListView::item:selected {
                background-color: #8E44AD;
            }
        """)
        layout.addWidget(self.model_type_combo, 1, 1, 1, 2)

        # Botón de entrenamiento
        self.btn_train = QPushButton("🚀 Iniciar Entrenamiento del Modelo")
        self.btn_train.setStyleSheet("""
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
        self.btn_train.clicked.connect(self.start_training)
        self.btn_train.setEnabled(False)
        layout.addWidget(self.btn_train, 2, 1, 1, 2, Qt.AlignmentFlag.AlignRight)

        # Aplicar estilos a las etiquetas
        for i in range(layout.rowCount()):
            item = layout.itemAtPosition(i, 0)
            if item and item.widget() and isinstance(item.widget(), QLabel):
                item.widget().setStyleSheet("color: #BDC3C7; font-size: 13px;")

        parent_layout.addWidget(group)

    def create_training_log(self, parent_layout):
        """Crea la sección de registro y progreso del entrenamiento"""
        group = QGroupBox("3. Registro y Progreso del Entrenamiento")
        self.widgets_to_update['group_log'] = group
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
        layout = QVBoxLayout(group)
        layout.setSpacing(15)

        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #E74C3C;
                border-radius: 10px;
                text-align: center;
                background-color: #34495E;
                color: white;
                font-weight: 500;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #E74C3C;
                border-radius: 9px;
                margin: 1px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Log de entrenamiento
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMinimumHeight(150)
        self.training_log.setPlaceholderText("El progreso detallado del entrenamiento aparecerá aquí...")
        self.training_log.setStyleSheet("""
            QTextEdit {
                background-color: #2C3E50;
                color: #BDC3C7;
                border: 1px solid #E74C3C;
                border-radius: 6px;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 9.5pt;
                padding: 8px;
            }
        """)
        layout.addWidget(self.training_log)

        parent_layout.addWidget(group)
        
    def select_profession_folder(self):
        """Selecciona una carpeta para la profesión"""
        folder_path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de CVs")
        if folder_path:
            self.selected_folder = folder_path
            self.btn_add_profession.setEnabled(bool(self.profession_name_input.text().strip()))
            self.training_log.append(f"📁 Carpeta seleccionada: {folder_path}")

    def add_profession(self):
        """Agrega una profesión con su carpeta asociada"""
        profession = self.profession_name_input.text().strip()
        if not profession or not hasattr(self, 'selected_folder'):
            return

        # Verificar si ya existe
        if profession in self.profession_folders:
            QMessageBox.information(self, "Profesión Existente", f"La profesión '{profession}' ya ha sido agregada.")
            return

        # Contar archivos PDF
        pdf_count = self._count_pdf_files(self.selected_folder)

        # Agregar a la lista
        self.profession_folders[profession] = self.selected_folder
        item_text = f"Profesión: {profession} | Carpeta: {self.selected_folder} (PDFs: {pdf_count})"
        self.profession_list.addItem(item_text)

        # Limpiar campos
        self.profession_name_input.clear()
        self.selected_folder = None
        self.btn_add_profession.setEnabled(False)

        # Habilitar entrenamiento si hay profesiones
        self.btn_train.setEnabled(len(self.profession_folders) > 0)

        self.training_log.append(f"➕ Profesión agregada: {profession}")

    def clear_professions(self):
        """Limpia la lista de profesiones"""
        self.profession_folders.clear()
        self.profession_list.clear()
        self.btn_train.setEnabled(False)
        self.training_log.append("🗑️ Lista de profesiones limpiada")

    def _count_pdf_files(self, folder_path):
        """Cuenta archivos PDF en una carpeta"""
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

    def start_training(self):
        """Inicia el proceso de entrenamiento"""
        if not self.profession_folders:
            QMessageBox.warning(self, "Sin Datos", "Por favor agregue al menos una profesión con su carpeta de datos.")
            return

        model_name = self.training_model_name_input.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Nombre Requerido", "Por favor ingrese un nombre para el modelo.")
            return

        # Obtener tipo de modelo seleccionado
        model_type = self.model_type_combo.currentData()

        # Mostrar progreso
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.training_log.clear()
        self.training_log.append("🔄 Iniciando entrenamiento de Machine Learning...")
        self.training_log.append(f"📊 Modelo: {model_name}")
        self.training_log.append(f"🤖 Algoritmo: {self.model_type_combo.currentText()}")
        self.training_log.append(f"📁 Profesiones: {len(self.profession_folders)}")
        self.training_log.append("=" * 50)

        # Deshabilitar botón de entrenamiento
        self.btn_train.setEnabled(False)

        # Crear y configurar worker thread
        self.training_worker = MLTrainingWorker(
            self.profession_folders,
            model_name,
            model_type
        )

        # Conectar señales
        self.training_worker.progress_updated.connect(self.update_training_progress)
        self.training_worker.training_completed.connect(self.on_training_completed)
        self.training_worker.training_failed.connect(self.on_training_failed)

        # Iniciar entrenamiento
        self.training_worker.start()
        self.entrenamiento_iniciado.emit()

    def update_training_progress(self, progress, message):
        """Actualiza el progreso del entrenamiento"""
        self.progress_bar.setValue(progress)
        self.training_log.append(f"⏳ {message}")

    def on_training_completed(self, results):
        """Maneja la finalización exitosa del entrenamiento"""
        self.training_log.append("=" * 50)
        self.training_log.append("✅ ¡Entrenamiento completado exitosamente!")
        self.training_log.append(f"📈 Accuracy: {results.get('accuracy', 0):.3f}")
        self.training_log.append(f"📊 Muestras entrenamiento: {results.get('train_samples', 0)}")
        self.training_log.append(f"🧪 Muestras prueba: {results.get('test_samples', 0)}")
        self.training_log.append(f"🔧 Características: {results.get('features', 0)}")
        self.training_log.append(f"👥 Clases: {', '.join(results.get('classes', []))}")
        self.training_log.append("💾 Modelo guardado correctamente")

        self.btn_train.setEnabled(True)
        self.entrenamiento_completado.emit()

    def on_training_failed(self, error_message):
        """Maneja errores durante el entrenamiento"""
        self.training_log.append("=" * 50)
        self.training_log.append("❌ Error durante el entrenamiento:")
        self.training_log.append(f"   {error_message}")

        self.btn_train.setEnabled(True)
        QMessageBox.critical(self, "Error de Entrenamiento", f"Error durante el entrenamiento:\n{error_message}")



        




