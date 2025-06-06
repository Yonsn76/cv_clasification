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
    progress_updated = pyqtSignal(int, str)
    training_completed = pyqtSignal(dict)
    training_failed = pyqtSignal(str)

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
                            cv_data.append({'text': text, 'profession': profession, 'filename': filename, 'status': 'success'})
                        else:
                            cv_data.append({'text': '', 'profession': profession, 'filename': filename, 'status': 'failed'})
                        processed_files += 1

            self.progress_updated.emit(70, "Entrenando modelo de Machine Learning...")
            results = self.classifier.train_model(cv_data, model_type=self.model_type)
            self.progress_updated.emit(90, "Guardando modelo...")
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
        self.init_ui()
        self.profession_folders = {}
        self.selected_folder = None
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(20)
        
        header_layout = QHBoxLayout()
        self.btn_volver = QPushButton("← Volver")
        self.btn_volver.setFixedSize(100, 35)
        self.btn_volver.clicked.connect(self.volver_solicitado.emit)
        header_layout.addWidget(self.btn_volver)
        header_layout.addStretch()
        
        title_label = QLabel("🤖 Entrenamiento Machine Learning")
        title_font = QFont(); title_font.setPointSize(20); title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        main_content = QWidget()
        content_layout = QVBoxLayout(main_content)
        content_layout.setSpacing(25)
        self.create_profession_config(content_layout)
        self.create_training_config(content_layout)
        self.create_training_log(content_layout)
        layout.addWidget(main_content)

    def create_profession_config(self, parent_layout):
        group = QGroupBox("1. Configuración de Profesiones y Datos")
        group.setObjectName("ProfessionGroupML")
        layout = QVBoxLayout(group)
        layout.setSpacing(15)

        add_layout = QGridLayout(); add_layout.setSpacing(10)
        self.profession_name_input = QLineEdit()
        self.profession_name_input.setPlaceholderText("Ej: Ingeniero de Software, Agrónomo")
        self.btn_select_folder = QPushButton("📁 Seleccionar Carpeta de CVs")
        self.btn_add_profession = QPushButton("➕ Agregar Profesión")
        self.btn_add_profession.setEnabled(False)
        add_layout.addWidget(QLabel("Nombre de Profesión:"), 0, 0)
        add_layout.addWidget(self.profession_name_input, 0, 1, 1, 2)
        add_layout.addWidget(self.btn_select_folder, 1, 1)
        add_layout.addWidget(self.btn_add_profession, 1, 2)
        layout.addLayout(add_layout)

        self.profession_list = QListWidget(); self.profession_list.setMaximumHeight(180)
        layout.addWidget(QLabel("Profesiones y carpetas añadidas:"))
        layout.addWidget(self.profession_list)
        self.btn_clear_professions = QPushButton("🗑️ Limpiar Lista de Profesiones")
        layout.addWidget(self.btn_clear_professions, 0, Qt.AlignmentFlag.AlignRight)

        self.btn_select_folder.clicked.connect(self.select_profession_folder)
        self.btn_add_profession.clicked.connect(self.add_profession)
        self.btn_clear_professions.clicked.connect(self.clear_professions)
        self.profession_name_input.textChanged.connect(lambda text: self.btn_add_profession.setEnabled(bool(text.strip()) and hasattr(self, 'selected_folder')))
        parent_layout.addWidget(group)

    def create_training_config(self, parent_layout):
        group = QGroupBox("2. Configuración del Modelo y Entrenamiento")
        group.setObjectName("TrainingGroupML")
        layout = QGridLayout(group); layout.setSpacing(10)

        layout.addWidget(QLabel("Nombre del modelo:"), 0, 0)
        self.training_model_name_input = QLineEdit()
        self.training_model_name_input.setPlaceholderText("Ej: modelo_tecnologia_rf_2025")
        layout.addWidget(self.training_model_name_input, 0, 1, 1, 2)

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
        layout.addWidget(self.model_type_combo, 1, 1, 1, 2)

        self.btn_train = QPushButton("🚀 Iniciar Entrenamiento del Modelo")
        self.btn_train.clicked.connect(self.start_training)
        self.btn_train.setEnabled(False)
        layout.addWidget(self.btn_train, 2, 1, 1, 2, Qt.AlignmentFlag.AlignRight)
        parent_layout.addWidget(group)

    def create_training_log(self, parent_layout):
        group = QGroupBox("3. Registro y Progreso del Entrenamiento")
        group.setObjectName("LogGroupML")
        layout = QVBoxLayout(group); layout.setSpacing(15)

        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False); self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.training_log = QTextEdit()
        self.training_log.setObjectName("MLTrainingLog")
        self.training_log.setReadOnly(True)
        self.training_log.setMinimumHeight(150)
        self.training_log.setPlaceholderText("El progreso detallado del entrenamiento aparecerá aquí...")
        layout.addWidget(self.training_log)
        parent_layout.addWidget(group)
        
    def select_profession_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de CVs")
        if folder_path:
            self.selected_folder = folder_path
            self.btn_add_profession.setEnabled(bool(self.profession_name_input.text().strip()))
            self.training_log.append(f"📁 Carpeta seleccionada: {folder_path}")

    def add_profession(self):
        profession = self.profession_name_input.text().strip()
        if not profession or not hasattr(self, 'selected_folder'): return
        if profession in self.profession_folders:
            QMessageBox.information(self, "Profesión Existente", f"La profesión '{profession}' ya ha sido agregada.")
            return

        pdf_count = self._count_pdf_files(self.selected_folder)
        self.profession_folders[profession] = self.selected_folder
        self.profession_list.addItem(f"Profesión: {profession} | Carpeta: {self.selected_folder} (PDFs: {pdf_count})")
        self.profession_name_input.clear()
        self.selected_folder = None
        self.btn_add_profession.setEnabled(False)
        self.btn_train.setEnabled(len(self.profession_folders) > 0)
        self.training_log.append(f"➕ Profesión agregada: {profession}")

    def clear_professions(self):
        self.profession_folders.clear(); self.profession_list.clear()
        self.btn_train.setEnabled(False)
        self.training_log.append("🗑️ Lista de profesiones limpiada")

    def _count_pdf_files(self, folder_path):
        if not os.path.isdir(folder_path): return 0
        try:
            return sum(1 for fname in os.listdir(folder_path) if fname.lower().endswith('.pdf'))
        except Exception as e:
            print(f"Error contando archivos PDF en {folder_path}: {e}")
            return 0

    def start_training(self):
        if not self.profession_folders:
            QMessageBox.warning(self, "Sin Datos", "Por favor agregue al menos una profesión con su carpeta de datos.")
            return
        model_name = self.training_model_name_input.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Nombre Requerido", "Por favor ingrese un nombre para el modelo.")
            return

        model_type = self.model_type_combo.currentData()
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0); self.training_log.clear()
        self.training_log.append("🔄 Iniciando entrenamiento de Machine Learning...")
        self.training_log.append(f"📊 Modelo: {model_name}")
        self.training_log.append(f"🤖 Algoritmo: {self.model_type_combo.currentText()}")
        self.training_log.append(f"📁 Profesiones: {len(self.profession_folders)}")
        self.training_log.append("=" * 50)
        self.btn_train.setEnabled(False)

        self.training_worker = MLTrainingWorker(self.profession_folders, model_name, model_type)
        self.training_worker.progress_updated.connect(self.update_training_progress)
        self.training_worker.training_completed.connect(self.on_training_completed)
        self.training_worker.training_failed.connect(self.on_training_failed)
        self.training_worker.start()
        self.entrenamiento_iniciado.emit()

    def update_training_progress(self, progress, message):
        self.progress_bar.setValue(progress)
        self.training_log.append(f"⏳ {message}")

    def on_training_completed(self, results):
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
        self.training_log.append("=" * 50)
        self.training_log.append("❌ Error durante el entrenamiento:")
        self.training_log.append(f"   {error_message}")
        self.btn_train.setEnabled(True)
        QMessageBox.critical(self, "Error de Entrenamiento", f"Error durante el entrenamiento:\n{error_message}")