# vista_dl_entrenamiento.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QProgressBar, QTextEdit,
                             QComboBox, QGroupBox, QGridLayout, QLineEdit,
                             QListWidget, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont
import os
import PyPDF2
from models.deep_learning_classifier import DeepLearningClassifier


class DLTrainingWorker(QThread):
    """Worker thread para entrenamiento Deep Learning en segundo plano"""
    progress_updated = pyqtSignal(int, str)  # progreso, mensaje
    epoch_updated = pyqtSignal(int, int, dict)  # época actual, total épocas, métricas
    training_completed = pyqtSignal(dict)  # resultados
    training_failed = pyqtSignal(str)  # error

    def __init__(self, profession_folders, model_name, model_type, epochs, batch_size):
        super().__init__()
        self.profession_folders = profession_folders
        self.model_name = model_name
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.classifier = DeepLearningClassifier()

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
        """Ejecuta el entrenamiento de Deep Learning"""
        try:
            self.progress_updated.emit(10, "Preparando datos para Deep Learning...")

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

                        processed_files += 1

            self.progress_updated.emit(70, f"Entrenando modelo {self.model_type.upper()}...")

            # Entrenar modelo
            results = self.classifier.train_model(
                cv_data,
                model_type=self.model_type,
                epochs=self.epochs,
                batch_size=self.batch_size
            )

            if results.get('success', False):
                self.progress_updated.emit(90, "Guardando modelo Deep Learning...")

                # Guardar modelo
                save_success = self.classifier.save_model(self.model_name)

                if save_success:
                    results['model_saved'] = True
                    results['model_name'] = self.model_name
                    self.progress_updated.emit(100, "Entrenamiento Deep Learning completado!")
                    self.training_completed.emit(results)
                else:
                    self.training_failed.emit("Error guardando el modelo Deep Learning")
            else:
                self.training_failed.emit(results.get('error', 'Error desconocido durante el entrenamiento'))

        except Exception as e:
            self.training_failed.emit(f"Error durante el entrenamiento DL: {str(e)}")


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

        # Secciones principales
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
        group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
                border: 2px solid #1ABC9C;
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
        name_input.setPlaceholderText("Ej: Científico de Datos, Analista de IA")
        name_input.setStyleSheet("""
            QLineEdit {
                background-color: #34495E;
                color: white;
                border: 1px solid #1ABC9C;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)

        btn_select_folder = QPushButton("📁 Seleccionar Carpeta de CVs")
        btn_select_folder.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)

        btn_add_profession = QPushButton("➕ Agregar Profesión")
        btn_add_profession.setStyleSheet("""
            QPushButton {
                background-color: #2ECC71;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 15px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27AE60;
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
        layout.addWidget(QLabel("Profesiones y carpetas añadidas:"))
        layout.addWidget(list_widget)

        btn_clear_professions = QPushButton("🗑️ Limpiar Lista de Profesiones")
        btn_clear_professions.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C0392B;
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
        group = QGroupBox("2. Configuración del Modelo Deep Learning")
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
        layout.setSpacing(10)

        # Nombre del modelo
        layout.addWidget(QLabel("Nombre del modelo DL:"), 0, 0)
        self.dl_model_name_input = QLineEdit()
        self.dl_model_name_input.setPlaceholderText("Ej: modelo_dl_bert_multilingue")
        self.dl_model_name_input.setStyleSheet("""
            QLineEdit {
                background-color: #34495E;
                color: white;
                border: 1px solid #9B59B6;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        layout.addWidget(self.dl_model_name_input, 0, 1, 1, 3)

        # Tipo de arquitectura
        layout.addWidget(QLabel("Tipo de arquitectura DL:"), 1, 0)
        self.dl_model_type_combo = QComboBox()
        dl_models = [
            ("bert", "BERT (Transformer, Alta Precisión)"),
            ("lstm", "LSTM (Red Neuronal Recurrente)"),
            ("cnn", "CNN (Red Neuronal Convolucional para Texto)")
        ]
        for value, display_name in dl_models:
            self.dl_model_type_combo.addItem(display_name, value)
        self.dl_model_type_combo.setCurrentIndex(0)
        self.dl_model_type_combo.setStyleSheet("""
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
        layout.addWidget(self.dl_model_type_combo, 1, 1, 1, 3)

        # Épocas
        layout.addWidget(QLabel("Número de Épocas:"), 2, 0)
        self.dl_epochs_input = QLineEdit("5")
        self.dl_epochs_input.setMaximumWidth(100)
        self.dl_epochs_input.setStyleSheet("""
            QLineEdit {
                background-color: #34495E;
                color: white;
                border: 1px solid #9B59B6;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.dl_epochs_input, 2, 1)

        # Batch size
        layout.addWidget(QLabel("Tamaño de Batch:"), 2, 2)
        self.dl_batch_size_input = QLineEdit("16")
        self.dl_batch_size_input.setMaximumWidth(100)
        self.dl_batch_size_input.setStyleSheet("""
            QLineEdit {
                background-color: #34495E;
                color: white;
                border: 1px solid #9B59B6;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.dl_batch_size_input, 2, 3)

        # Botón de entrenamiento
        self.btn_dl_train = QPushButton("🧠 Iniciar Entrenamiento Deep Learning")
        self.btn_dl_train.setStyleSheet("""
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
        self.btn_dl_train.clicked.connect(self.start_dl_training)
        self.btn_dl_train.setEnabled(False)
        layout.addWidget(self.btn_dl_train, 3, 1, 1, 3, Qt.AlignmentFlag.AlignRight)

        # Aplicar estilos a las etiquetas
        for i in range(layout.rowCount()):
            item = layout.itemAtPosition(i, 0)
            if item and item.widget() and isinstance(item.widget(), QLabel):
                item.widget().setStyleSheet("color: #BDC3C7; font-size: 13px;")

        parent_layout.addWidget(group)

    def create_training_log(self, parent_layout):
        """Crea la sección de registro y progreso del entrenamiento"""
        group = QGroupBox("3. Registro y Progreso del Entrenamiento")
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
    def select_profession_folder(self):
        """Selecciona una carpeta para la profesión"""
        folder_path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de CVs")
        if folder_path:
            self.selected_folder = folder_path
            self.btn_add_profession.setEnabled(bool(self.profession_name_input.text().strip()))
            self.log_entrenamiento.append(f"📁 Carpeta seleccionada: {folder_path}")

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
        self.btn_dl_train.setEnabled(len(self.profession_folders) > 0)

        self.log_entrenamiento.append(f"➕ Profesión agregada: {profession}")

    def clear_professions(self):
        """Limpia la lista de profesiones"""
        self.profession_folders.clear()
        self.profession_list.clear()
        self.btn_dl_train.setEnabled(False)
        self.log_entrenamiento.append("🗑️ Lista de profesiones limpiada")

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
    def start_dl_training(self):
        """Inicia el proceso de entrenamiento de Deep Learning"""
        if not self.profession_folders:
            QMessageBox.warning(self, "Sin Datos", "Por favor agregue al menos una profesión con su carpeta de datos.")
            return

        model_name = self.dl_model_name_input.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Nombre Requerido", "Por favor ingrese un nombre para el modelo.")
            return

        # Validar parámetros
        try:
            epochs = int(self.dl_epochs_input.text())
            batch_size = int(self.dl_batch_size_input.text())
        except ValueError:
            QMessageBox.warning(self, "Parámetros Inválidos", "Por favor ingrese valores numéricos válidos para épocas y batch size.")
            return

        # Obtener tipo de modelo
        model_type = self.dl_model_type_combo.currentData()

        # Mostrar progreso
        self.progress_bar.setValue(0)
        self.log_entrenamiento.clear()
        self.log_entrenamiento.append("🔄 Iniciando entrenamiento Deep Learning...")
        self.log_entrenamiento.append(f"🧠 Modelo: {model_name}")
        self.log_entrenamiento.append(f"🏗️ Arquitectura: {self.dl_model_type_combo.currentText()}")
        self.log_entrenamiento.append(f"📊 Épocas: {epochs}")
        self.log_entrenamiento.append(f"📦 Batch Size: {batch_size}")
        self.log_entrenamiento.append(f"📁 Profesiones: {len(self.profession_folders)}")
        self.log_entrenamiento.append("=" * 50)

        # Actualizar label de época
        self.label_epoca.setText(f"Época: 0 / {epochs}")

        # Resetear métricas
        self.label_loss.setText("Loss: --")
        self.label_accuracy.setText("Accuracy: --")
        self.label_val_loss.setText("Val Loss: --")
        self.label_val_accuracy.setText("Val Accuracy: --")

        # Deshabilitar botón
        self.btn_dl_train.setEnabled(False)

        # Crear y configurar worker thread
        self.dl_training_worker = DLTrainingWorker(
            self.profession_folders,
            model_name,
            model_type,
            epochs,
            batch_size
        )

        # Conectar señales
        self.dl_training_worker.progress_updated.connect(self.update_dl_training_progress)
        self.dl_training_worker.epoch_updated.connect(self.update_epoch_metrics)
        self.dl_training_worker.training_completed.connect(self.on_dl_training_completed)
        self.dl_training_worker.training_failed.connect(self.on_dl_training_failed)

        # Iniciar entrenamiento
        self.dl_training_worker.start()
        self.entrenamiento_iniciado.emit()

    def update_dl_training_progress(self, progress, message):
        """Actualiza el progreso del entrenamiento"""
        self.progress_bar.setValue(progress)
        self.log_entrenamiento.append(f"⏳ {message}")

    def update_epoch_metrics(self, current_epoch, total_epochs, metrics):
        """Actualiza las métricas de época"""
        self.label_epoca.setText(f"Época: {current_epoch} / {total_epochs}")

        if metrics:
            self.label_loss.setText(f"Loss: {metrics.get('loss', '--')}")
            self.label_accuracy.setText(f"Accuracy: {metrics.get('accuracy', '--')}")
            self.label_val_loss.setText(f"Val Loss: {metrics.get('val_loss', '--')}")
            self.label_val_accuracy.setText(f"Val Accuracy: {metrics.get('val_accuracy', '--')}")

    def on_dl_training_completed(self, results):
        """Maneja la finalización exitosa del entrenamiento DL"""
        self.log_entrenamiento.append("=" * 50)
        self.log_entrenamiento.append("✅ ¡Entrenamiento Deep Learning completado!")
        self.log_entrenamiento.append(f"🎯 Accuracy final: {results.get('accuracy', 0):.3f}")
        self.log_entrenamiento.append(f"📊 Épocas entrenadas: {results.get('epochs_trained', 0)}")
        self.log_entrenamiento.append(f"🏗️ Tipo de modelo: {results.get('model_type', 'N/A')}")
        self.log_entrenamiento.append(f"👥 Clases: {results.get('num_classes', 0)}")
        self.log_entrenamiento.append("💾 Modelo guardado correctamente")

        self.btn_dl_train.setEnabled(True)
        self.entrenamiento_completado.emit()

    def on_dl_training_failed(self, error_message):
        """Maneja errores durante el entrenamiento DL"""
        self.log_entrenamiento.append("=" * 50)
        self.log_entrenamiento.append("❌ Error durante el entrenamiento Deep Learning:")
        self.log_entrenamiento.append(f"   {error_message}")

        self.btn_dl_train.setEnabled(True)
        QMessageBox.critical(self, "Error de Entrenamiento DL", f"Error durante el entrenamiento:\n{error_message}")


