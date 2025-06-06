from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QProgressBar, QTextEdit,
                             QComboBox, QGroupBox, QGridLayout, QLineEdit,
                             QListWidget, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QDragEnterEvent, QDropEvent
import os
import PyPDF2
from models.deep_learning_classifier import DeepLearningClassifier


class DropGroupBox(QGroupBox):
    """Un QGroupBox que acepta carpetas arrastradas y soltadas."""
    folder_dropped = pyqtSignal(str)

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setAcceptDrops(True)
        self.setProperty("dropping", "false")

    def dragEnterEvent(self, event: QDragEnterEvent):
        mime_data = event.mimeData()
        if mime_data.hasUrls() and len(mime_data.urls()) == 1:
            url = mime_data.urls()[0]
            if url.isLocalFile() and os.path.isdir(url.toLocalFile()):
                event.acceptProposedAction()
                self.setProperty("dropping", "true")
                self._update_style()

    def dragLeaveEvent(self, event):
        self.setProperty("dropping", "false")
        self._update_style()

    def dropEvent(self, event: QDropEvent):
        self.setProperty("dropping", "false")
        self._update_style()
        url = event.mimeData().urls()[0]
        folder_path = url.toLocalFile()
        self.folder_dropped.emit(folder_path)
        event.acceptProposedAction()
        
    def _update_style(self):
        self.style().unpolish(self)
        self.style().polish(self)


class DLTrainingWorker(QThread):
    progress_updated = pyqtSignal(int, str)
    epoch_updated = pyqtSignal(int, int, dict)
    training_completed = pyqtSignal(dict)
    training_failed = pyqtSignal(str)

    def __init__(self, profession_folders, model_name, model_type, epochs, batch_size):
        super().__init__()
        self.profession_folders = profession_folders
        self.model_name = model_name
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.classifier = DeepLearningClassifier()

    def extract_text_from_pdf(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = "".join(page.extract_text() + "\n" for page in reader.pages)
                return text.strip()
        except Exception:
            return ""

    def run(self):
        try:
            self.progress_updated.emit(10, "Preparando datos...")
            cv_data, processed_files = [], 0
            total_files = sum(len([f for f in os.listdir(p) if f.lower().endswith('.pdf')]) for p in self.profession_folders.values())
            
            for profession, folder_path in self.profession_folders.items():
                if not os.path.isdir(folder_path): continue
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith('.pdf'):
                        file_path = os.path.join(folder_path, filename)
                        text = self.extract_text_from_pdf(file_path)
                        status = 'success' if text else 'failed'
                        cv_data.append({'text': text, 'profession': profession, 'filename': filename, 'status': status})
                        processed_files += 1
                        progress = 20 + (processed_files * 40 // total_files if total_files > 0 else 0)
                        self.progress_updated.emit(progress, f"Procesando: {filename}")

            self.progress_updated.emit(70, f"Entrenando modelo {self.model_type.upper()}...")
            results = self.classifier.train_model(cv_data, model_type=self.model_type, epochs=self.epochs, batch_size=self.batch_size)
            if results.get('success', False):
                self.progress_updated.emit(90, "Guardando modelo...")
                save_success = self.classifier.save_model(self.model_name)
                if save_success:
                    results['model_saved'] = True
                    results['model_name'] = self.model_name
                    self.progress_updated.emit(100, "¡Entrenamiento completado!")
                    self.training_completed.emit(results)
                else:
                    self.training_failed.emit("Error al guardar el modelo")
            else:
                self.training_failed.emit(results.get('error', 'Error desconocido'))
        except Exception as e:
            self.training_failed.emit(f"Error durante el entrenamiento: {str(e)}")


class VistaDLEntrenamiento(QWidget):
    entrenamiento_iniciado = pyqtSignal()
    entrenamiento_completado = pyqtSignal()
    volver_solicitado = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VistaDLEntrenamiento")
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
        
        title_label = QLabel("🧠 Entrenamiento Deep Learning")
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
        group = DropGroupBox("1. Configuración de Profesiones y Datos")
        group.setObjectName("ProfessionGroupDL")
        group.folder_dropped.connect(self._handle_folder_selection)
        layout = QGridLayout(group); layout.setSpacing(10)

        layout.addWidget(QLabel("Nombre de Profesión:"), 0, 0, 1, 2)
        self.profession_name_input = QLineEdit()
        self.profession_name_input.setPlaceholderText("Ej: Científico de Datos, Analista de IA")
        layout.addWidget(self.profession_name_input, 1, 0, 1, 2)

        layout.addWidget(QLabel("Carpeta de CVs:"), 2, 0, 1, 2)
        self.selected_folder_label = QLabel("Ninguna carpeta seleccionada.")
        self.selected_folder_label.setObjectName("SelectedFolderLabel")
        self.selected_folder_label.setProperty("selected", "false")
        self.selected_folder_label.setWordWrap(True)
        layout.addWidget(self.selected_folder_label, 3, 0, 1, 2)

        self.btn_select_folder = QPushButton("📁 Seleccionar Carpeta")
        self.btn_select_folder.clicked.connect(self.select_profession_folder)
        layout.addWidget(self.btn_select_folder, 4, 0)
        
        drop_label = QLabel("... o arrastre la carpeta aquí.")
        drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_label.setStyleSheet("color: #7F8C8D; font-style: italic;")
        layout.addWidget(drop_label, 4, 1)

        self.btn_add_profession = QPushButton("➕ Agregar Profesión")
        self.btn_add_profession.setEnabled(False)
        self.btn_add_profession.clicked.connect(self.add_profession)
        layout.addWidget(self.btn_add_profession, 5, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)

        list_container = QVBoxLayout(); list_container.setSpacing(5)
        list_container.addWidget(QLabel("Profesiones a Entrenar:"))
        self.profession_list = QListWidget()
        list_container.addWidget(self.profession_list)
        self.btn_clear_professions = QPushButton("🗑️ Limpiar Lista")
        self.btn_clear_professions.clicked.connect(self.clear_professions)
        list_container.addWidget(self.btn_clear_professions)
        layout.addLayout(list_container, 0, 2, 6, 1)
        layout.setColumnStretch(2, 1)

        self.profession_name_input.textChanged.connect(self._update_add_button_state)
        parent_layout.addWidget(group)

    def create_training_config(self, parent_layout):
        group = QGroupBox("2. Configuración del Modelo Deep Learning")
        group.setObjectName("TrainingGroupDL")
        layout = QGridLayout(group); layout.setSpacing(10)
        layout.addWidget(QLabel("Nombre del modelo DL:"), 0, 0)
        self.dl_model_name_input = QLineEdit()
        self.dl_model_name_input.setPlaceholderText("Ej: modelo_dl_bert_multilingue")
        layout.addWidget(self.dl_model_name_input, 0, 1, 1, 3)
        layout.addWidget(QLabel("Tipo de arquitectura DL:"), 1, 0)
        self.dl_model_type_combo = QComboBox()
        dl_models = [("bert", "BERT"), ("lstm", "LSTM"), ("cnn", "CNN (Texto)")]
        for value, display_name in dl_models: self.dl_model_type_combo.addItem(display_name, value)
        layout.addWidget(self.dl_model_type_combo, 1, 1, 1, 3)
        layout.addWidget(QLabel("Épocas:"), 2, 0)
        self.dl_epochs_input = QLineEdit("5"); self.dl_epochs_input.setMaximumWidth(100)
        layout.addWidget(self.dl_epochs_input, 2, 1)
        layout.addWidget(QLabel("Batch Size:"), 2, 2)
        self.dl_batch_size_input = QLineEdit("16"); self.dl_batch_size_input.setMaximumWidth(100)
        layout.addWidget(self.dl_batch_size_input, 2, 3)
        self.btn_dl_train = QPushButton("🧠 Iniciar Entrenamiento")
        self.btn_dl_train.clicked.connect(self.start_dl_training)
        self.btn_dl_train.setEnabled(False)
        layout.addWidget(self.btn_dl_train, 3, 1, 1, 3, Qt.AlignmentFlag.AlignRight)
        parent_layout.addWidget(group)

    def create_training_log(self, parent_layout):
        group = QGroupBox("3. Registro y Progreso del Entrenamiento")
        group.setObjectName("LogGroupDL")
        layout = QVBoxLayout(group); layout.setSpacing(15)
        self.label_epoca = QLabel("Época: 0 / 0"); self.label_epoca.setObjectName("DLEpochLabel")
        layout.addWidget(self.label_epoca)
        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        metrics_layout = QHBoxLayout()
        self.label_loss = QLabel("Loss: --"); self.label_loss.setObjectName("DLMetricLabel")
        self.label_accuracy = QLabel("Accuracy: --"); self.label_accuracy.setObjectName("DLMetricLabel")
        self.label_val_loss = QLabel("Val Loss: --"); self.label_val_loss.setObjectName("DLMetricLabel")
        self.label_val_accuracy = QLabel("Val Acc: --"); self.label_val_accuracy.setObjectName("DLMetricLabel")
        metrics_layout.addWidget(self.label_loss); metrics_layout.addWidget(self.label_accuracy); metrics_layout.addWidget(self.label_val_loss); metrics_layout.addWidget(self.label_val_accuracy)
        layout.addLayout(metrics_layout)
        self.log_entrenamiento = QTextEdit()
        self.log_entrenamiento.setObjectName("DLTrainingLog")
        self.log_entrenamiento.setMaximumHeight(120); self.log_entrenamiento.setReadOnly(True)
        self.log_entrenamiento.setPlaceholderText("Los logs del entrenamiento aparecerán aquí...")
        layout.addWidget(self.log_entrenamiento)
        parent_layout.addWidget(group)

    def _update_add_button_state(self):
        has_name = bool(self.profession_name_input.text().strip())
        has_folder = self.selected_folder is not None
        self.btn_add_profession.setEnabled(has_name and has_folder)
        
    def _reset_profession_inputs(self):
        self.profession_name_input.clear()
        self.selected_folder = None
        self.selected_folder_label.setText("Ninguna carpeta seleccionada.")
        self.selected_folder_label.setProperty("selected", "false")
        self.selected_folder_label.style().unpolish(self.selected_folder_label)
        self.selected_folder_label.style().polish(self.selected_folder_label)
        self._update_add_button_state()
        
    def _handle_folder_selection(self, folder_path: str):
        self.selected_folder = folder_path
        self.selected_folder_label.setText(folder_path)
        self.selected_folder_label.setProperty("selected", "true")
        self.selected_folder_label.style().unpolish(self.selected_folder_label)
        self.selected_folder_label.style().polish(self.selected_folder_label)
        self.log_entrenamiento.append(f"📁 Carpeta lista: {folder_path}")
        self._update_add_button_state()
        
    def select_profession_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de CVs")
        if folder_path: self._handle_folder_selection(folder_path)

    def add_profession(self):
        profession = self.profession_name_input.text().strip()
        if not profession or not self.selected_folder: return
        if profession in self.profession_folders:
            QMessageBox.warning(self, "Profesión Existente", f"'{profession}' ya ha sido agregada.")
            return

        pdf_count = self._count_pdf_files(self.selected_folder)
        if pdf_count == 0:
            QMessageBox.warning(self, "Carpeta Vacía", "La carpeta seleccionada no contiene archivos PDF.")
            return

        self.profession_folders[profession] = self.selected_folder
        self.profession_list.addItem(f"• {profession} (PDFs: {pdf_count})")
        self.log_entrenamiento.append(f"➕ Profesión agregada: {profession}")
        self._reset_profession_inputs()
        self.btn_dl_train.setEnabled(len(self.profession_folders) > 0)

    def clear_professions(self):
        reply = QMessageBox.question(self, "Confirmar", "¿Seguro que quieres limpiar la lista?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.profession_folders.clear(); self.profession_list.clear()
            self._reset_profession_inputs()
            self.btn_dl_train.setEnabled(False)
            self.log_entrenamiento.append("🗑️ Lista de profesiones limpiada.")

    def _count_pdf_files(self, folder_path):
        if not os.path.isdir(folder_path): return 0
        try: return sum(1 for f in os.listdir(folder_path) if f.lower().endswith('.pdf'))
        except: return 0
            
    def start_dl_training(self):
        if not self.profession_folders:
            QMessageBox.warning(self, "Sin Datos", "Agregue al menos una profesión.")
            return
        model_name = self.dl_model_name_input.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Nombre Requerido", "Ingrese un nombre para el modelo.")
            return

        try:
            epochs = int(self.dl_epochs_input.text()); batch_size = int(self.dl_batch_size_input.text())
        except ValueError:
            QMessageBox.warning(self, "Parámetros Inválidos", "Épocas y Batch Size deben ser números.")
            return

        model_type = self.dl_model_type_combo.currentData()
        self.progress_bar.setValue(0); self.log_entrenamiento.clear()
        self.log_entrenamiento.append("🚀 Iniciando entrenamiento..."); self.btn_dl_train.setEnabled(False)
        self.dl_training_worker = DLTrainingWorker(self.profession_folders, model_name, model_type, epochs, batch_size)
        self.dl_training_worker.progress_updated.connect(self.update_dl_training_progress)
        self.dl_training_worker.epoch_updated.connect(self.update_epoch_metrics)
        self.dl_training_worker.training_completed.connect(self.on_dl_training_completed)
        self.dl_training_worker.training_failed.connect(self.on_dl_training_failed)
        self.dl_training_worker.start()
        self.entrenamiento_iniciado.emit()

    def update_dl_training_progress(self, progress, message):
        self.progress_bar.setValue(progress)
        self.log_entrenamiento.append(f"⏳ {message}")

    def update_epoch_metrics(self, current_epoch, total_epochs, metrics):
        self.label_epoca.setText(f"Época: {current_epoch} / {total_epochs}")
        if metrics:
            self.label_loss.setText(f"Loss: {metrics.get('loss', '--'):.4f}")
            self.label_accuracy.setText(f"Acc: {metrics.get('accuracy', '--'):.4f}")
            self.label_val_loss.setText(f"Val Loss: {metrics.get('val_loss', '--'):.4f}")
            self.label_val_accuracy.setText(f"Val Acc: {metrics.get('val_accuracy', '--'):.4f}")

    def on_dl_training_completed(self, results):
        self.log_entrenamiento.append("=" * 50)
        self.log_entrenamiento.append("✅ ¡Entrenamiento Deep Learning completado!")
        self.log_entrenamiento.append(f"🎯 Accuracy final: {results.get('accuracy', 0):.3f}")
        self.log_entrenamiento.append("💾 Modelo guardado correctamente.")
        self.btn_dl_train.setEnabled(True)
        self.entrenamiento_completado.emit()

    def on_dl_training_failed(self, error_message):
        self.log_entrenamiento.append("=" * 50)
        self.log_entrenamiento.append(f"❌ Error en entrenamiento DL: {error_message}")
        self.btn_dl_train.setEnabled(True)
        QMessageBox.critical(self, "Error de Entrenamiento DL", f"Error:\n{error_message}")