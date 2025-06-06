# vista_importar_exportar.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QScrollArea, QProgressBar,
                             QComboBox, QCheckBox, QGroupBox, QGridLayout,
                             QTextEdit, QFileDialog, QMessageBox, QInputDialog)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont
import os
import zipfile
import tempfile
import shutil
import json
import datetime
from models.cv_classifier import CVClassifier
from models.deep_learning_classifier import DeepLearningClassifier


class ExportWorker(QThread):
    """Worker thread para exportar modelos en segundo plano"""
    progress_updated = pyqtSignal(str)
    export_completed = pyqtSignal(str)
    export_failed = pyqtSignal(str)

    def __init__(self, model_name, export_path, model_type):
        super().__init__()
        self.model_name = model_name
        self.export_path = export_path
        self.model_type = model_type  # 'ml' o 'dl'

    def run(self):
        """Ejecuta la exportación del modelo"""
        try:
            self.progress_updated.emit(f"Iniciando exportación de {self.model_name}...")

            # Determinar directorio fuente
            if self.model_type == 'ml':
                source_dir = os.path.join('saved_models', self.model_name)
            else:
                source_dir = os.path.join('saved_deep_models', self.model_name)

            if not os.path.exists(source_dir):
                self.export_failed.emit(f"No se encontró el modelo: {source_dir}")
                return

            self.progress_updated.emit("Comprimiendo archivos del modelo...")

            # Crear archivo .zip (anteriormente .senati)
            with zipfile.ZipFile(self.export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Agregar metadatos especiales para el archivo
                package_info = {
                    'format_version': '1.0',
                    'model_type': self.model_type,
                    'model_name': self.model_name,
                    'exported_by': 'ClasificaTalento PRO',
                    'export_date': datetime.datetime.now().isoformat()
                }

                zipf.writestr('package_info.json', json.dumps(package_info, indent=2))

                # Agregar todos los archivos del modelo
                for root, dirs, files in os.walk(source_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, source_dir)
                        zipf.write(file_path, arcname)
                        self.progress_updated.emit(f"Agregando: {file}")

            self.export_completed.emit(f"Modelo exportado exitosamente a: {self.export_path}")

        except Exception as e:
            self.export_failed.emit(f"Error durante la exportación: {str(e)}")


class ImportWorker(QThread):
    """Worker thread para importar modelos en segundo plano"""
    progress_updated = pyqtSignal(str)
    import_completed = pyqtSignal(str)
    import_failed = pyqtSignal(str)

    def __init__(self, file_path, target_model_name=None):
        super().__init__()
        self.file_path = file_path
        self.target_model_name = target_model_name

    def run(self):
        """Ejecuta la importación del modelo"""
        try:
            self.progress_updated.emit("Verificando archivo de modelo...")

            if not zipfile.is_zipfile(self.file_path):
                self.import_failed.emit("El archivo no es un formato de modelo válido")
                return

            with zipfile.ZipFile(self.file_path, 'r') as zipf:
                if 'package_info.json' not in zipf.namelist():
                    self.import_failed.emit("El archivo no contiene metadatos de modelo válidos")
                    return

                package_info = json.loads(zipf.read('package_info.json').decode('utf-8'))
                model_type = package_info.get('model_type', 'unknown')
                original_name = package_info.get('model_name', 'imported_model')

                self.progress_updated.emit(f"Importando modelo {model_type.upper()}: {original_name}")
                final_model_name = self.target_model_name if self.target_model_name else original_name

                if model_type == 'ml':
                    target_dir = os.path.join('saved_models', final_model_name)
                elif model_type == 'dl':
                    target_dir = os.path.join('saved_deep_models', final_model_name)
                else:
                    self.import_failed.emit(f"Tipo de modelo desconocido: {model_type}")
                    return

                os.makedirs(target_dir, exist_ok=True)
                self.progress_updated.emit("Extrayendo archivos del modelo...")
                for file_info in zipf.filelist:
                    if file_info.filename != 'package_info.json':
                        zipf.extract(file_info, target_dir)
                        self.progress_updated.emit(f"Extraído: {file_info.filename}")
                self.import_completed.emit(f"Modelo importado exitosamente como: {final_model_name}")
        except Exception as e:
            self.import_failed.emit(f"Error durante la importación: {str(e)}")


class VistaImportarExportar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VistaImportarExportar")

        self.ml_classifier = CVClassifier()
        self.dl_classifier = DeepLearningClassifier()
        self.export_worker = None
        self.import_worker = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(25, 20, 25, 20)
        main_layout.setSpacing(20)

        # Título principal (palabra "SENATI" eliminada)
        title_label = QLabel("📦 Importar / Exportar Modelos 📦")
        title_font = QFont(); title_font.setPointSize(20); title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(25)

        content_layout.addWidget(self.create_import_section())
        content_layout.addWidget(self.create_export_section())
        content_layout.addWidget(self.create_management_section())
        content_layout.addStretch(1)
        
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        self.refresh_model_lists()

    def create_import_section(self):
        section = QGroupBox("📥 Importar Modelos")
        section.setObjectName("ImportGroup")
        layout = QVBoxLayout(section)
        layout.setSpacing(20)

        drop_area = QFrame()
        drop_area.setObjectName("DropArea")
        drop_area.setFixedHeight(150)
        drop_layout = QVBoxLayout(drop_area)
        drop_icon = QLabel("📁")
        drop_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_icon.setStyleSheet("font-size: 48px; margin: 10px;")
        drop_text = QLabel("Arrastra archivos de modelo aquí\n(Formato .zip de ClasificaTalento PRO)")
        drop_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(drop_icon)
        drop_layout.addWidget(drop_text)
        layout.addWidget(drop_area)

        import_buttons_layout = QHBoxLayout()
        select_file_btn = QPushButton("📂 Seleccionar Archivo")
        select_file_btn.setFixedHeight(40)
        select_file_btn.clicked.connect(self.select_import_file)
        refresh_models_btn = QPushButton("🔄 Actualizar Lista")
        refresh_models_btn.setFixedHeight(40)
        refresh_models_btn.clicked.connect(self.refresh_model_lists)
        import_buttons_layout.addWidget(select_file_btn)
        import_buttons_layout.addWidget(refresh_models_btn)
        import_buttons_layout.addStretch()
        layout.addLayout(import_buttons_layout)

        config_layout = QGridLayout()
        config_layout.addWidget(QLabel("Nombre del modelo:"), 0, 0)
        self.import_model_name_combo = QComboBox()
        self.import_model_name_combo.setEditable(True)
        self.import_model_name_combo.setPlaceholderText("Usar nombre original o escribir nuevo...")
        self.import_model_name_combo.addItem("Usar nombre original")
        config_layout.addWidget(self.import_model_name_combo, 0, 1)
        self.validate_checkbox = QCheckBox("Validar integridad del modelo")
        self.validate_checkbox.setChecked(True)
        config_layout.addWidget(self.validate_checkbox, 1, 0, 1, 2)
        self.overwrite_checkbox = QCheckBox("Sobrescribir si el modelo ya existe")
        self.overwrite_checkbox.setChecked(False)
        config_layout.addWidget(self.overwrite_checkbox, 2, 0, 1, 2)
        layout.addLayout(config_layout)
        return section

    def create_export_section(self):
        section = QGroupBox("📤 Exportar Modelos")
        section.setObjectName("ExportGroup")
        layout = QVBoxLayout(section)
        layout.setSpacing(20)
        
        model_selection_layout = QHBoxLayout()
        model_label = QLabel("Modelo a exportar:")
        self.export_model_combo = QComboBox()
        model_selection_layout.addWidget(model_label)
        model_selection_layout.addWidget(self.export_model_combo)
        model_selection_layout.addStretch()
        layout.addLayout(model_selection_layout)

        export_options_layout = QGridLayout()
        format_info_label = QLabel("Formato: .zip (Compatible con ClasificaTalento PRO)")
        format_info_label.setStyleSheet("font-weight: bold;")
        export_options_layout.addWidget(format_info_label, 0, 0, 1, 2)
        self.include_metadata_checkbox = QCheckBox("Incluir metadatos completos (recomendado)")
        self.include_metadata_checkbox.setChecked(True)
        self.include_metadata_checkbox.setEnabled(False)
        export_options_layout.addWidget(self.include_metadata_checkbox, 1, 0, 1, 2)
        self.compress_checkbox = QCheckBox("Compresión optimizada (automática)")
        self.compress_checkbox.setChecked(True)
        self.compress_checkbox.setEnabled(False)
        export_options_layout.addWidget(self.compress_checkbox, 2, 0, 1, 2)
        layout.addLayout(export_options_layout)

        export_buttons_layout = QHBoxLayout()
        export_file_btn = QPushButton("💾 Exportar como .zip")
        export_file_btn.setFixedHeight(40)
        export_file_btn.clicked.connect(self.export_to_file)
        backup_all_btn = QPushButton("📦 Backup Todos los Modelos")
        backup_all_btn.setFixedHeight(40)
        backup_all_btn.clicked.connect(self.backup_all_models)
        export_buttons_layout.addWidget(export_file_btn)
        export_buttons_layout.addWidget(backup_all_btn)
        export_buttons_layout.addStretch()
        layout.addLayout(export_buttons_layout)
        return section

    def create_management_section(self):
        section = QGroupBox("⚙️ Gestión de Modelos")
        section.setObjectName("ManagementGroup")
        layout = QVBoxLayout(section)
        layout.setSpacing(20)
        
        log_label = QLabel("📋 Registro de Actividades:")
        log_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(log_label)

        self.activity_log = QTextEdit()
        self.activity_log.setMaximumHeight(120)
        self.activity_log.setReadOnly(True)
        self.activity_log.setPlaceholderText("Las actividades de importación/exportación aparecerán aquí...")
        self.activity_log.setStyleSheet("font-family: 'Courier New', monospace; font-size: 11px;")
        layout.addWidget(self.activity_log)
        
        management_buttons_layout = QHBoxLayout()
        backup_btn = QPushButton("💾 Backup Completo")
        restore_btn = QPushButton("🔄 Restaurar")
        clean_btn = QPushButton("🧹 Limpiar Cache")
        management_buttons_layout.addWidget(backup_btn)
        management_buttons_layout.addWidget(restore_btn)
        management_buttons_layout.addWidget(clean_btn)
        management_buttons_layout.addStretch()
        layout.addLayout(management_buttons_layout)
        return section

    def refresh_model_lists(self):
        try:
            ml_models = self.ml_classifier.list_available_models()
            self.export_model_combo.clear()

            if not ml_models:
                self.export_model_combo.addItem("No hay modelos disponibles")
                self.activity_log.append("⚠️ No se encontraron modelos para exportar")
                return

            for model in ml_models:
                model_type_prefix = "🧠 DL" if model.get('is_deep_learning', False) else "🤖 ML"
                display_name = f"{model_type_prefix} - {model['display_name']}"
                self.export_model_combo.addItem(display_name, model)
            self.activity_log.append(f"✅ Lista actualizada: {len(ml_models)} modelos encontrados")
        except Exception as e:
            self.activity_log.append(f"❌ Error actualizando lista: {str(e)}")
            self.export_model_combo.clear()
            self.export_model_combo.addItem("Error al cargar modelos")

    def select_import_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo de Modelo para Importar", "", "Archivos Zip (*.zip);;Todos los archivos (*)")
        if file_path:
            self.activity_log.append(f"📁 Archivo seleccionado: {os.path.basename(file_path)}")
            custom_name = self.import_model_name_combo.currentText().strip()
            target_name = None if custom_name == "Usar nombre original" or not custom_name else custom_name
            self.import_worker = ImportWorker(file_path, target_name)
            self.import_worker.progress_updated.connect(self.update_import_progress)
            self.import_worker.import_completed.connect(self.on_import_completed)
            self.import_worker.import_failed.connect(self.on_import_failed)
            self.import_worker.start()

    def export_to_file(self):
        if self.export_model_combo.count() == 0 or self.export_model_combo.currentText() == "No hay modelos disponibles":
            QMessageBox.warning(self, "Sin Modelos", "No hay modelos disponibles para exportar.")
            return

        model_data = self.export_model_combo.currentData()
        if not model_data:
            QMessageBox.warning(self, "Error", "No se pudo obtener información del modelo seleccionado.")
            return

        model_name = model_data['name']
        suggested_filename = f"{model_name}.zip"
        file_path, _ = QFileDialog.getSaveFileName(self, "Exportar Modelo como .zip", suggested_filename, "Archivos Zip (*.zip)")

        if file_path:
            if not file_path.lower().endswith('.zip'):
                file_path += '.zip'
            model_type = 'dl' if model_data.get('is_deep_learning', False) else 'ml'
            self.activity_log.append(f"💾 Iniciando exportación: {model_name}")
            self.export_worker = ExportWorker(model_name, file_path, model_type)
            self.export_worker.progress_updated.connect(self.update_export_progress)
            self.export_worker.export_completed.connect(self.on_export_completed)
            self.export_worker.export_failed.connect(self.on_export_failed)
            self.export_worker.start()

    def backup_all_models(self):
        try:
            models = self.ml_classifier.list_available_models()
            if not models:
                QMessageBox.information(self, "Sin Modelos", "No hay modelos para hacer backup.")
                return

            backup_dir = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio para Backup", "")
            if backup_dir:
                self.activity_log.append(f"📦 Iniciando backup de {len(models)} modelos...")
                success_count = 0
                for model in models:
                    try:
                        worker = ExportWorker(model['name'], os.path.join(backup_dir, f"{model['name']}.zip"), 'dl' if model.get('is_deep_learning', False) else 'ml')
                        worker.run() 
                        success_count += 1
                    except Exception as e:
                        self.activity_log.append(f"❌ Error en {model['name']}: {str(e)}")
                self.activity_log.append(f"✅ Backup completado: {success_count}/{len(models)} modelos")
                QMessageBox.information(self, "Backup Completado", f"Se exportaron {success_count} de {len(models)} modelos al directorio:\n{backup_dir}")
        except Exception as e:
            self.activity_log.append(f"❌ Error en backup: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error durante el backup: {str(e)}")

    def update_import_progress(self, message): self.activity_log.append(f"📥 {message}")
    def update_export_progress(self, message): self.activity_log.append(f"📤 {message}")
    def on_import_completed(self, message):
        self.activity_log.append(f"✅ {message}")
        self.refresh_model_lists()
        QMessageBox.information(self, "Importación Exitosa", message)
    def on_import_failed(self, error_message):
        self.activity_log.append(f"❌ {error_message}")
        QMessageBox.critical(self, "Error de Importación", error_message)
    def on_export_completed(self, message):
        self.activity_log.append(f"✅ {message}")
        QMessageBox.information(self, "Exportación Exitosa", message)
    def on_export_failed(self, error_message):
        self.activity_log.append(f"❌ {error_message}")
        QMessageBox.critical(self, "Error de Exportación", error_message)