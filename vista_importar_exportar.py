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

    def __init__(self, model_name, export_path, model_type, protection_info=None):
        super().__init__()
        self.model_name = model_name
        self.export_path = export_path
        self.model_type = model_type  # 'ml' o 'dl'
        self.protection_info = protection_info or {
            'encrypted': True,
            'format': '.zip',
            'protection_level': 'high'
        }

    def encrypt_data(self, data):
        """Encripta los datos usando un método simple (para demostración)"""
        from itertools import cycle
        key = b'ClasificaTalentoPRO'  # Clave de ejemplo
        return bytes(a ^ b for a, b in zip(data, cycle(key)))

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

            self.progress_updated.emit("Comprimiendo y protegiendo archivos del modelo...")

            # Crear archivo comprimido con el formato especificado
            with zipfile.ZipFile(self.export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Agregar metadatos especiales con información de protección
                package_info = {
                    'format_version': '2.0',
                    'model_type': self.model_type,
                    'model_name': self.model_name,
                    'exported_by': 'ClasificaTalento PRO',
                    'export_date': datetime.datetime.now().isoformat(),
                    'protection': {
                        'enabled': self.protection_info['encrypted'],
                        'level': self.protection_info['protection_level'],
                        'format': self.protection_info['format']
                    }
                }

                zipf.writestr('package_info.json', json.dumps(package_info, indent=2))

                # Agregar todos los archivos del modelo
                for root, dirs, files in os.walk(source_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, source_dir)
                        
                        # Leer y posiblemente encriptar el archivo
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                            
                        if self.protection_info['encrypted']:
                            self.progress_updated.emit(f"Encriptando: {file}")
                            file_data = self.encrypt_data(file_data)
                        
                        zipf.writestr(arcname, file_data)
                        self.progress_updated.emit(f"Agregando: {file}")

            self.export_completed.emit(f"Modelo exportado y protegido exitosamente en: {self.export_path}")

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
        # Lista de extensiones válidas
        self.valid_extensions = ['.zip', '.senati', '.mlmodel', '.aimodel', '.ctpro']

    def decrypt_data(self, data):
        """Desencripta los datos (usa el mismo método que la encriptación por ser XOR)"""
        from itertools import cycle
        key = b'ClasificaTalentoPRO'  # Debe ser la misma clave que en ExportWorker
        return bytes(a ^ b for a, b in zip(data, cycle(key)))

    def is_valid_model_file(self):
        """Verifica si el archivo tiene una extensión válida"""
        return any(self.file_path.lower().endswith(ext.lower()) for ext in self.valid_extensions)

    def run(self):
        """Ejecuta la importación del modelo"""
        try:
            self.progress_updated.emit("Verificando archivo de modelo...")

            # Verificar extensión válida
            if not self.is_valid_model_file():
                self.import_failed.emit(f"Formato de archivo no válido. Formatos soportados: {', '.join(self.valid_extensions)}")
                return

            # Verificar si es un archivo ZIP válido
            if not zipfile.is_zipfile(self.file_path):
                self.import_failed.emit("El archivo no es un formato de modelo válido")
                return

            with zipfile.ZipFile(self.file_path, 'r') as zipf:
                # Verificar metadatos
                if 'package_info.json' not in zipf.namelist():
                    self.import_failed.emit("El archivo no contiene metadatos de modelo válidos")
                    return

                # Leer y verificar la información del paquete
                try:
                    package_info = json.loads(zipf.read('package_info.json').decode('utf-8'))
                    model_type = package_info.get('model_type', 'unknown')
                    original_name = package_info.get('model_name', 'imported_model')
                    format_version = package_info.get('format_version', '1.0')
                    
                    # Obtener información de protección si existe
                    protection_info = package_info.get('protection', {
                        'enabled': False,
                        'level': 'none',
                        'format': '.zip'
                    })

                    self.progress_updated.emit(f"Importando modelo {model_type.upper()}: {original_name} (Versión: {format_version})")
                    
                    if protection_info['enabled']:
                        self.progress_updated.emit("Modelo protegido detectado - Iniciando proceso de desencriptación")

                    final_model_name = self.target_model_name if self.target_model_name else original_name

                    if model_type == 'ml':
                        target_dir = os.path.join('saved_models', final_model_name)
                    elif model_type == 'dl':
                        target_dir = os.path.join('saved_deep_models', final_model_name)
                    else:
                        self.import_failed.emit(f"Tipo de modelo desconocido: {model_type}")
                        return

                    # Crear directorio destino
                    os.makedirs(target_dir, exist_ok=True)
                    self.progress_updated.emit("Extrayendo archivos del modelo...")

                    # Procesar cada archivo
                    for file_info in zipf.filelist:
                        if file_info.filename != 'package_info.json':
                            # Leer el contenido del archivo
                            file_data = zipf.read(file_info.filename)
                            
                            # Desencriptar si es necesario
                            if protection_info['enabled']:
                                self.progress_updated.emit(f"Desencriptando: {file_info.filename}")
                                try:
                                    file_data = self.decrypt_data(file_data)
                                except Exception as e:
                                    self.import_failed.emit(f"Error al desencriptar {file_info.filename}: {str(e)}")
                                    return

                            # Escribir el archivo
                            target_path = os.path.join(target_dir, file_info.filename)
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            with open(target_path, 'wb') as f:
                                f.write(file_data)
                            
                            self.progress_updated.emit(f"Extraído: {file_info.filename}")

                    self.import_completed.emit(f"Modelo importado exitosamente como: {final_model_name}")

                except json.JSONDecodeError:
                    self.import_failed.emit("Error al leer los metadatos del modelo")
                    return
                except Exception as e:
                    self.import_failed.emit(f"Error al procesar el modelo: {str(e)}")
                    return

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
        
        # Sección de formato personalizado
        format_label = QLabel("Formato de exportación:")
        self.format_combo = QComboBox()
        self.format_combo.addItems([".zip", "Personalizado"])
        self.format_combo.currentTextChanged.connect(self.on_format_changed)
        export_options_layout.addWidget(format_label, 0, 0)
        export_options_layout.addWidget(self.format_combo, 0, 1)

        # Campo para formato personalizado
        self.custom_format_label = QLabel("Extensión personalizada:")
        self.custom_format_input = QComboBox()
        self.custom_format_input.setEditable(True)
        self.custom_format_input.addItems([".senati", ".mlmodel", ".aimodel", ".ctpro"])
        self.custom_format_input.setEnabled(False)
        export_options_layout.addWidget(self.custom_format_label, 1, 0)
        export_options_layout.addWidget(self.custom_format_input, 1, 1)

        # Sección de protección
        protection_label = QLabel("🔒 Protección:")
        protection_label.setStyleSheet("font-weight: bold;")
        export_options_layout.addWidget(protection_label, 2, 0, 1, 2)

        self.encrypt_checkbox = QCheckBox("Encriptar contenido del modelo")
        self.encrypt_checkbox.setChecked(True)
        export_options_layout.addWidget(self.encrypt_checkbox, 3, 0, 1, 2)

        self.include_metadata_checkbox = QCheckBox("Incluir metadatos completos (recomendado)")
        self.include_metadata_checkbox.setChecked(True)
        export_options_layout.addWidget(self.include_metadata_checkbox, 4, 0, 1, 2)

        self.compress_checkbox = QCheckBox("Compresión optimizada (automática)")
        self.compress_checkbox.setChecked(True)
        export_options_layout.addWidget(self.compress_checkbox, 5, 0, 1, 2)

        # Información de seguridad
        security_info = QLabel("ℹ️ La protección ayuda a prevenir el uso no autorizado del modelo")
        security_info.setStyleSheet("color: #666; font-style: italic;")
        export_options_layout.addWidget(security_info, 6, 0, 1, 2)

        layout.addLayout(export_options_layout)

        export_buttons_layout = QHBoxLayout()
        export_file_btn = QPushButton("💾 Exportar Modelo")
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
        """Permite seleccionar un archivo de modelo para importar"""
        formats = "Archivos de Modelo (*.zip *.senati *.mlmodel *.aimodel *.ctpro);;Todos los archivos (*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Seleccionar Archivo de Modelo para Importar", 
            "", 
            formats
        )
        
        if file_path:
            self.activity_log.append(f"📁 Archivo seleccionado: {os.path.basename(file_path)}")
            custom_name = self.import_model_name_combo.currentText().strip()
            target_name = None if custom_name == "Usar nombre original" or not custom_name else custom_name
            
            # Verificar si es un formato personalizado
            is_custom = not file_path.lower().endswith('.zip')
            if is_custom:
                self.activity_log.append("⚠️ Detectado formato personalizado - Verificando protección...")
            
            self.import_worker = ImportWorker(file_path, target_name)
            self.import_worker.progress_updated.connect(self.update_import_progress)
            self.import_worker.import_completed.connect(self.on_import_completed)
            self.import_worker.import_failed.connect(self.on_import_failed)
            self.import_worker.start()

    def dragEnterEvent(self, event):
        """Maneja el inicio del drag de archivos"""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                valid_extensions = ['.zip', '.senati', '.mlmodel', '.aimodel', '.ctpro']
                if any(file_path.lower().endswith(ext) for ext in valid_extensions):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        """Maneja el drop de archivos"""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            valid_extensions = ['.zip', '.senati', '.mlmodel', '.aimodel', '.ctpro']
            if any(file_path.lower().endswith(ext) for ext in valid_extensions):
                self.activity_log.append(f"📁 Archivo arrastrado: {os.path.basename(file_path)}")
                self.import_model(file_path)
                event.acceptProposedAction()

    def on_format_changed(self, text):
        """Maneja el cambio en el formato de exportación"""
        self.custom_format_input.setEnabled(text == "Personalizado")

    def export_to_file(self):
        if self.export_model_combo.count() == 0 or self.export_model_combo.currentText() == "No hay modelos disponibles":
            QMessageBox.warning(self, "Sin Modelos", "No hay modelos disponibles para exportar.")
            return

        model_data = self.export_model_combo.currentData()
        if not model_data:
            QMessageBox.warning(self, "Error", "No se pudo obtener información del modelo seleccionado.")
            return

        model_name = model_data['name']
        
        # Determinar la extensión del archivo
        if self.format_combo.currentText() == "Personalizado":
            extension = self.custom_format_input.currentText()
            if not extension.startswith('.'):
                extension = '.' + extension
        else:
            extension = ".zip"

        suggested_filename = f"{model_name}{extension}"
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Exportar Modelo", 
            suggested_filename, 
            f"Archivos {extension} (*{extension});;Todos los archivos (*)"
        )

        if file_path:
            if not file_path.lower().endswith(extension.lower()):
                file_path += extension
            
            model_type = 'dl' if model_data.get('is_deep_learning', False) else 'ml'
            
            # Agregar información de protección
            protection_info = {
                'encrypted': self.encrypt_checkbox.isChecked(),
                'format': extension,
                'protection_level': 'high' if self.encrypt_checkbox.isChecked() else 'none'
            }
            
            self.activity_log.append(f"💾 Iniciando exportación protegida: {model_name}")
            self.export_worker = ExportWorker(model_name, file_path, model_type, protection_info)
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