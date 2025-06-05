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

            # Crear archivo .senati (que es un ZIP)
            with zipfile.ZipFile(self.export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Agregar metadatos especiales para .senati
                senati_info = {
                    'format_version': '1.0',
                    'model_type': self.model_type,
                    'model_name': self.model_name,
                    'exported_by': 'ClasificaTalento PRO',
                    'export_date': datetime.datetime.now().isoformat()
                }

                zipf.writestr('senati_info.json', json.dumps(senati_info, indent=2))

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

    def __init__(self, senati_file_path, target_model_name=None):
        super().__init__()
        self.senati_file_path = senati_file_path
        self.target_model_name = target_model_name

    def run(self):
        """Ejecuta la importación del modelo"""
        try:
            self.progress_updated.emit("Verificando archivo .senati...")

            # Verificar que es un archivo ZIP válido
            if not zipfile.is_zipfile(self.senati_file_path):
                self.import_failed.emit("El archivo no es un formato .senati válido")
                return

            with zipfile.ZipFile(self.senati_file_path, 'r') as zipf:
                # Verificar que contiene metadatos .senati
                if 'senati_info.json' not in zipf.namelist():
                    self.import_failed.emit("El archivo no contiene metadatos .senati válidos")
                    return

                # Leer metadatos
                senati_info_data = zipf.read('senati_info.json')
                senati_info = json.loads(senati_info_data.decode('utf-8'))

                model_type = senati_info.get('model_type', 'unknown')
                original_name = senati_info.get('model_name', 'imported_model')

                self.progress_updated.emit(f"Importando modelo {model_type.upper()}: {original_name}")

                # Determinar nombre final del modelo
                final_model_name = self.target_model_name if self.target_model_name else original_name

                # Determinar directorio destino
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

                # Extraer archivos (excepto senati_info.json)
                for file_info in zipf.filelist:
                    if file_info.filename != 'senati_info.json':
                        zipf.extract(file_info, target_dir)
                        self.progress_updated.emit(f"Extraído: {file_info.filename}")

                self.import_completed.emit(f"Modelo importado exitosamente como: {final_model_name}")

        except Exception as e:
            self.import_failed.emit(f"Error durante la importación: {str(e)}")


class VistaImportarExportar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VistaImportarExportar")

        # Inicializar clasificadores para obtener lista de modelos
        self.ml_classifier = CVClassifier()
        self.dl_classifier = DeepLearningClassifier()

        # Workers para operaciones en segundo plano
        self.export_worker = None
        self.import_worker = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(25, 20, 25, 20)
        main_layout.setSpacing(20)

        # Título principal
        title_label = QLabel("📦 Importar / Exportar Modelos SENATI 📦")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #E0E0E0; margin-bottom: 10px;")
        main_layout.addWidget(title_label)

        # Crear scroll area para el contenido
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(25)

        # Sección de importar modelos
        import_section = self.create_import_section()
        content_layout.addWidget(import_section)

        # Sección de exportar modelos
        export_section = self.create_export_section()
        content_layout.addWidget(export_section)

        # Sección de gestión de modelos
        management_section = self.create_management_section()
        content_layout.addWidget(management_section)

        content_layout.addStretch(1)
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

        # Cargar lista inicial de modelos
        self.refresh_model_lists()

    def create_import_section(self):
        """Crea la sección de importar modelos"""
        section = QGroupBox("📥 Importar Modelos")
        section.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #E0E0E0;
                border: 2px solid #3498DB;
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
            }
        """)

        layout = QVBoxLayout(section)
        layout.setSpacing(20)

        # Área de arrastrar y soltar
        drop_area = QFrame()
        drop_area.setFixedHeight(150)
        drop_area.setStyleSheet("""
            QFrame {
                border: 3px dashed #3498DB;
                border-radius: 12px;
                background-color: rgba(52, 152, 219, 0.1);
            }
        """)

        drop_layout = QVBoxLayout(drop_area)
        drop_icon = QLabel("📁")
        drop_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_icon.setStyleSheet("font-size: 48px; color: #3498DB; margin: 10px;")

        drop_text = QLabel("Arrastra archivos de modelo aquí\n(.senati - Formato ClasificaTalento PRO)")
        drop_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_text.setStyleSheet("color: #3498DB; font-size: 14px; font-weight: bold;")

        drop_layout.addWidget(drop_icon)
        drop_layout.addWidget(drop_text)
        layout.addWidget(drop_area)

        # Botones de importar
        import_buttons_layout = QHBoxLayout()

        select_file_btn = QPushButton("📂 Seleccionar Archivo .senati")
        select_file_btn.setFixedHeight(40)
        select_file_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)
        select_file_btn.clicked.connect(self.select_import_file)

        refresh_models_btn = QPushButton("🔄 Actualizar Lista")
        refresh_models_btn.setFixedHeight(40)
        refresh_models_btn.setStyleSheet("""
            QPushButton {
                background-color: #9B59B6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8E44AD;
            }
        """)
        refresh_models_btn.clicked.connect(self.refresh_model_lists)

        import_buttons_layout.addWidget(select_file_btn)
        import_buttons_layout.addWidget(refresh_models_btn)
        import_buttons_layout.addStretch()
        layout.addLayout(import_buttons_layout)

        # Configuración de importación
        config_layout = QGridLayout()

        config_layout.addWidget(QLabel("Nombre del modelo:"), 0, 0)
        self.import_model_name_combo = QComboBox()
        self.import_model_name_combo.setEditable(True)
        self.import_model_name_combo.setPlaceholderText("Usar nombre original o escribir nuevo...")
        self.import_model_name_combo.addItem("Usar nombre original")
        self.import_model_name_combo.setStyleSheet("""
            QComboBox {
                background-color: #34495E;
                color: white;
                border: 1px solid #3498DB;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        config_layout.addWidget(self.import_model_name_combo, 0, 1)

        self.validate_checkbox = QCheckBox("Validar integridad del modelo")
        self.validate_checkbox.setChecked(True)
        self.validate_checkbox.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        config_layout.addWidget(self.validate_checkbox, 1, 0, 1, 2)

        self.overwrite_checkbox = QCheckBox("Sobrescribir si el modelo ya existe")
        self.overwrite_checkbox.setChecked(False)
        self.overwrite_checkbox.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        config_layout.addWidget(self.overwrite_checkbox, 2, 0, 1, 2)

        # Aplicar estilos a las etiquetas
        for i in range(config_layout.rowCount()):
            item = config_layout.itemAtPosition(i, 0)
            if item and item.widget() and isinstance(item.widget(), QLabel):
                item.widget().setStyleSheet("color: #BDC3C7; font-size: 13px;")

        layout.addLayout(config_layout)

        return section

    def create_export_section(self):
        """Crea la sección de exportar modelos"""
        section = QGroupBox("📤 Exportar Modelos")
        section.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #E0E0E0;
                border: 2px solid #E74C3C;
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
            }
        """)

        layout = QVBoxLayout(section)
        layout.setSpacing(20)

        # Selección de modelo a exportar
        model_selection_layout = QHBoxLayout()
        
        model_label = QLabel("Modelo a exportar:")
        model_label.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        
        self.export_model_combo = QComboBox()
        # Se llenará dinámicamente con refresh_model_lists()
        self.export_model_combo.setStyleSheet("""
            QComboBox {
                background-color: #34495E;
                color: white;
                border: 1px solid #E74C3C;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        
        model_selection_layout.addWidget(model_label)
        model_selection_layout.addWidget(self.export_model_combo)
        model_selection_layout.addStretch()
        layout.addLayout(model_selection_layout)

        # Opciones de exportación
        export_options_layout = QGridLayout()

        # Información del formato
        format_info_label = QLabel("Formato: .senati (Formato nativo ClasificaTalento PRO)")
        format_info_label.setStyleSheet("color: #27AE60; font-size: 13px; font-weight: bold;")
        export_options_layout.addWidget(format_info_label, 0, 0, 1, 2)

        # Opciones adicionales
        self.include_metadata_checkbox = QCheckBox("Incluir metadatos completos (recomendado)")
        self.include_metadata_checkbox.setChecked(True)
        self.include_metadata_checkbox.setEnabled(False)  # Siempre incluido en .senati
        self.include_metadata_checkbox.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        export_options_layout.addWidget(self.include_metadata_checkbox, 1, 0, 1, 2)

        self.compress_checkbox = QCheckBox("Compresión optimizada (automática)")
        self.compress_checkbox.setChecked(True)
        self.compress_checkbox.setEnabled(False)  # Siempre comprimido en .senati
        self.compress_checkbox.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        export_options_layout.addWidget(self.compress_checkbox, 2, 0, 1, 2)

        # Aplicar estilos a las etiquetas
        for i in range(export_options_layout.rowCount()):
            item = export_options_layout.itemAtPosition(i, 0)
            if item and item.widget() and isinstance(item.widget(), QLabel):
                item.widget().setStyleSheet("color: #BDC3C7; font-size: 13px;")

        layout.addLayout(export_options_layout)

        # Botones de exportar
        export_buttons_layout = QHBoxLayout()

        export_file_btn = QPushButton("💾 Exportar como .senati")
        export_file_btn.setFixedHeight(40)
        export_file_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        export_file_btn.clicked.connect(self.export_to_file)

        backup_all_btn = QPushButton("📦 Backup Todos los Modelos")
        backup_all_btn.setFixedHeight(40)
        backup_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #F39C12;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E67E22;
            }
        """)
        backup_all_btn.clicked.connect(self.backup_all_models)

        export_buttons_layout.addWidget(export_file_btn)
        export_buttons_layout.addWidget(backup_all_btn)
        export_buttons_layout.addStretch()
        layout.addLayout(export_buttons_layout)

        return section

    def create_management_section(self):
        """Crea la sección de gestión de modelos"""
        section = QGroupBox("⚙️ Gestión de Modelos")
        section.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #E0E0E0;
                border: 2px solid #F39C12;
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
            }
        """)

        layout = QVBoxLayout(section)
        layout.setSpacing(20)

        # Log de actividades
        log_label = QLabel("📋 Registro de Actividades:")
        log_label.setStyleSheet("color: #F39C12; font-size: 14px; font-weight: bold;")
        layout.addWidget(log_label)

        self.activity_log = QTextEdit()
        self.activity_log.setMaximumHeight(120)
        self.activity_log.setReadOnly(True)
        self.activity_log.setPlaceholderText("Las actividades de importación/exportación aparecerán aquí...")
        self.activity_log.setStyleSheet("""
            QTextEdit {
                background-color: #2C3E50;
                color: #BDC3C7;
                border: 1px solid #F39C12;
                border-radius: 5px;
                padding: 8px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.activity_log)

        # Botones de gestión
        management_buttons_layout = QHBoxLayout()

        backup_btn = QPushButton("💾 Backup Completo")
        backup_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)

        restore_btn = QPushButton("🔄 Restaurar")
        restore_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)

        clean_btn = QPushButton("🧹 Limpiar Cache")
        clean_btn.setStyleSheet("""
            QPushButton {
                background-color: #7F8C8D;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #95A5A6;
            }
        """)

        management_buttons_layout.addWidget(backup_btn)
        management_buttons_layout.addWidget(restore_btn)
        management_buttons_layout.addWidget(clean_btn)
        management_buttons_layout.addStretch()
        layout.addLayout(management_buttons_layout)

        return section

    def refresh_model_lists(self):
        """Actualiza las listas de modelos disponibles"""
        try:
            # Obtener modelos ML y DL
            ml_models = self.ml_classifier.list_available_models()

            # Limpiar combo de exportación
            self.export_model_combo.clear()

            if not ml_models:
                self.export_model_combo.addItem("No hay modelos disponibles")
                self.activity_log.append("⚠️ No se encontraron modelos para exportar")
                return

            # Agregar modelos a la lista de exportación
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
        """Abre el diálogo para seleccionar archivo .senati"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Seleccionar Archivo .senati para Importar",
            "",
            "Archivos SENATI (*.senati);;Todos los archivos (*)"
        )

        if file_path:
            self.activity_log.append(f"📁 Archivo .senati seleccionado: {os.path.basename(file_path)}")

            # Obtener nombre personalizado si se especificó
            custom_name = self.import_model_name_combo.currentText().strip()
            target_name = None if custom_name == "Usar nombre original" or not custom_name else custom_name

            # Iniciar importación en segundo plano
            self.import_worker = ImportWorker(file_path, target_name)
            self.import_worker.progress_updated.connect(self.update_import_progress)
            self.import_worker.import_completed.connect(self.on_import_completed)
            self.import_worker.import_failed.connect(self.on_import_failed)
            self.import_worker.start()

    def export_to_file(self):
        """Abre el diálogo para exportar modelo como .senati"""
        if self.export_model_combo.count() == 0 or self.export_model_combo.currentText() == "No hay modelos disponibles":
            QMessageBox.warning(self, "Sin Modelos", "No hay modelos disponibles para exportar.")
            return

        model_data = self.export_model_combo.currentData()
        if not model_data:
            QMessageBox.warning(self, "Error", "No se pudo obtener información del modelo seleccionado.")
            return

        model_name = model_data['name']
        suggested_filename = f"{model_name}.senati"

        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Exportar Modelo como .senati",
            suggested_filename,
            "Archivos SENATI (*.senati);;Todos los archivos (*)"
        )

        if file_path:
            # Asegurar extensión .senati
            if not file_path.lower().endswith('.senati'):
                file_path += '.senati'

            model_type = 'dl' if model_data.get('is_deep_learning', False) else 'ml'

            self.activity_log.append(f"💾 Iniciando exportación: {model_name}")

            # Iniciar exportación en segundo plano
            self.export_worker = ExportWorker(model_name, file_path, model_type)
            self.export_worker.progress_updated.connect(self.update_export_progress)
            self.export_worker.export_completed.connect(self.on_export_completed)
            self.export_worker.export_failed.connect(self.on_export_failed)
            self.export_worker.start()

    def backup_all_models(self):
        """Crea un backup de todos los modelos disponibles"""
        try:
            models = self.ml_classifier.list_available_models()
            if not models:
                QMessageBox.information(self, "Sin Modelos", "No hay modelos disponibles para hacer backup.")
                return

            # Seleccionar directorio de destino
            backup_dir = QFileDialog.getExistingDirectory(
                self, "Seleccionar Directorio para Backup", ""
            )

            if backup_dir:
                self.activity_log.append(f"� Iniciando backup de {len(models)} modelos...")

                success_count = 0
                for model in models:
                    try:
                        model_name = model['name']
                        model_type = 'dl' if model.get('is_deep_learning', False) else 'ml'
                        backup_file = os.path.join(backup_dir, f"{model_name}.senati")

                        # Crear worker para cada modelo
                        worker = ExportWorker(model_name, backup_file, model_type)
                        worker.run()  # Ejecutar sincrónicamente para el backup
                        success_count += 1

                    except Exception as e:
                        self.activity_log.append(f"❌ Error en {model_name}: {str(e)}")

                self.activity_log.append(f"✅ Backup completado: {success_count}/{len(models)} modelos")
                QMessageBox.information(
                    self, "Backup Completado",
                    f"Se exportaron {success_count} de {len(models)} modelos al directorio:\n{backup_dir}"
                )

        except Exception as e:
            self.activity_log.append(f"❌ Error en backup: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error durante el backup: {str(e)}")

    # Métodos para manejar señales de los workers
    def update_import_progress(self, message):
        """Actualiza el progreso de importación"""
        self.activity_log.append(f"📥 {message}")

    def update_export_progress(self, message):
        """Actualiza el progreso de exportación"""
        self.activity_log.append(f"📤 {message}")

    def on_import_completed(self, message):
        """Maneja la finalización exitosa de importación"""
        self.activity_log.append(f"✅ {message}")
        self.refresh_model_lists()  # Actualizar lista después de importar
        QMessageBox.information(self, "Importación Exitosa", message)

    def on_import_failed(self, error_message):
        """Maneja errores de importación"""
        self.activity_log.append(f"❌ {error_message}")
        QMessageBox.critical(self, "Error de Importación", error_message)

    def on_export_completed(self, message):
        """Maneja la finalización exitosa de exportación"""
        self.activity_log.append(f"✅ {message}")
        QMessageBox.information(self, "Exportación Exitosa", message)

    def on_export_failed(self, error_message):
        """Maneja errores de exportación"""
        self.activity_log.append(f"❌ {error_message}")
        QMessageBox.critical(self, "Error de Exportación", error_message)
