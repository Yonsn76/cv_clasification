# vista_importar_exportar.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QScrollArea, QProgressBar,
                             QComboBox, QCheckBox, QGroupBox, QGridLayout,
                             QTextEdit, QFileDialog)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

class VistaImportarExportar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VistaImportarExportar")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(25, 20, 25, 20)
        main_layout.setSpacing(20)

        # Título principal
        title_label = QLabel("📦 Importar / Exportar Modelos 📦")
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

        drop_text = QLabel("Arrastra archivos de modelo aquí\n(.pkl, .joblib, .h5, .onnx)")
        drop_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_text.setStyleSheet("color: #3498DB; font-size: 14px; font-weight: bold;")

        drop_layout.addWidget(drop_icon)
        drop_layout.addWidget(drop_text)
        layout.addWidget(drop_area)

        # Botones de importar
        import_buttons_layout = QHBoxLayout()

        select_file_btn = QPushButton("📂 Seleccionar Archivo")
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

        import_url_btn = QPushButton("🌐 Importar desde URL")
        import_url_btn.setFixedHeight(40)
        import_url_btn.setStyleSheet("""
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

        import_buttons_layout.addWidget(select_file_btn)
        import_buttons_layout.addWidget(import_url_btn)
        import_buttons_layout.addStretch()
        layout.addLayout(import_buttons_layout)

        # Configuración de importación
        config_layout = QGridLayout()

        config_layout.addWidget(QLabel("Tipo de modelo:"), 0, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Auto-detectar", "Scikit-learn", "TensorFlow/Keras", "PyTorch", "ONNX"])
        self.model_type_combo.setStyleSheet("""
            QComboBox {
                background-color: #34495E;
                color: white;
                border: 1px solid #3498DB;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        config_layout.addWidget(self.model_type_combo, 0, 1)

        self.validate_checkbox = QCheckBox("Validar modelo al importar")
        self.validate_checkbox.setChecked(True)
        self.validate_checkbox.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        config_layout.addWidget(self.validate_checkbox, 1, 0, 1, 2)

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
        self.export_model_combo.addItems([
            "Random Forest CV (94.2%)",
            "Neural Network Pro (96.8%)",
            "SVM Classifier (91.5%)",
            "BERT Analyzer (97.3%)",
            "Quick Classifier (89.7%)"
        ])
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

        # Formato de exportación
        export_options_layout.addWidget(QLabel("Formato:"), 0, 0)
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["Pickle (.pkl)", "Joblib (.joblib)", "ONNX (.onnx)", "TensorFlow SavedModel"])
        self.export_format_combo.setStyleSheet(self.export_model_combo.styleSheet())
        export_options_layout.addWidget(self.export_format_combo, 0, 1)

        # Opciones adicionales
        self.include_metadata_checkbox = QCheckBox("Incluir metadatos del modelo")
        self.include_metadata_checkbox.setChecked(True)
        self.include_metadata_checkbox.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        export_options_layout.addWidget(self.include_metadata_checkbox, 1, 0, 1, 2)

        self.compress_checkbox = QCheckBox("Comprimir archivo de salida")
        self.compress_checkbox.setChecked(False)
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

        export_file_btn = QPushButton("💾 Exportar a Archivo")
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

        export_cloud_btn = QPushButton("☁️ Subir a la Nube")
        export_cloud_btn.setFixedHeight(40)
        export_cloud_btn.setStyleSheet("""
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

        export_buttons_layout.addWidget(export_file_btn)
        export_buttons_layout.addWidget(export_cloud_btn)
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

    def select_import_file(self):
        """Abre el diálogo para seleccionar archivo de modelo"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Seleccionar Modelo para Importar",
            "",
            "Archivos de Modelo (*.pkl *.joblib *.h5 *.onnx);;Todos los archivos (*)"
        )
        
        if file_path:
            self.activity_log.append(f"📁 Archivo seleccionado: {file_path}")
            self.activity_log.append("🔄 Iniciando proceso de importación...")
            # Aquí iría la lógica real de importación

    def export_to_file(self):
        """Abre el diálogo para exportar modelo"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Exportar Modelo",
            "",
            "Archivos Pickle (*.pkl);;Archivos Joblib (*.joblib);;Archivos ONNX (*.onnx)"
        )
        
        if file_path:
            model_name = self.export_model_combo.currentText()
            self.activity_log.append(f"💾 Exportando modelo: {model_name}")
            self.activity_log.append(f"📁 Destino: {file_path}")
            self.activity_log.append("✅ Exportación completada exitosamente")
            # Aquí iría la lógica real de exportación
