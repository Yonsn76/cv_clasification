# vista_centro_accion.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox, QComboBox, QTextEdit,
                             QFileDialog, QMessageBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QSplitter, QGridLayout,
                             QFrame, QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QFont
import os
import PyPDF2
from models.cv_classifier import CVClassifier
from models.deep_learning_classifier import DeepLearningClassifier


class PulsingButton(QPushButton):
    """Botón rectangular con efecto de pulsación para clasificación"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setObjectName("PulsingButton")
        self.setFixedSize(200, 60)
        self.is_pulsing = False
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self.toggle_pulse_style)
        self.setProperty("pulsing", "false")

    def start_pulsing(self):
        """Inicia el efecto de pulsación"""
        self.is_pulsing = True
        self.pulse_timer.start(700)

    def stop_pulsing(self):
        """Detiene el efecto de pulsación"""
        self.is_pulsing = False
        self.pulse_timer.stop()
        self.setProperty("pulsing", "false")
        self._update_style()

    def toggle_pulse_style(self):
        """Alterna entre estilos para crear efecto de pulsación"""
        if self.is_pulsing:
            current_state = self.property("pulsing")
            self.setProperty("pulsing", "false" if current_state == "true" else "true")
            self._update_style()
    
    def _update_style(self):
        """Refresca el estilo del widget para aplicar cambios de QSS"""
        self.style().unpolish(self)
        self.style().polish(self)


class ModelStatusCard(QFrame):
    """Tarjeta visual para mostrar el estado del modelo"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ModelStatusCard")
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setFixedHeight(120)
        self.setup_ui()
        self.set_no_model()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(15,15,15,15)

        self.title_label = QLabel("🤖 Estado del Modelo")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        self.status_label = QLabel("❌ Sin modelo cargado")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_font = QFont()
        status_font.setPointSize(11)
        self.status_label.setFont(status_font)
        layout.addWidget(self.status_label)

        self.info_label = QLabel("Selecciona y carga un modelo para comenzar")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setWordWrap(True)
        info_font = QFont()
        info_font.setPointSize(10)
        self.info_label.setFont(info_font)
        layout.addWidget(self.info_label)

    def _update_style(self):
        self.style().unpolish(self)
        self.style().polish(self)

    def set_model_loaded(self, model_name, model_type, professions):
        """Actualiza la tarjeta cuando se carga un modelo"""
        self.status_label.setText(f"✅ {model_name}")
        self.info_label.setText(f"{model_type} • {len(professions)} profesiones")
        self.setProperty("status", "loaded")
        self._update_style()


    def set_no_model(self):
        """Actualiza la tarjeta cuando no hay modelo"""
        self.status_label.setText("❌ Sin modelo cargado")
        self.info_label.setText("Selecciona y carga un modelo para comenzar")
        self.setProperty("status", "no_model")
        self._update_style()


class ClassificationWorker(QThread):
    """Worker thread para clasificación de CVs en segundo plano"""
    progress_updated = pyqtSignal(str)
    classification_completed = pyqtSignal(dict)
    classification_failed = pyqtSignal(str)

    def __init__(self, cv_file_path, classifier, is_deep_learning=False):
        super().__init__()
        self.cv_file_path = cv_file_path
        self.classifier = classifier
        self.is_deep_learning = is_deep_learning

    def extract_text_from_pdf(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            return ""

    def run(self):
        try:
            self.progress_updated.emit("Extrayendo texto del CV...")

            if self.cv_file_path.lower().endswith('.pdf'):
                cv_text = self.extract_text_from_pdf(self.cv_file_path)
            else:
                with open(self.cv_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    cv_text = f.read()

            if not cv_text or not cv_text.strip():
                self.classification_failed.emit("No se pudo extraer texto o el archivo está vacío.")
                return

            self.progress_updated.emit("Clasificando CV con modelo entrenado...")
            result = self.classifier.predict_cv(cv_text)

            if result.get('error', False):
                self.classification_failed.emit(result.get('message', 'Error desconocido en clasificación'))
            else:
                result['cv_file'] = os.path.basename(self.cv_file_path)
                result['model_type'] = 'Deep Learning' if self.is_deep_learning else 'Machine Learning'
                self.classification_completed.emit(result)

        except Exception as e:
            self.classification_failed.emit(f"Error durante la clasificación: {str(e)}")


class VistaCentroAccion(QWidget):
    """Vista creativa para clasificación de CVs con estilo de entrenamiento"""

    clasificacion_iniciada = pyqtSignal()
    clasificacion_completada = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VistaCentroAccion")

        self.current_loaded_model = None
        self.current_model_is_dl = False
        self.selected_cv_file = None
        self.ml_classifier = CVClassifier()
        self.dl_classifier = DeepLearningClassifier()
        self.classification_worker = None
        
        self.init_ui()
        self.refresh_model_selector()
        self.update_model_status_ui() 

    def init_ui(self):
        """Inicializa la interfaz de usuario con diseño creativo y responsive"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        self.create_header(layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        main_content = QWidget()
        content_layout = QVBoxLayout(main_content)
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.setSpacing(20)

        self.create_model_arsenal(content_layout)
        self.create_classification_center(content_layout)
        self.create_results_dashboard(content_layout)

        content_layout.addStretch()
        scroll_area.setWidget(main_content)
        layout.addWidget(scroll_area)

    def create_header(self, parent_layout):
        header_frame = QFrame()
        header_frame.setObjectName("HeaderFrame")
        header_layout = QHBoxLayout(header_frame)

        title_label = QLabel("🎯 Centro de Clasificación de CVs")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        header_layout.addWidget(title_label)
        parent_layout.addWidget(header_frame)


    def create_model_arsenal(self, parent_layout):
        group = QGroupBox("🏭 Arsenal de Modelos IA")
        group.setObjectName("ArsenalGroup")
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 10, 15, 15)


        self.model_status_card = ModelStatusCard()
        layout.addWidget(self.model_status_card)

        selector_frame = QFrame()
        selector_layout = QGridLayout(selector_frame)
        selector_layout.setSpacing(12)

        model_label = QLabel("🤖 Seleccionar Modelo:")
        model_label.setStyleSheet("font-weight: bold;")
        selector_layout.addWidget(model_label, 0, 0)

        self.model_selector_combo = QComboBox()
        self.model_selector_combo.setMinimumHeight(38)
        self.model_selector_combo.currentTextChanged.connect(self.on_model_selector_changed)
        selector_layout.addWidget(self.model_selector_combo, 0, 1, 1, 2)

        self.btn_load_selected_model = QPushButton("⚡ Cargar")
        self.btn_load_selected_model.clicked.connect(self.load_model_from_selector)
        self.btn_load_selected_model.setEnabled(False)
        selector_layout.addWidget(self.btn_load_selected_model, 1, 1)

        self.btn_refresh_selector = QPushButton("🔄 Actualizar")
        self.btn_refresh_selector.clicked.connect(self.refresh_model_selector)
        selector_layout.addWidget(self.btn_refresh_selector, 1, 2)

        layout.addWidget(selector_frame)
        parent_layout.addWidget(group)

    def create_classification_center(self, parent_layout):
        group = QGroupBox("🚀 Centro de Clasificación")
        group.setObjectName("ClassificationGroup")
        main_layout = QVBoxLayout(group)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 10, 15, 15)


        cv_panel = QFrame()
        cv_layout = QVBoxLayout(cv_panel)
        cv_layout.setSpacing(12)

        cv_title = QLabel("📄 Selección de Archivo CV")
        cv_title.setStyleSheet("font-weight: bold;")
        cv_layout.addWidget(cv_title)

        self.selected_file_label = QLabel("🔍 Ningún archivo seleccionado")
        self.selected_file_label.setObjectName("selected_file_label")
        self.selected_file_label.setProperty("fileState", "none")
        self.selected_file_label.setWordWrap(True)
        self.selected_file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cv_layout.addWidget(self.selected_file_label)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(15)

        self.btn_select_cv = QPushButton("📁 Seleccionar CV")
        self.btn_select_cv.clicked.connect(self.select_cv_file)
        buttons_layout.addWidget(self.btn_select_cv)
        buttons_layout.addStretch()
        
        cv_layout.addLayout(buttons_layout)
        main_layout.addWidget(cv_panel)
        
        classify_button_layout = QHBoxLayout()
        classify_button_layout.addStretch()
        self.btn_classify = PulsingButton("🎯 Clasificar CV")
        self.btn_classify.clicked.connect(self.classify_cv)
        self.btn_classify.setEnabled(False)
        classify_button_layout.addWidget(self.btn_classify)
        classify_button_layout.addStretch()
        main_layout.addLayout(classify_button_layout)

        parent_layout.addWidget(group)


    def create_results_dashboard(self, parent_layout):
        group = QGroupBox("📊 Dashboard de Resultados")
        group.setObjectName("ResultsGroup")
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 10, 15, 15)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)

        result_title = QLabel("🎯 Resultado de Clasificación")
        result_title.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(result_title)

        self.main_result = QTextEdit()
        self.main_result.setReadOnly(True)
        self.main_result.setMinimumHeight(180)
        self.main_result.setPlaceholderText("🔮 Los resultados de la clasificación aparecerán aquí...")
        left_layout.addWidget(self.main_result)
        splitter.addWidget(left_panel)

        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)

        ranking_title = QLabel("🏆 Ranking de Probabilidades")
        ranking_title.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(ranking_title)

        self.ranking_table = QTableWidget()
        self.ranking_table.setColumnCount(2)
        self.ranking_table.setHorizontalHeaderLabels(["Profesión", "Prob."])
        self.ranking_table.setMinimumHeight(180)
        self.ranking_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.ranking_table.setAlternatingRowColors(True)

        header = self.ranking_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.ranking_table.verticalHeader().setVisible(False)

        right_layout.addWidget(self.ranking_table)
        splitter.addWidget(right_panel)

        splitter.setSizes([380, 220])
        splitter.setMinimumHeight(220)
        layout.addWidget(splitter)
        parent_layout.addWidget(group)

    def refresh_model_selector(self):
        try:
            current_selection_data = self.model_selector_combo.currentData()
            self.model_selector_combo.clear()
            self.model_selector_combo.addItem("Selecciona un modelo...", None)

            models = self.ml_classifier.list_available_models()

            selected_index = 0
            for i, model in enumerate(models):
                model_name = model.get('display_name', model['name'])
                model_type_prefix = "🧠 DL" if model.get('is_deep_learning', False) else "🤖 ML"
                display_text = f"{model_type_prefix} - {model_name}"
                self.model_selector_combo.addItem(display_text, model)
                if current_selection_data and model['name'] == current_selection_data['name'] and \
                   model.get('is_deep_learning', False) == current_selection_data.get('is_deep_learning', False):
                    selected_index = i + 1 
            
            self.model_selector_combo.setCurrentIndex(selected_index)
            if not models:
                 self.model_selector_combo.addItem("No hay modelos disponibles", None)
                 self.model_selector_combo.setCurrentIndex(1)
            self.update_ui_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error actualizando lista de modelos: {str(e)}")
            self.model_selector_combo.clear()
            self.model_selector_combo.addItem("Error al cargar modelos", None)


    def on_model_selector_changed(self):
        self.btn_load_selected_model.setEnabled(
            self.model_selector_combo.currentData() is not None
        )

    def load_model_from_selector(self):
        model_data = self.model_selector_combo.currentData()
        if not model_data:
            QMessageBox.warning(self, "Sin Selección", "Por favor, selecciona un modelo de la lista.")
            return
        self.load_model_by_data(model_data)

    def load_model_by_data(self, model_data):
        model_name = model_data['name']
        is_deep_learning = model_data.get('is_deep_learning', False)
        display_name = model_data.get('display_name', model_name)

        try:
            classifier_to_use = self.dl_classifier if is_deep_learning else self.ml_classifier
            success = classifier_to_use.load_model(model_name)

            if success:
                self.current_loaded_model = model_name
                self.current_model_is_dl = is_deep_learning
                QMessageBox.information(
                    self, "Modelo Cargado",
                    f"El modelo '{display_name}' ha sido cargado exitosamente."
                )
                self.update_model_status_ui()
            else:
                self.current_loaded_model = None
                self.current_model_is_dl = False
                QMessageBox.critical(
                    self, "Error al Cargar",
                    f"No se pudo cargar el modelo '{display_name}'. Verifica su integridad."
                )
                self.update_model_status_ui()
            
            self.update_ui_state()

        except Exception as e:
            self.current_loaded_model = None
            self.current_model_is_dl = False
            QMessageBox.critical(
                self, "Error Inesperado",
                f"Ocurrió un error al cargar el modelo '{display_name}':\n{str(e)}"
            )
            self.update_model_status_ui()
            self.update_ui_state()


    def update_model_status_ui(self):
        if self.current_loaded_model:
            model_type = "Deep Learning" if self.current_model_is_dl else "Machine Learning"
            classifier = self.dl_classifier if self.current_model_is_dl else self.ml_classifier
            professions = []
            if hasattr(classifier, 'is_model_loaded') and classifier.is_model_loaded():
                 if hasattr(classifier, 'label_encoder') and classifier.label_encoder:
                    professions = list(classifier.label_encoder.classes_)
            
            self.model_status_card.set_model_loaded(
                self.current_loaded_model, model_type, professions
            )
        else:
            self.model_status_card.set_no_model()

    def update_ui_state(self):
        model_loaded = self.current_loaded_model is not None
        cv_selected = self.selected_cv_file is not None

        self.btn_classify.setEnabled(model_loaded and cv_selected)

        if not model_loaded:
            self.btn_classify.setText("🤖 Cargar Modelo")
        elif not cv_selected:
            self.btn_classify.setText("📄 Seleccionar CV")
        else:
            self.btn_classify.setText("🎯 Clasificar CV")


    def select_cv_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Archivo de CV", "",
            "Archivos PDF (*.pdf);;Archivos de Texto (*.txt);;Todos los archivos (*)"
        )

        if file_path:
            self.selected_cv_file = file_path
            file_name = os.path.basename(file_path)
            
            try:
                file_size = os.path.getsize(file_path)
                file_size_mb = file_size / (1024 * 1024)
                label_text = f"✅ {file_name}\n💾 Tamaño: {file_size_mb:.2f} MB"
            except OSError:
                label_text = f"✅ {file_name}\n💾 No se pudo leer el tamaño."

            self.selected_file_label.setText(label_text)
            self.selected_file_label.setProperty("fileState", "selected")
            self.selected_file_label.style().unpolish(self.selected_file_label)
            self.selected_file_label.style().polish(self.selected_file_label)
            self.update_ui_state()
        else:
            if not self.selected_cv_file:
                self.selected_file_label.setText("🔍 Ningún archivo seleccionado")
                self.selected_file_label.setProperty("fileState", "none")
                self.selected_file_label.style().unpolish(self.selected_file_label)
                self.selected_file_label.style().polish(self.selected_file_label)
            self.update_ui_state()


    def classify_cv(self):
        if not self.current_loaded_model:
            QMessageBox.warning(self, "Sin Modelo", "Por favor, carga un modelo antes de clasificar.")
            return
        if not self.selected_cv_file:
            QMessageBox.warning(self, "Sin CV", "Por favor, selecciona un archivo de CV.")
            return
        if not os.path.exists(self.selected_cv_file):
            QMessageBox.critical(self, "Archivo No Encontrado", "El archivo de CV seleccionado no existe.")
            self.selected_cv_file = None
            self.update_ui_state()
            self.select_cv_file()
            return

        self.btn_classify.setEnabled(False)
        self.btn_classify.setText("🔄 Clasificando...")
        self.btn_classify.start_pulsing()

        self.main_result.setHtml("<p style='color:#E0E0E0; text-align:center;'>⏳ Preparando clasificación...</p>")
        self.ranking_table.setRowCount(0)

        classifier_to_use = self.dl_classifier if self.current_model_is_dl else self.ml_classifier

        self.classification_worker = ClassificationWorker(
            self.selected_cv_file, classifier_to_use, self.current_model_is_dl
        )
        self.classification_worker.progress_updated.connect(self.update_classification_progress)
        self.classification_worker.classification_completed.connect(self.on_classification_completed)
        self.classification_worker.classification_failed.connect(self.on_classification_failed)
        self.classification_worker.start()
        self.clasificacion_iniciada.emit()

    def update_classification_progress(self, message):
        self.main_result.setHtml(f"<p style='color:#E0E0E0; text-align:center;'>⏳ {message}</p>")


    def on_classification_completed(self, result):
        try:
            self.btn_classify.stop_pulsing()
            predicted_profession = result.get('predicted_profession', 'Desconocido')
            confidence = result.get('confidence', 0.0)
            cv_file = result.get('cv_file', 'N/A')
            model_type_str = result.get('model_type', 'N/A')

            main_text = f"""
            <div style='font-family: "Segoe UI", Arial, sans-serif; color: #E0E0E0; font-size: 12px;'>
                <h2 style='color: #27AE60; margin-bottom: 8px; text-align:center;'>🎯 RESULTADO</h2>
                <div style='background-color: rgba(0,0,0,0.1); padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                    <p><strong>📄 Archivo:</strong> {cv_file}</p>
                    <p><strong>🤖 Modelo:</strong> {self.current_loaded_model} ({model_type_str})</p>
                </div>
                <div style='background-color: rgba(39, 174, 96, 0.2); padding: 15px; border-radius: 8px; text-align: center;'>
                    <h3 style='color: #2ECC71; margin-bottom: 8px;'>PROFESIÓN PREDICHA</h3>
                    <h2 style='color: #58D68D; margin-bottom: 10px; font-size: 18px;'>{predicted_profession.upper()}</h2>
                    <h3 style='color: #F39C12;'>📊 CONFIANZA: {confidence:.1%}</h3>
                </div>
            """
            if confidence >= 0.8: main_text += "<p style='color: #27AE60; font-weight: bold; font-size: 13px; text-align:center; margin-top:10px;'>✅ Alta confianza.</p>"
            elif confidence >= 0.6: main_text += "<p style='color: #F39C12; font-weight: bold; font-size: 13px; text-align:center; margin-top:10px;'>⚠️ Confianza moderada.</p>"
            else: main_text += "<p style='color: #E74C3C; font-weight: bold; font-size: 13px; text-align:center; margin-top:10px;'>❌ Baja confianza.</p>"
            main_text += "</div>"
            self.main_result.setHtml(main_text)

            profession_ranking = result.get('profession_ranking', [])
            if profession_ranking:
                probabilities = {item['profession']: item['probability'] for item in profession_ranking}
                self.populate_ranking_table(probabilities)

        except Exception as e:
            self.on_classification_failed(f"Error procesando resultados: {str(e)}")
        finally:
            self.btn_classify.setEnabled(True)
            self.update_ui_state()
            self.clasificacion_completada.emit()


    def on_classification_failed(self, error_message):
        self.btn_classify.stop_pulsing()
        error_html = f"""
        <div style='font-family: "Segoe UI", Arial, sans-serif; color: #E0E0E0; font-size: 12px;'>
            <h2 style='color: #E74C3C; margin-bottom: 10px; text-align:center;'>❌ ERROR EN CLASIFICACIÓN</h2>
            <div style='background-color: rgba(231, 76, 60, 0.2); padding: 15px; border-radius: 8px; margin-bottom:10px;'>
                <p style='color: #E74C3C; font-weight: bold; text-align: center;'>{error_message}</p>
            </div>
            <p style='color: #BDC3C7; text-align:center;'>Verifica el archivo y el modelo, luego intenta nuevamente.</p>
        </div>
        """
        self.main_result.setHtml(error_html)
        self.btn_classify.setEnabled(True)
        self.update_ui_state()
        QMessageBox.critical(self, "Error de Clasificación", f"Error: {error_message}")


    def populate_ranking_table(self, probabilities):
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        self.ranking_table.setRowCount(len(sorted_probs))

        for row, (profession, probability) in enumerate(sorted_probs):
            profession_item = QTableWidgetItem(profession)
            prob_text = f"{probability:.1%}"
            prob_item = QTableWidgetItem(prob_text)
            prob_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.ranking_table.setItem(row, 0, profession_item)
            self.ranking_table.setItem(row, 1, prob_item)

            if row == 0 :
                font = profession_item.font()
                font.setBold(True)
                profession_item.setFont(font)
                prob_item.setFont(font)
                if probability > 0:
                    text_color = QColor("#FFFFFF") 
                    bg_color = QColor(39,174,96, 90)
                    profession_item.setForeground(text_color)
                    profession_item.setBackground(bg_color)
                    prob_item.setForeground(text_color)
                    prob_item.setBackground(bg_color)