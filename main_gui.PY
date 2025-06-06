import sys
import os 
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QToolBar, QSizePolicy,
    QStyle, QColorDialog, QInputDialog, QFontDialog, 
    QStackedWidget
)
from PyQt6.QtGui import (
    QColor, QBrush, QPen, QPainter, QFont, QAction, QIcon, QPixmap,
    QPalette, QMouseEvent, QImage 
)
from PyQt6.QtCore import Qt, QRectF, QSize, QPoint, QEvent, pyqtSignal
from PyQt6.QtSvg import QSvgRenderer 
from PyQt6.QtSvgWidgets import QSvgWidget 


# --- Importar las vistas desde sus archivos separados ---
from entrenar_vista import seleccion 
from vista_herramientas import VistaHerramientas
from vista_centro_accion import VistaCentroAccion
from vista_importar_exportar import VistaImportarExportar

# --- Icon Resource Function ---
def get_icon(icon_name_or_path, color_str=None): 
    if icon_name_or_path == "app_icon":
        return QApplication.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
    elif icon_name_or_path == "pc":
        return QApplication.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
    elif icon_name_or_path == "windows":
        return QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DriveHDIcon)
    elif icon_name_or_path == "joystick":
        pixmap = QPixmap(24,24); pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(pixmap); p.setPen(QColor("white")); p.setFont(QFont("Arial", 14))
        p.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "🎮"); p.end()
        return QIcon(pixmap)
    elif icon_name_or_path == "dots":
        pixmap = QPixmap(24,24); pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(pixmap); p.setPen(QColor("white")); p.setFont(QFont("Arial", 14))
        p.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "⋮"); p.end()
        return QIcon(pixmap)
    elif icon_name_or_path == "paintbrush":
        pixmap = QPixmap(22,22); pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(pixmap)
        pen_color = QColor(color_str if color_str else "#FFFFFF") 
        p.setPen(pen_color)
        p.setFont(QFont("Segoe UI Symbol", 14)) 
        p.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "🖌️") 
        p.end()
        return QIcon(pixmap)
    if os.path.exists(icon_name_or_path): 
        return QIcon(icon_name_or_path)
    return QIcon()

# --- Custom Title Bar ---
class CustomTitleBar(QWidget):
    MINIMIZE_SYMBOL = "−"; MAXIMIZE_SYMBOL = "🗖"; RESTORE_SYMBOL = "🗗"; CLOSE_SYMBOL = "✕"
    theme_button_clicked = pyqtSignal()
    def __init__(self, title, parent_window): 
        super().__init__(parent_window)
        self.parent_window = parent_window; self.setFixedHeight(40); self.setObjectName("CustomTitleBar")
        layout = QHBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)
        self.app_icon_label = QLabel()
        app_icon_pix = self.parent_window.windowIcon().pixmap(QSize(22,22))
        if not app_icon_pix.isNull():
            self.app_icon_label.setPixmap(app_icon_pix)
            self.app_icon_label.setContentsMargins(8,0,5,0); layout.addWidget(self.app_icon_label)
        self.title_label = QLabel(title); self.title_label.setObjectName("TitleBarText")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.title_label); layout.addStretch(1)
        self.btn_theme = QPushButton()
        self.btn_theme.setIcon(get_icon("paintbrush", self.parent_window.color_text_light))
        self.btn_theme.setObjectName("ThemeButton"); self.btn_theme.setToolTip("Apariencia")
        btn_theme_width = 35; self.btn_theme.setFixedSize(btn_theme_width, self.height())
        self.btn_theme.clicked.connect(self.theme_button_clicked.emit); layout.addWidget(self.btn_theme)
        self.btn_minimize = QPushButton(self.MINIMIZE_SYMBOL); self.btn_maximize = QPushButton(self.MAXIMIZE_SYMBOL); self.btn_close = QPushButton(self.CLOSE_SYMBOL)
        control_button_width = 45 
        for btn, obj_name, tooltip in [(self.btn_minimize, "MinimizeButton", "Minimizar"), (self.btn_maximize, "MaximizeButton", "Maximizar"), (self.btn_close, "CloseButton", "Cerrar")]:
            btn.setObjectName(obj_name); btn.setFixedSize(control_button_width, self.height()); btn.setToolTip(tooltip); btn.clicked.connect(self.handle_button_click); layout.addWidget(btn)
        self.update_maximize_button_icon(); self._drag_start_position = None
    def update_maximize_button_icon(self):
        if self.parent_window.isMaximized(): self.btn_maximize.setText(self.RESTORE_SYMBOL); self.btn_maximize.setToolTip("Restaurar")
        else: self.btn_maximize.setText(self.MAXIMIZE_SYMBOL); self.btn_maximize.setToolTip("Maximizar")
    def handle_button_click(self):
        sender = self.sender()
        if sender == self.btn_minimize: self.parent_window.showMinimized()
        elif sender == self.btn_maximize:
            if self.parent_window.isMaximized(): self.parent_window.showNormal()
            else: self.parent_window.showMaximized()
        elif sender == self.btn_close: self.parent_window.close()
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            clickable_widgets = [self.btn_theme, self.btn_minimize, self.btn_maximize, self.btn_close]
            is_on_button = any(btn_widget.geometry().contains(event.pos()) for btn_widget in clickable_widgets)
            if not is_on_button: self._drag_start_position = event.globalPosition().toPoint() - self.parent_window.pos(); event.accept()
            else: event.ignore() 
        else: event.ignore()
    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_start_position is not None: self.parent_window.move(event.globalPosition().toPoint() - self._drag_start_position); event.accept()
        else: event.ignore()
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start_position is not None: self._drag_start_position = None; event.accept()
        else: event.ignore()
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            clickable_widgets = [self.btn_theme, self.btn_minimize, self.btn_maximize, self.btn_close]
            is_on_button = any(btn_widget.geometry().contains(event.pos()) for btn_widget in clickable_widgets)
            if not is_on_button:
                if self.parent_window.isMaximized(): self.parent_window.showNormal()
                else: self.parent_window.showMaximized()
                event.accept()
            else: event.ignore()
        else: event.ignore()

# --- Sidebar Button Custom Widget ---
class SidebarButtonWidget(QWidget):
    clicked = pyqtSignal()

    def __init__(self, icon_path: str, text: str, parent=None): 
        super().__init__(parent)
        self.setObjectName("SidebarButtonWidget")
        self.setMinimumHeight(70) 
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("active", "false")
        self.icon_path = icon_path 
        self.active_color_hex = "#E74C3C" 

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 8, 5, 8) 
        layout.setSpacing(4) 

        self.icon_label = QLabel() 
        self.icon_label.setObjectName("SidebarIconLabel") 
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.text_label = QLabel(text)
        self.text_label.setObjectName("SidebarTextLabel")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font_text = QFont()
        font_text.setPointSize(9) 
        self.text_label.setFont(font_text)

        layout.addStretch(1) 
        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)
        layout.addStretch(1) 

        self._update_icon_pixmap(False) 

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def set_active(self, active: bool, active_color_hex: str = None):
        self.setProperty("active", "true" if active else "false")
        if active and active_color_hex:
            self.active_color_hex = active_color_hex
        
        self._update_icon_pixmap(active)
        
        self.style().unpolish(self) 
        self.style().polish(self)

    def _update_icon_pixmap(self, active: bool):
        pixmap_size = QSize(28, 28) 
        
        if os.path.exists(self.icon_path) and self.icon_path.lower().endswith((".svg", ".png", ".jpg", ".jpeg", ".ico")):
            if self.icon_path.lower().endswith(".svg"):
                try:
                    from PyQt6.QtSvg import QSvgRenderer
                    renderer = QSvgRenderer(self.icon_path)
                    if not renderer.isValid():
                        self.icon_label.setText("!")
                        return

                    image = QImage(pixmap_size, QImage.Format.Format_ARGB32_Premultiplied)
                    image.fill(Qt.GlobalColor.transparent)

                    painter = QPainter(image)
                    renderer.render(painter)

                    if active:
                        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
                        painter.fillRect(image.rect(), QColor(self.active_color_hex))

                    painter.end()
                    self.icon_label.setPixmap(QPixmap.fromImage(image))
                except ImportError:
                    icon = QIcon(self.icon_path)
                    pixmap = icon.pixmap(pixmap_size)
                    self.icon_label.setPixmap(pixmap)
            else:
                icon = QIcon(self.icon_path)
                pixmap = icon.pixmap(pixmap_size)

                if active:
                    colored_pixmap = QPixmap(pixmap.size())
                    colored_pixmap.fill(Qt.GlobalColor.transparent)
                    painter = QPainter(colored_pixmap)
                    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                    painter.drawPixmap(0, 0, pixmap)
                    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
                    painter.fillRect(colored_pixmap.rect(), QColor(self.active_color_hex))
                    painter.end()
                    self.icon_label.setPixmap(colored_pixmap)
                else:
                    self.icon_label.setPixmap(pixmap)

        else: 
            font_icon = QFont(); font_icon.setPointSize(22)
            self.icon_label.setFont(font_icon)
            self.icon_label.setText(self.icon_path) 
            
            main_window = self.window()
            if isinstance(main_window, MainWindow):
                 inactive_color = main_window.color_sidebar_icon_inactive
            else:
                inactive_color = "#8A959E" 

            current_color = self.active_color_hex if active else inactive_color
            self.icon_label.setStyleSheet(f"color: {current_color}; background-color: transparent;")


# --- Circular Actualizar Button ---
class CircularActualizarButton(QWidget):
    def __init__(self, accent_color_hex="#E74C3C", parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200); self.text = "ACTUALIZAR"
        self.accent_color = QColor(accent_color_hex)
    def set_accent_color(self, accent_color_hex: str):
        self.accent_color = QColor(accent_color_hex)
        self.update()
    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        widget_rect = self.rect(); drawing_rect = widget_rect.adjusted(2, 2, -2, -2)
        outer_diameter = min(drawing_rect.width(), drawing_rect.height()) - 4
        center_x = widget_rect.width() / 2; center_y = widget_rect.height() / 2
        track_thickness = 12; progress_thickness = 10
        pen_track = QPen(QColor("#4A4A4A"), track_thickness); pen_track.setCapStyle(Qt.PenCapStyle.FlatCap); painter.setPen(pen_track)
        painter.drawEllipse(QRectF(center_x - outer_diameter / 2, center_y - outer_diameter / 2, outer_diameter, outer_diameter))
        pen_progress = QPen(self.accent_color, progress_thickness); pen_progress.setCapStyle(Qt.PenCapStyle.FlatCap); painter.setPen(pen_progress)
        painter.drawEllipse(QRectF(center_x - outer_diameter / 2, center_y - outer_diameter / 2, outer_diameter, outer_diameter))
        inner_fill_diameter = outer_diameter - track_thickness - 2
        brush_inner = QBrush(QColor("#303841")); painter.setBrush(brush_inner); painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(center_x - inner_fill_diameter / 2, center_y - inner_fill_diameter / 2, inner_fill_diameter, inner_fill_diameter))
        font = QFont(); font.setPointSize(15); font.setBold(True); painter.setFont(font); painter.setPen(QColor(Qt.GlobalColor.white))
        painter.drawText(QRectF(0, 0, widget_rect.width(), widget_rect.height()), Qt.AlignmentFlag.AlignCenter, self.text)
    def sizeHint(self): return QSize(220, 220)

# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.app_font_family = QApplication.font().family()
        self.app_font_size_pt = QApplication.font().pointSize()
        if self.app_font_size_pt <= 0: self.app_font_size_pt = 10 
        self.init_dark_theme_colors() 

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("ClasificaTalento PRO") 
        self.setWindowIcon(get_icon("app_icon")) 
        self.setGeometry(100, 100, 1200, 800)

        self.overall_widget = QWidget()
        self.overall_widget.setObjectName("OverallWidgetFrame")
        overall_layout = QVBoxLayout(self.overall_widget)
        overall_layout.setContentsMargins(1,1,1,1); overall_layout.setSpacing(0)

        self.title_bar = CustomTitleBar("ClasificaTalento PRO", self)
        self.title_bar.theme_button_clicked.connect(self.open_theme_options)
        overall_layout.addWidget(self.title_bar)

        main_content_widget = QWidget()
        main_content_layout = QHBoxLayout(main_content_widget)
        main_content_layout.setContentsMargins(0,0,0,0); main_content_layout.setSpacing(0)

        sidebar = QWidget(); sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar); sidebar_layout.setContentsMargins(5,10,5,10); sidebar_layout.setSpacing(5) 
        sidebar_layout.addStretch(1) 
        self.sidebar_buttons = []
        self.button_to_page_map = {} 
        
        base_path = os.path.dirname(os.path.abspath(__file__)) 
        icons_dir = os.path.join(base_path, "icons_png")

        button_data = [
            (os.path.join(icons_dir, "inicio.png"), "Inicio"),
            (os.path.join(icons_dir, "entrenamiento.png"), "Entrenamiento"),
            (os.path.join(icons_dir, "modelos.png"), "Modelos"),
            (os.path.join(icons_dir, "clasificacion_cv.png"), "Clasificar CV"),
            (os.path.join(icons_dir, "importar_exportar.png"), "Import/Export")
        ]
        
        actual_button_data = []
        for icon_path, text in button_data:
            if os.path.exists(icon_path):
                actual_button_data.append((icon_path, text))
            else:
                fallback_chars = {"Inicio":"🏠", "Entrenamiento":"🏋️", "Modelos":"🧠", "Clasificar CV":"📄", "Import/Export":"🔄"}
                actual_button_data.append((fallback_chars.get(text, "?"), text))


        for i, (icon_path_or_char, text) in enumerate(actual_button_data):
            button = SidebarButtonWidget(icon_path_or_char, text, parent=self)
            button.clicked.connect(self.handle_sidebar_button_click)
            sidebar_layout.addWidget(button); self.sidebar_buttons.append(button)
            self.button_to_page_map[button] = i 

        sidebar_layout.addStretch(1) 
        main_content_layout.addWidget(sidebar); sidebar.setFixedWidth(120)

        self.central_stacked_widget = QStackedWidget()
        self.central_stacked_widget.setObjectName("CentralStackedWidget")

        # Página 0: Inicio
        self.inicio_page_widget = QWidget() 
        self.inicio_page_widget.setObjectName("InicioPage")
        inicio_page_layout = QVBoxLayout(self.inicio_page_widget)
        inicio_page_layout.setContentsMargins(25, 15, 25, 15); inicio_page_layout.setSpacing(12)
        
        self.notification1_widget_inicio = QWidget(); notification1_layout_inicio = QHBoxLayout(self.notification1_widget_inicio); notification1_layout_inicio.setContentsMargins(0,0,0,0)
        icon_label1_inicio = QLabel("✨"); icon_font = icon_label1_inicio.font(); icon_font.setPointSize(18); icon_label1_inicio.setFont(icon_font); icon_label1_inicio.setObjectName("notificationIcon")
        text_label1_inicio = QLabel("Bienvenido a ClasificaTalento PRO. Selecciona una opción de la barra lateral."); text_label1_inicio.setObjectName("notificationText")
        notification1_layout_inicio.addWidget(icon_label1_inicio); notification1_layout_inicio.addWidget(text_label1_inicio); notification1_layout_inicio.addStretch()
        
        self.actualizar_button_inicio = CircularActualizarButton(accent_color_hex=self.color_accent_red) 

        inicio_page_layout.addWidget(self.notification1_widget_inicio)
        inicio_page_layout.addStretch(1)
        inicio_page_layout.addWidget(self.actualizar_button_inicio, 0, Qt.AlignmentFlag.AlignCenter)
        inicio_page_layout.addStretch(1)
        self.central_stacked_widget.addWidget(self.inicio_page_widget)

        # Página 1: Entrenamiento
        self.entrenamiento_page_widget = seleccion(parent_window=self) 
        self.entrenamiento_page_widget.setObjectName("EntrenamientoPage")
        self.central_stacked_widget.addWidget(self.entrenamiento_page_widget)

        # Página 2: Modelos (usa VistaHerramientas)
        self.modelos_page_widget = VistaHerramientas() 
        self.modelos_page_widget.setObjectName("ModelosPage")
        self.central_stacked_widget.addWidget(self.modelos_page_widget)

        # Página 3: Clasificar CV (usa VistaCentroAccion)
        self.clasificar_cv_page_widget = VistaCentroAccion() 
        self.clasificar_cv_page_widget.setObjectName("ClasificarCVPage")
        self.central_stacked_widget.addWidget(self.clasificar_cv_page_widget)

        # Página 4: Importar/Exportar
        self.importar_exportar_page_widget = VistaImportarExportar()
        self.importar_exportar_page_widget.setObjectName("ImportarExportarPage")
        self.central_stacked_widget.addWidget(self.importar_exportar_page_widget)
        
        main_content_layout.addWidget(self.central_stacked_widget, 1)
        overall_layout.addWidget(main_content_widget)
        self.setCentralWidget(self.overall_widget)

        if self.sidebar_buttons: self.setActiveSidebarButton(self.sidebar_buttons[0])

        self.right_toolbar = QToolBar("Barra de Herramientas Derecha"); self.right_toolbar.setOrientation(Qt.Orientation.Vertical); self.right_toolbar.setIconSize(QSize(20,20))
        try: self.addToolBar(Qt.ToolBarArea.RightToolBarArea, self.right_toolbar)
        except Exception as e: print(f"Could not add toolbar directly: {e}.")
        action_pc = QAction(get_icon("pc"),"Documentación",self); action_windows = QAction(get_icon("windows"),"Windows",self)
        action_joystick = QAction(get_icon("joystick"),"Juegos",self); action_dots = QAction(get_icon("dots"),"Más",self)

        # Conectar la acción PC con la documentación
        action_pc.triggered.connect(self.open_documentation)
        action_pc.setToolTip("Abrir documentación del sistema")

        self.right_toolbar.addAction(action_pc); self.right_toolbar.addAction(action_windows); self.right_toolbar.addAction(action_joystick); self.right_toolbar.addAction(action_dots)
        
        self.apply_stylesheet()

    def init_dark_theme_colors(self):
        self.color_sidebar_bg = "#1E272E"; self.color_central_bg = "#252C33"
        self.color_text_light = "#E0E0E0"; self.color_text_medium = "#BDC3C7"; self.color_text_dark = "#7F8C8D"
        self.color_accent_red = "#E74C3C"; self.color_accent_blue = "#3498DB"; self.color_accent_green = "#27AE60"; self.color_accent_purple = "#8E44AD"; self.color_accent_orange = "#F39C12";
        self.color_titlebar_bg = "#181D20"; self.color_window_border = "#050505"
        self.color_sidebar_icon_inactive = "#8A959E"; self.color_sidebar_button_active_bg = "#2C3A47"
        self.color_sidebar_button_hover_bg = "#3A4750"; self.color_sidebar_button_active_hover_bg = "#35424D"
        self.color_titlebar_button_hover = "#3A4045"; self.color_titlebar_close_hover = self.color_accent_red
        darker_accent = QColor(self.color_accent_red).darker(120); self.color_titlebar_close_pressed = darker_accent.name()

        # Colores para componentes
        self.color_combo_bg = "#2C3E50"; self.color_combo_border = self.color_text_dark; self.color_combo_hover_bg = "#34495E"
        self.color_combo_item_selected_bg = self.color_accent_blue; self.color_combo_text = self.color_text_light
        self.color_input_bg = "#34495E"; self.color_input_border = self.color_text_dark;
        self.color_groupbox_bg = "#333B47"; self.color_groupbox_border = "#4A5568";
        self.color_table_bg = self.color_central_bg; self.color_table_alt_bg = self.color_groupbox_bg; self.color_table_grid = self.color_groupbox_border;
        self.color_table_header_bg = self.color_accent_orange;
        self.color_progressbar_bg = self.color_input_bg
        self.color_dialog_bg_start = "#2C3E50"; self.color_dialog_bg_end = "#34495E";


    def init_light_theme_colors(self):
        self.color_sidebar_bg = "#E8EAF6"; self.color_central_bg = "#FAFAFA"
        self.color_text_light = "#263238"; self.color_text_medium = "#546E7A"; self.color_text_dark = "#90A4AE"
        self.color_accent_red = "#D32F2F"; self.color_accent_blue = "#1976D2"; self.color_accent_green = "#388E3C"; self.color_accent_purple = "#7B1FA2"; self.color_accent_orange = "#F57C00";
        self.color_titlebar_bg = "#E0E0E0"; self.color_window_border = "#BDBDBD"
        self.color_sidebar_icon_inactive = "#78909C"; self.color_sidebar_button_active_bg = "#C5CAE9"
        self.color_sidebar_button_hover_bg = "#E0E5F0"; self.color_sidebar_button_active_hover_bg = "#D0D0D0"
        self.color_titlebar_button_hover = "#CFD8DC"; self.color_titlebar_close_hover = self.color_accent_red
        darker_accent = QColor(self.color_accent_red).darker(120); self.color_titlebar_close_pressed = darker_accent.name()
        
        # Colores para componentes
        self.color_combo_bg = "#CFD8DC"; self.color_combo_border = "#B0BEC5"; self.color_combo_hover_bg = "#B0BEC5"
        self.color_combo_item_selected_bg = "#90A4AE"; self.color_combo_text = self.color_text_light
        self.color_input_bg = "#ECEFF1"; self.color_input_border = self.color_text_dark
        self.color_groupbox_bg = "#FFFFFF"; self.color_groupbox_border = "#CFD8DC"
        self.color_table_bg = self.color_central_bg; self.color_table_alt_bg = "#F5F5F5"; self.color_table_grid = self.color_groupbox_border
        self.color_table_header_bg = self.color_accent_orange
        self.color_progressbar_bg = self.color_input_bg
        self.color_dialog_bg_start = "#EEEEEE"; self.color_dialog_bg_end = "#E0E0E0";

    def open_theme_options(self):
        items = ["Color de Acento", "Color de Fondo Principal", "Color de Texto Global", "Fuente de la Aplicación", 
                 "Tema Oscuro (Predefinido)", "Tema Claro (Predefinido)"]
        item, ok = QInputDialog.getItem(self, "Configuración de Apariencia",
                                        "Selecciona qué quieres cambiar:", items, 0, False)
        if ok and item:
            if item == "Color de Acento": self.change_accent_color()
            elif item == "Color de Fondo Principal": self.change_main_background_color()
            elif item == "Color de Texto Global": self.change_global_text_color()
            elif item == "Fuente de la Aplicación": self.change_application_font()
            elif item == "Tema Oscuro (Predefinido)":
                self.init_dark_theme_colors()
                self.apply_stylesheet()
                self.update_dependent_widgets_color()
            elif item == "Tema Claro (Predefinido)":
                self.init_light_theme_colors()
                self.apply_stylesheet()
                self.update_dependent_widgets_color()

    def change_global_text_color(self):
        initial_color = QColor(self.color_text_light)
        new_color = QColorDialog.getColor(initial_color, self, "Seleccionar Color de Texto Global")
        if new_color.isValid():
            self.color_text_light = new_color.name()
            self.color_text_medium = new_color.lighter(150).name()
            self.color_text_dark = new_color.darker(110).name()
            self.apply_stylesheet()
            self.update_dependent_widgets_color()

    def change_accent_color(self):
        initial_color = QColor(self.color_accent_red)
        new_color = QColorDialog.getColor(initial_color, self, "Seleccionar Nuevo Color de Acento")
        if new_color.isValid():
            # Asignar todos los colores de acento basados en el nuevo color
            self.color_accent_red = new_color.name()
            q_new_color = QColor(new_color)
            self.color_accent_blue = q_new_color.lighter(115).name()
            self.color_accent_green = q_new_color.darker(115).name()
            self.color_accent_purple = q_new_color.darker(105).name()
            self.color_accent_orange = q_new_color.lighter(125).name()
            self.color_titlebar_close_hover = self.color_accent_red
            self.color_titlebar_close_pressed = q_new_color.darker(120).name()
            self.apply_stylesheet()
            self.update_dependent_widgets_color()
    
    def change_main_background_color(self):
        """
        CORREGIDO: Este método ahora deriva todos los colores relacionados del
        color de fondo seleccionado por el usuario, en lugar de resetear al tema.
        """
        initial_color = QColor(self.color_central_bg)
        new_color = QColorDialog.getColor(initial_color, self, "Seleccionar Color de Fondo Principal")
        
        if new_color.isValid():
            self.color_central_bg = new_color.name()
            q_central_color = QColor(self.color_central_bg)
            
            is_light_bg = (q_central_color.red() * 0.299 + q_central_color.green() * 0.587 + q_central_color.blue() * 0.114) > 186

            # Derivar todos los demás colores del nuevo fondo
            if is_light_bg:
                # Derivaciones para tema claro
                self.color_sidebar_bg = q_central_color.darker(105).name()
                self.color_titlebar_bg = q_central_color.darker(110).name()
                self.color_groupbox_bg = "#FFFFFF"
                self.color_input_bg = q_central_color.darker(102).name()
                self.color_text_light = "#212121"
                self.color_text_medium = "#616161"
                self.color_text_dark = "#9E9E9E"
                self.color_window_border = "#BDBDBD"
                self.color_groupbox_border = "#E0E0E0"
                self.color_input_border = "#BDBDBD"
                self.color_sidebar_icon_inactive = "#757575"
                self.color_sidebar_button_hover_bg = q_central_color.darker(110).name()
            else:
                # Derivaciones para tema oscuro
                self.color_sidebar_bg = q_central_color.lighter(110).name()
                self.color_titlebar_bg = q_central_color.darker(110).name()
                self.color_groupbox_bg = q_central_color.lighter(105).name()
                self.color_input_bg = q_central_color.lighter(115).name()
                self.color_text_light = "#E0E0E0"
                self.color_text_medium = "#BDC3C7"
                self.color_text_dark = "#7F8C8D"
                self.color_window_border = "#050505"
                self.color_groupbox_border = q_central_color.lighter(120).name()
                self.color_input_border = self.color_text_dark
                self.color_sidebar_icon_inactive = "#8A959E"
                self.color_sidebar_button_hover_bg = q_central_color.lighter(115).name()

            # Colores comunes derivados
            self.color_combo_bg = self.color_input_bg
            self.color_combo_text = self.color_text_light
            self.color_combo_border = self.color_input_border
            self.color_combo_hover_bg = QColor(self.color_input_bg).lighter(110).name() if not is_light_bg else QColor(self.color_input_bg).darker(110).name()
            self.color_table_alt_bg = self.color_groupbox_bg
            self.color_table_bg = self.color_central_bg
            self.color_table_grid = self.color_groupbox_border
            self.color_progressbar_bg = self.color_input_bg
            
            # Reaplicar estilos y actualizar widgets dependientes
            self.apply_stylesheet()
            self.update_dependent_widgets_color()
            
    def change_application_font(self):
        current_font = QFont(self.app_font_family, self.app_font_size_pt)
        font, ok = QFontDialog.getFont(current_font, self, "Seleccionar Fuente Principal")
        if ok:
            QApplication.setFont(font)
            self.app_font_family = font.family()
            self.app_font_size_pt = font.pointSize()
            if self.app_font_size_pt <= 0 : self.app_font_size_pt = font.pixelSize() 
            if self.app_font_size_pt <= 0 : self.app_font_size_pt = 10 
            self.apply_stylesheet(); self.update_dependent_widgets_color()
            
    def open_documentation(self):
        """Abre la ventana de documentación del sistema"""
        try:
            # Crear y mostrar la ventana de documentación
            self.documentation_dialog = DocumentationDialog(self)
            self.documentation_dialog.show()
        except Exception as e:
            print(f"Error al abrir la documentación: {e}")

    def update_dependent_widgets_color(self):
        if hasattr(self, 'actualizar_button_inicio'): 
            self.actualizar_button_inicio.set_accent_color(self.color_accent_red)
        if hasattr(self, 'title_bar') and hasattr(self.title_bar, 'btn_theme'):
            self.title_bar.btn_theme.setIcon(get_icon("paintbrush", color_str=self.color_text_light))
        for btn in self.sidebar_buttons:
            is_active = btn.property("active") == "true"
            btn.set_active(is_active, self.color_accent_red)

    def handle_sidebar_button_click(self):
        clicked_button = self.sender()
        if isinstance(clicked_button, SidebarButtonWidget):
            self.setActiveSidebarButton(clicked_button)

    def setActiveSidebarButton(self, active_button: SidebarButtonWidget):
        page_index = self.button_to_page_map.get(active_button, 0) 
        for button in self.sidebar_buttons:
            is_active = (button == active_button)
            button.set_active(is_active, self.color_accent_red)
        if hasattr(self, 'central_stacked_widget'):
            self.central_stacked_widget.setCurrentIndex(page_index)

    def changeEvent(self, event: QEvent):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange: 
            if hasattr(self, 'title_bar') and self.title_bar:
                self.title_bar.update_maximize_button_icon()
    
    def apply_stylesheet(self):
        font_family_for_qss = self.app_font_family
        font_size_for_qss_val = self.app_font_size_pt
        if font_size_for_qss_val <=0: font_size_for_qss_val = 10 
        font_size_for_qss = f"{font_size_for_qss_val}pt"

        accent_blue_rgba = f"rgba({QColor(self.color_accent_blue).red()}, {QColor(self.color_accent_blue).green()}, {QColor(self.color_accent_blue).blue()}, 40)"

        self.setStyleSheet(f"""
            /* --- GENERAL --- */
            QWidget {{
                font-family: "{font_family_for_qss}";
                color: {self.color_text_light};
                background-color: {self.color_central_bg};
            }}
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
            QScrollArea > QWidget > QWidget {{
                 background-color: transparent;
            }}
            
            /* --- BARRA DE TÍTULO --- */
            #CustomTitleBar {{ background-color: {self.color_titlebar_bg}; }}
            #TitleBarText {{ color: {self.color_text_light}; font-size: 13px; padding-left: 8px; font-weight: bold;}}
            #CustomTitleBar QPushButton {{ background-color: transparent; color: {self.color_text_light}; border: none; font-size: 16px; font-family: "Segoe UI Symbol", "DejaVu Sans", "{font_family_for_qss}"; }}
            #CustomTitleBar QPushButton#ThemeButton {{ font-size: 14px; }}
            #CustomTitleBar QPushButton:hover {{ background-color: {self.color_titlebar_button_hover}; }}
            #CustomTitleBar QPushButton#CloseButton:hover {{ background-color: {self.color_titlebar_close_hover}; color: white; }}
            #CustomTitleBar QPushButton:pressed {{ background-color: #4A5055; }}
            #CustomTitleBar QPushButton#CloseButton:pressed {{ background-color: {self.color_titlebar_close_pressed}; }}

            /* --- BARRA LATERAL (SIDEBAR) --- */
            #sidebar {{ background-color: {self.color_sidebar_bg}; }}
            SidebarButtonWidget {{
                background-color: transparent;
                border-radius: 8px;
                margin: 2px;
                border: 2px solid transparent;
            }}
            SidebarButtonWidget[active="false"]:hover {{
                background-color: {self.color_sidebar_button_hover_bg};
            }}
            SidebarButtonWidget[active="true"] {{
                background-color: {self.color_sidebar_button_active_bg};
                border: 2px solid {self.color_accent_red};
            }}
            SidebarButtonWidget #SidebarTextLabel {{
                color: {self.color_text_medium};
                font-size: 9pt;
            }}
            SidebarButtonWidget[active="true"] #SidebarTextLabel {{
                color: {self.color_text_light};
                font-weight: bold;
            }}
            SidebarButtonWidget[active="false"]:hover #SidebarTextLabel {{
                color: {self.color_text_light};
            }}
            
            /* --- ETIQUETAS (LABELS) GENERALES --- */
            QLabel {{
                background-color: transparent;
                color: {self.color_text_light};
            }}
            /* Títulos de sección grandes */
            #Mejorar_MainTitleLabel, #TituloEntrenamiento, #ClasificarCVPage QLabel[font-size="24"], VistaMLEntrenamiento QLabel[font-size="20"], VistaDLEntrenamiento QLabel[font-size="20"], VistaImportarExportar QLabel[font-size="20"] {{
                color: {self.color_text_light}; 
                font-size: 22pt;
                font-weight: bold; 
            }}
            /* Subtítulos de sección */
            #Mejorar_SubtitleLabel, #SubtituloEntrenamiento {{
                color: {self.color_text_medium};
                font-size: 14pt;
            }}
            
            /* --- BOTONES (PUSHBUTTON) GENERALES --- */
            QPushButton {{
                background-color: {self.color_accent_blue};
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                font-weight: bold;
                padding: 9px 18px;
            }}
            QPushButton:hover {{
                background-color: {QColor(self.color_accent_blue).lighter(115).name()};
            }}
            QPushButton:pressed {{
                background-color: {QColor(self.color_accent_blue).darker(115).name()};
            }}
            QPushButton:disabled {{
                background-color: {self.color_text_dark};
                color: {self.color_text_medium};
            }}

            /* --- ENTRADAS DE TEXTO (QLineEdit & QTextEdit) --- */
            QLineEdit, QTextEdit {{
                background-color: {self.color_input_bg};
                color: {self.color_text_light};
                border: 1px solid {self.color_input_border};
                border-radius: 6px;
                padding: 8px;
                font-size: 13px;
            }}
            QLineEdit:focus, QTextEdit:focus {{
                border: 1px solid {self.color_accent_blue};
            }}
            QTextEdit {{
                 font-family: "Segoe UI", Arial, sans-serif;
            }}

            /* --- MENÚS DESPLEGABLES (QComboBox) --- */
            QComboBox {{
                background-color: {self.color_combo_bg};
                color: {self.color_combo_text};
                border: 1px solid {self.color_combo_border};
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 13px;
            }}
            QComboBox:hover {{
                border-color: {self.color_accent_blue};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: {self.color_combo_border};
                border-left-style: solid;
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.color_combo_bg};
                border: 1px solid {self.color_accent_blue};
                color: {self.color_combo_text};
                selection-background-color: {self.color_combo_item_selected_bg};
                padding: 5px;
            }}
            
            /* --- GRUPOS (QGroupBox) --- */
            QGroupBox {{
                font-weight: bold;
                color: {self.color_text_light};
                border: 1px solid {self.color_groupbox_border};
                border-radius: 10px;
                margin-top: 12px; 
                padding: 10px;
                padding-top: 28px;
                background-color: {self.color_groupbox_bg};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                padding: 5px 15px;
                border-radius: 5px;
                color: white;
                font-size: 14px;
                background-color: {self.color_text_dark};
            }}
            
            /* --- PÁGINA DE ENTRENAMIENTO (SELECCIÓN) --- */
            #TrainingOptionCard {{
                background-color: {QColor(self.color_central_bg).lighter(115).name()};
                border-radius: 15px;
                border: 2px solid {self.color_groupbox_border};
            }}
            #TrainingOptionCard:hover {{ border: 2px solid {self.color_accent_blue}; }}
            #CardMainButton[optionType="ml"] {{ background-color: {self.color_accent_blue}; }}
            #CardMainButton[optionType="dl"] {{ background-color: {self.color_accent_red}; }}
            #CardMainButton:hover {{ background-color: {QColor(self.color_accent_red).lighter(115).name()}; }}
            #CardMainButton[optionType="ml"]:hover {{ background-color: {QColor(self.color_accent_blue).lighter(115).name()}; }}
            #CardConfigButton {{
                background-color: transparent;
                color: {self.color_text_medium};
                border: 1px solid {self.color_text_medium};
                border-radius: 18px; font-size: 11pt; padding: 8px 18px;
            }}
            #CardConfigButton:hover {{ color: {self.color_text_light}; border: 1px solid {self.color_text_light}; }}
            #CardDescriptionLabel {{ color: {self.color_text_medium}; font-size: 10pt; }}

            /* --- PÁGINAS DE ENTRENAMIENTO ML y DL --- */
            QGroupBox#ProfessionGroupML {{ border-color: {self.color_accent_blue}; }}
            QGroupBox#ProfessionGroupML::title {{ background-color: {self.color_accent_blue}; }}
            QGroupBox#TrainingGroupML {{ border-color: {self.color_accent_purple}; }}
            QGroupBox#TrainingGroupML::title {{ background-color: {self.color_accent_purple}; }}
            QGroupBox#LogGroupML {{ border-color: {self.color_accent_red}; }}
            QGroupBox#LogGroupML::title {{ background-color: {self.color_accent_red}; }}
            
            QGroupBox#ProfessionGroupDL {{ border-color: {self.color_accent_green}; }}
            QGroupBox#ProfessionGroupDL::title {{ background-color: {self.color_accent_green}; }}
            QGroupBox#TrainingGroupDL {{ border-color: {self.color_accent_purple}; }}
            QGroupBox#TrainingGroupDL::title {{ background-color: {self.color_accent_purple}; }}
            QGroupBox#LogGroupDL {{ border-color: {self.color_accent_green}; }}
            QGroupBox#LogGroupDL::title {{ background-color: {self.color_accent_green}; }}

            #MLTrainingLog, #DLTrainingLog {{
                font-family: "Consolas", "Courier New", monospace;
                font-size: 11px;
            }}
            #DLEpochLabel {{
                font-size: 14px; font-weight: bold; color: {self.color_accent_green};
            }}
            #DLMetricLabel {{ font-size: 12px; font-weight: bold; }}

            /* --- PÁGINA CENTRO DE ACCIÓN --- */
            #HeaderFrame {{ background-color: {self.color_groupbox_bg}; border-radius: 10px; padding: 5px; border: 1px solid {self.color_accent_blue}; }}
            #HeaderFrame QLabel {{ color: {QColor(self.color_accent_blue).lighter(110).name()}; font-size: 24pt; }}
            PulsingButton, #PulsingButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {self.color_accent_blue}, stop:1 {QColor(self.color_accent_blue).darker(120).name()});
                border-radius: 30px; font-size: 15px; padding: 10px;
            }}
            PulsingButton:hover, #PulsingButton:hover {{
                 background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {QColor(self.color_accent_blue).lighter(110).name()}, stop:1 {QColor(self.color_accent_blue).darker(110).name()});
            }}
            #ModelStatusCard[status="no_model"] {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3A4750, stop:1 #303841);
                border: 1px solid {self.color_accent_red}; border-radius: 10px;
            }}
            #ModelStatusCard[status="loaded"] {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {self.color_accent_green}, stop:1 {QColor(self.color_accent_green).darker(120).name()});
                border: 1px solid {QColor(self.color_accent_green).lighter(120).name()}; border-radius: 10px;
            }}
            #ModelStatusCard QLabel {{ color: white; }}
            #selected_file_label[fileState="none"] {{
                color: {self.color_text_dark}; background-color: {self.color_input_bg};
                border: 1px dashed {self.color_text_dark}; border-radius: 6px; padding: 12px; min-height: 35px;
            }}
            #selected_file_label[fileState="selected"] {{
                color: {self.color_accent_green}; background-color: {QColor(self.color_accent_green).lighter(180).name() if QColor(self.color_accent_green).lightnessF() > 0.5 else "rgba(39, 174, 96, 0.15)"};
                border: 1px solid {self.color_accent_green}; border-radius: 6px; padding: 12px;
            }}
            QSplitter::handle {{ background: {self.color_groupbox_border}; width: 2px; }}
            QSplitter::handle:hover {{ background: {self.color_accent_blue}; }}
            
            /* TABLAS (QTableWidget) */
            QTableWidget {{
                background-color: {self.color_table_bg}; color: {self.color_text_light};
                border: 1px solid {self.color_table_header_bg}; border-radius: 6px;
                gridline-color: {self.color_table_grid}; font-size: 11px;
                alternate-background-color: {self.color_table_alt_bg};
            }}
            QTableWidget::item {{ padding: 7px; border-bottom: 1px solid {self.color_table_grid}; }}
            QTableWidget::item:selected {{ background-color: {self.color_accent_orange}; color: white; }}
            QHeaderView::section {{
                background-color: {self.color_table_header_bg}; color: white;
                padding: 7px; border: none; font-weight: bold; font-size: 12px;
            }}
            
            /* --- PÁGINA DE MODELOS/HERRAMIENTAS --- */
            #ModelCard {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3A4750, stop:1 #303841);
                border: 2px solid {self.color_groupbox_border}; border-radius: 15px; padding: 15px;
            }}
            #ModelCard:hover {{ border: 2px solid {self.color_accent_blue}; }}
            QDialog#ModelInfoDialog {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {self.color_dialog_bg_start}, stop:1 {self.color_dialog_bg_end});
                border-radius: 10px;
            }}
            QFrame#InfoCard {{
                background-color: {QColor(self.color_dialog_bg_end).lighter(105).name() if QColor(self.color_dialog_bg_end).lightnessF() < 0.5 else QColor(self.color_dialog_bg_end).darker(105).name()};
                border: 1px solid {self.color_groupbox_border}; border-radius: 8px;
            }}
            QMenu {{
                background-color: {self.color_combo_bg}; color: {self.color_text_light};
                border: 1px solid {self.color_combo_border}; border-radius: 5px; padding: 5px;
            }}
            QMenu::item {{ padding: 8px 20px; border-radius: 3px; }}
            QMenu::item:selected {{ background-color: {self.color_accent_blue}; }}

            /* --- PÁGINA IMPORTAR/EXPORTAR --- */
            QGroupBox#ImportGroup::title {{ background-color: {self.color_accent_blue}; }}
            QGroupBox#ImportGroup {{ border-color: {self.color_accent_blue}; }}
            QGroupBox#ExportGroup::title {{ background-color: {self.color_accent_red}; }}
            QGroupBox#ExportGroup {{ border-color: {self.color_accent_red}; }}
            QGroupBox#ManagementGroup::title {{ background-color: {self.color_accent_orange}; }}
            QGroupBox#ManagementGroup {{ border-color: {self.color_accent_orange}; }}
            #DropArea {{
                border: 3px dashed {self.color_accent_blue};
                border-radius: 12px;
                background-color: {accent_blue_rgba};
            }}
            #DropArea QLabel {{
                color: {self.color_accent_blue};
            }}
            
            /* --- BARRAS DE PROGRESO --- */
            QProgressBar {{
                border: 1px solid {self.color_accent_green}; border-radius: 8px;
                text-align: center; color: {self.color_text_light}; font-weight: bold;
                background-color: {self.color_progressbar_bg};
            }}
            QProgressBar::chunk {{ background-color: {self.color_accent_green}; border-radius: 6px; }}

            /* --- LISTAS --- */
            QListWidget {{
                background-color: {self.color_input_bg}; color: {self.color_text_light};
                border: 1px solid {self.color_input_border}; border-radius: 5px;
            }}
            QListWidget::item:selected {{ background-color: {self.color_accent_blue}; color: white; }}
        """)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName("ClasificaTalento PRO") 
    app.setOrganizationName("TalentHunter") 
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())