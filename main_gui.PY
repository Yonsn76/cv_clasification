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
        # Intenta cargar un icono específico de la app si existe
        # Ejemplo, reemplaza con tu ruta o usa el placeholder
        # app_icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "app_logo.png") 
        # if os.path.exists(app_icon_path):
        #     return QIcon(app_icon_path)
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

    def __init__(self, icon_path: str, text: str, parent=None): # 'parent' es el nombre correcto
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
        
        # Verificar si icon_path es una ruta de archivo o un carácter
        if os.path.exists(self.icon_path) and self.icon_path.lower().endswith((".svg", ".png", ".jpg", ".jpeg", ".ico")):
            if self.icon_path.lower().endswith(".svg"):
                # Manejar archivos SVG
                try:
                    from PyQt6.QtSvg import QSvgRenderer
                    renderer = QSvgRenderer(self.icon_path)
                    if not renderer.isValid():
                        self.icon_label.setText("!")
                        print(f"Error: No se pudo cargar/renderizar SVG desde '{self.icon_path}'")
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
                    print(f"Icono SVG cargado exitosamente: {self.icon_path}")
                except ImportError:
                    print(f"Módulo SVG no disponible, usando QIcon como fallback para: {self.icon_path}")
                    # Fallback usando QIcon
                    icon = QIcon(self.icon_path)
                    pixmap = icon.pixmap(pixmap_size)
                    self.icon_label.setPixmap(pixmap)
            else:
                # Manejar archivos PNG, JPG, etc.
                icon = QIcon(self.icon_path)
                pixmap = icon.pixmap(pixmap_size)

                if active:
                    # Aplicar color de acento a la imagen
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

                print(f"Icono PNG cargado exitosamente: {self.icon_path}")
        else: # Asumir que es un carácter de texto (fallback)
            font_icon = QFont(); font_icon.setPointSize(22) # Usar el tamaño de icono definido en el widget
            self.icon_label.setFont(font_icon)
            self.icon_label.setText(self.icon_path) # Mostrar el carácter/ruta como texto
            
            # Intentar obtener el color del tema desde la ventana principal para el estado inactivo del texto-icono
            # Esto es un poco indirecto, idealmente el tema se manejaría de forma más centralizada
            inactive_color = self.active_color_hex # Fallback si no se puede obtener el color inactivo
            # El widget padre directo es el QWidget de la sidebar, su padre es main_content_widget, luego overall_widget, luego MainWindow
            main_window = self.window() # Obtiene la ventana de nivel superior (MainWindow)
            if isinstance(main_window, MainWindow):
                 inactive_color = main_window.color_sidebar_icon_inactive

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
        self.setGeometry(100, 100, 1000, 720) 

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
        
        # Verificar si los archivos de iconos existen
        actual_button_data = []
        for icon_path, text in button_data:
            if os.path.exists(icon_path) and icon_path.lower().endswith((".svg", ".png", ".jpg", ".jpeg", ".ico")):
                actual_button_data.append((icon_path, text))
                print(f"Icono encontrado: {icon_path}")
            else:
                print(f"Advertencia: Icono '{icon_path}' no encontrado. Usando fallback de texto para '{text}'.")
                # Asignar un carácter de fallback si el archivo no existe
                fallback_chars = {"Inicio":"🏠", "Entrenamiento":"🏋️", "Modelos":"🧠", "Clasificar CV":"📄", "Import/Export":"🔄"}
                actual_button_data.append((fallback_chars.get(text, "?"), text))


        for i, (icon_path_or_char, text) in enumerate(actual_button_data):
            button = SidebarButtonWidget(icon_path_or_char, text, parent=self) # CORREGIDO: parent=self
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
        
        # Ejemplo de contenido para la página de Inicio
        self.notification1_widget_inicio = QWidget(); notification1_layout_inicio = QHBoxLayout(self.notification1_widget_inicio); notification1_layout_inicio.setContentsMargins(0,0,0,0)
        icon_label1_inicio = QLabel("✨"); icon_font = icon_label1_inicio.font(); icon_font.setPointSize(18); icon_label1_inicio.setFont(icon_font); icon_label1_inicio.setObjectName("notificationIcon") # Puede necesitar estilo específico si el icono es texto
        text_label1_inicio = QLabel("Bienvenido a ClasificaTalento PRO. Selecciona una opción de la barra lateral."); text_label1_inicio.setObjectName("notificationText")
        notification1_layout_inicio.addWidget(icon_label1_inicio); notification1_layout_inicio.addWidget(text_label1_inicio); notification1_layout_inicio.addStretch()
        
        self.actualizar_button_inicio = CircularActualizarButton(accent_color_hex=self.color_accent_red) 

        inicio_page_layout.addWidget(self.notification1_widget_inicio)
        inicio_page_layout.addStretch(1)
        inicio_page_layout.addWidget(self.actualizar_button_inicio, 0, Qt.AlignmentFlag.AlignCenter)
        inicio_page_layout.addStretch(1)
        self.central_stacked_widget.addWidget(self.inicio_page_widget)

        # Página 1: Entrenamiento
        self.entrenamiento_page_widget = seleccion() 
        self.entrenamiento_page_widget.setObjectName("EntrenamientoPage")
        self.central_stacked_widget.addWidget(self.entrenamiento_page_widget)

        # Página 2: Modelos
        self.modelos_page_widget = VistaHerramientas() 
        self.modelos_page_widget.setObjectName("ModelosPage")
        self.central_stacked_widget.addWidget(self.modelos_page_widget)

        # Página 3: Clasificar CV
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
        action_pc = QAction(get_icon("pc"),"PC",self); action_windows = QAction(get_icon("windows"),"Windows",self)
        action_joystick = QAction(get_icon("joystick"),"Juegos",self); action_dots = QAction(get_icon("dots"),"Más",self)
        self.right_toolbar.addAction(action_pc); self.right_toolbar.addAction(action_windows); self.right_toolbar.addAction(action_joystick); self.right_toolbar.addAction(action_dots)
        
        self.apply_stylesheet()

    def init_dark_theme_colors(self):
        self.color_sidebar_bg = "#1E272E"; self.color_central_bg = "#222a30"
        self.color_text_light = "#E0E0E0"; self.color_text_medium = "#BDC3C7"
        self.color_accent_red = "#E74C3C"; self.color_accent_blue_link = "#3498DB"
        self.color_titlebar_bg = "#181D20"; self.color_window_border = "#050505"
        self.color_sidebar_icon_inactive = "#8A959E"; self.color_sidebar_button_active_bg = "#2C3A47"
        self.color_sidebar_button_hover_celeste = "#4A5568"; self.color_sidebar_button_active_hover_bg = "#35424D"
        self.color_titlebar_button_hover = "#3A4045"; self.color_titlebar_close_hover = self.color_accent_red
        darker_accent = QColor(self.color_accent_red).darker(120); self.color_titlebar_close_pressed = darker_accent.name()
    def init_light_theme_colors(self):
        self.color_sidebar_bg = "#E8EAF6"; self.color_central_bg = "#FAFAFA"
        self.color_text_light = "#263238"; self.color_text_medium = "#546E7A"
        self.color_accent_red = "#D32F2F"; self.color_accent_blue_link = "#1976D2"
        self.color_titlebar_bg = "#E0E0E0"; self.color_window_border = "#BDBDBD"
        self.color_sidebar_icon_inactive = "#78909C"; self.color_sidebar_button_active_bg = "#C5CAE9"
        self.color_sidebar_button_hover_celeste = "#E0E5F0"; self.color_sidebar_button_active_hover_bg = "#D0D0D0"
        self.color_titlebar_button_hover = "#CFD8DC"; self.color_titlebar_close_hover = self.color_accent_red
        darker_accent = QColor(self.color_accent_red).darker(120); self.color_titlebar_close_pressed = darker_accent.name()

    def open_theme_options(self):
        items = ["Color de Acento", "Color de Fondo Principal", "Fuente de la Aplicación", 
                 "Tema Oscuro (Predefinido)", "Tema Claro (Predefinido)"]
        item, ok = QInputDialog.getItem(self, "Configuración de Apariencia",
                                        "Selecciona qué quieres cambiar:", items, 0, False)
        if ok and item:
            if item == "Color de Acento": self.change_accent_color()
            elif item == "Color de Fondo Principal": self.change_main_background_color()
            elif item == "Fuente de la Aplicación": self.change_application_font()
            elif item == "Tema Oscuro (Predefinido)":
                self.init_dark_theme_colors(); self.apply_stylesheet(); self.update_dependent_widgets_color()
            elif item == "Tema Claro (Predefinido)":
                self.init_light_theme_colors(); self.apply_stylesheet(); self.update_dependent_widgets_color()

    def change_accent_color(self):
        initial_color = QColor(self.color_accent_red)
        new_color = QColorDialog.getColor(initial_color, self, "Seleccionar Nuevo Color de Acento")
        if new_color.isValid():
            self.color_accent_red = new_color.name()
            self.color_titlebar_close_hover = self.color_accent_red
            darker_accent = QColor(self.color_accent_red).darker(120)
            self.color_titlebar_close_pressed = darker_accent.name()
            self.apply_stylesheet(); self.update_dependent_widgets_color()
    def change_main_background_color(self):
        initial_color = QColor(self.color_central_bg)
        new_color = QColorDialog.getColor(initial_color, self, "Seleccionar Color de Fondo Principal")
        if new_color.isValid():
            self.color_central_bg = new_color.name()
            self.color_sidebar_bg = QColor(self.color_central_bg).darker(110).name()
            self.color_titlebar_bg = QColor(self.color_central_bg).darker(125).name()
            avg_intensity = (new_color.red() + new_color.green() + new_color.blue()) / 3
            if avg_intensity > 127: 
                self.color_text_light = "#212121"; self.color_text_medium = "#757575"; self.color_sidebar_icon_inactive = "#546E7A"
            else: 
                self.color_text_light = "#E0E0E0"; self.color_text_medium = "#BDC3C7"; self.color_sidebar_icon_inactive = "#8A959E"
            self.apply_stylesheet(); self.update_dependent_widgets_color()
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
            
    def update_dependent_widgets_color(self):
        if hasattr(self, 'actualizar_button_inicio'): 
            self.actualizar_button_inicio.set_accent_color(self.color_accent_red)
        if hasattr(self, 'title_bar') and hasattr(self.title_bar, 'btn_theme'):
            self.title_bar.btn_theme.setIcon(get_icon("paintbrush", color_str=self.color_text_light))
        for btn in self.sidebar_buttons:
            if btn.property("active") == "true": 
                btn._update_icon_pixmap(True) 
            else:
                btn._update_icon_pixmap(False)

    def handle_sidebar_button_click(self):
        clicked_button = self.sender()
        if isinstance(clicked_button, SidebarButtonWidget):
            self.setActiveSidebarButton(clicked_button)

    def setActiveSidebarButton(self, active_button: SidebarButtonWidget):
        page_index = self.button_to_page_map.get(active_button, 0) 
        for button in self.sidebar_buttons:
            is_active = (button == active_button)
            if is_active:
                button.set_active(True, self.color_accent_red) 
            else:
                button.set_active(False)
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

        color_notif_critical_bg = QColor(self.color_accent_red).lighter(110).name() if QColor(self.color_accent_red).lightnessF() < 0.3 else QColor(self.color_accent_red).darker(150).name()
        color_notif_critical_border = self.color_accent_red
        color_notif_critical_icon = self.color_accent_red
        color_notif_warning_bg = "#4A3B2B"; color_notif_warning_border = "#D6893A"; color_notif_warning_icon = "#FFC107";
        color_notif_info_bg = QColor(self.color_central_bg).lighter(115).name(); color_notif_info_border = QColor(self.color_accent_blue_link).darker(110).name(); color_notif_info_icon = self.color_accent_blue_link;
        color_notif_success_bg = "#2D4A3A"; color_notif_success_border = "#3AD689"; color_notif_success_icon = "#5CB85C";
        color_notif_action_button_bg = self.color_sidebar_icon_inactive; 
        color_notif_action_button_hover_bg = QColor(self.color_sidebar_icon_inactive).lighter(120).name();

        self.setStyleSheet(f"""
            QWidget {{ font-family: "{font_family_for_qss}"; }}
            #OverallWidgetFrame {{ background-color: {self.color_central_bg}; border: 1px solid {self.color_window_border}; }}
            #CustomTitleBar {{ background-color: {self.color_titlebar_bg}; }}
            #TitleBarText {{ color: {self.color_text_light}; font-size: 13px; padding-left: 8px; font-weight: bold;}}
            #CustomTitleBar QPushButton {{ background-color: transparent; color: {self.color_text_light}; border: none; font-size: 16px; font-family: "Segoe UI Symbol", "DejaVu Sans", "{font_family_for_qss}"; }}
            #CustomTitleBar QPushButton#ThemeButton {{ font-size: 14px; }}
            #CustomTitleBar QPushButton:hover {{ background-color: {self.color_titlebar_button_hover}; }}
            #CustomTitleBar QPushButton#CloseButton:hover {{ background-color: {self.color_titlebar_close_hover}; color: white; }}
            #CustomTitleBar QPushButton#ThemeButton:hover {{ background-color: {self.color_titlebar_button_hover};}}
            #CustomTitleBar QPushButton:pressed {{ background-color: #4A5055; }}
            #CustomTitleBar QPushButton#CloseButton:pressed {{ background-color: {self.color_titlebar_close_pressed}; }}

            #sidebar {{ background-color: {self.color_sidebar_bg}; }}
            SidebarButtonWidget {{
                background-color: transparent;
                border-radius: 10px;
                margin: 3px;
                padding: 6px;
                border: 1px solid transparent;
            }}

            /* Estado normal - hover gris */
            SidebarButtonWidget[active="false"]:hover {{
                background-color: #3A3A3A;
                border: 1px solid #4A4A4A;
            }}

            /* Estado seleccionado - gris claro */
            SidebarButtonWidget[active="true"] {{
                background-color: #5A5A5A;
                border: 1px solid #777777;
            }}

            /* Estado seleccionado - hover gris más claro */
            SidebarButtonWidget[active="true"]:hover {{
                background-color: #333333;
                border: 1px solid #666666;
            }}

            /* Iconos y texto - estado normal */
            SidebarButtonWidget #SidebarIconLabel {{
                padding-bottom: 2px;
            }}
            SidebarButtonWidget #SidebarTextLabel {{
                color: {self.color_sidebar_icon_inactive};
                font-size: 9pt;
            }}

            /* Iconos y texto - estado seleccionado */
            SidebarButtonWidget[active="true"] #SidebarTextLabel {{
                color: #FFFFFF;
                font-weight: bold;
            }}

            /* Iconos y texto - hover */
            SidebarButtonWidget[active="false"]:hover #SidebarTextLabel {{
                color: #FFFFFF;
                font-weight: 500;
            }}


            #InicioPage, #EntrenamientoPage, #ModelosPage, #ClasificarCVPage, #ImportarExportarPage,
            #seleccion, #VistaHerramientas, #VistaCentroAccion,
            VistaMLEntrenamiento, VistaDLEntrenamiento, #VistaSeleccionContenido {{ /* Añadido VistaSeleccionContenido */
                background-color: {self.color_central_bg}; 
            }}
            QScrollArea {{ /* Hacer QScrollArea transparente */
                background-color: transparent;
                border: none; /* Opcional: quitar borde si lo tuviera por defecto */
            }}
            QScrollArea > QWidget > QWidget {{ /* Para el viewport del QScrollArea */
                 background-color: transparent;
            }}
            #TituloSeccion, #TituloEntrenamiento {{ /* Añadido #TituloEntrenamiento */
                color: {self.color_text_light}; 
                font-size: 18pt; /* Ajustado para consistencia, era 22pt en entrenar_vista */
                font-weight: bold; 
                padding: 20px; 
                margin-bottom: 15px; /* Margen inferior para #TituloEntrenamiento */
            }}
            #DescripcionSeccion, #SubtituloEntrenamiento {{ /* Añadido #SubtituloEntrenamiento */
                color: {self.color_text_medium}; 
                font-size: 12pt; /* Ajustado para consistencia, era 14pt en entrenar_vista */
                padding: 10px; 
                margin-bottom: 25px; /* Margen inferior para #SubtituloEntrenamiento */
            }}
            #InstruccionEntrenamiento {{
                color: {self.color_text_medium}; /* Era #95A5A6 */
                font-size: 11pt; /* Era 12px */
                margin: 20px 35px; /* Margen ajustado */
                padding: 15px;
                background-color: {QColor(self.color_central_bg).lighter(105).name()}; /* Era rgba(52, 73, 94, 0.3) */
                border-radius: 8px;
            }}
            QLabel {{ color: {self.color_text_light}; }}
            QTextEdit {{ background-color: {QColor(self.color_central_bg).lighter(110).name()}; color: {self.color_text_light}; border: 1px solid {QColor(self.color_window_border).lighter(150).name()}; border-radius: 4px; padding: 5px; }}
            QProgressBar {{ border: 1px solid {QColor(self.color_window_border).lighter(150).name()}; border-radius: 4px; text-align: center; color: {self.color_text_light}; background-color: {QColor(self.color_central_bg).lighter(110).name()}; }}
            QProgressBar::chunk {{ background-color: {self.color_accent_red}; border-radius: 3px; width:10px; margin:0.5px;}}

            QLabel#notificationIcon {{ color: {self.color_accent_red}; font-size: 15px; font-weight: bold; padding-right: 10px; }}
            QLabel#notificationText {{ color: {self.color_text_medium}; font-size: 12.5px; }}
            QLabel#notificationLink a, QLabel#licenseLink a {{ color: {self.color_accent_blue_link}; text-decoration: none; font-size: 12.5px; }}
            QLabel#notificationLink a:hover, QLabel#licenseLink a:hover {{ text-decoration: underline; }}
            QLabel#licenseLink {{ color: {self.color_text_medium}; padding-bottom: 5px; padding-right: 5px; }}

            QToolBar {{ background-color: {self.color_sidebar_bg}; border-left: 1px solid {self.color_window_border}; padding: 5px; }}
            QToolBar QToolButton {{ padding: 7px; margin: 1px 0px; border-radius: 4px; background-color: transparent; color: {self.color_text_light}; border: 1px solid transparent; }}
            QToolBar QToolButton:hover {{ background-color: {self.color_sidebar_button_hover_celeste}; border: 1px solid {self.color_sidebar_button_hover_celeste}; }}
            
            QFrame#ToolFrame {{ background-color: {QColor(self.color_central_bg).lighter(110).name()}; border-radius: 6px; padding: 15px; }}
            QPushButton#ToolActionButton {{ background-color: {self.color_accent_blue_link}; color: white; border-radius: 4px; font-size: 11pt; font-weight: bold; padding: 8px 12px; }}
            QPushButton#ToolActionButton:hover {{ background-color: {QColor(self.color_accent_blue_link).lighter(120).name()}; }}
            QPushButton#ToolActionButton:pressed {{ background-color: {QColor(self.color_accent_blue_link).darker(120).name()}; }}

            /* Estilos para TrainingOptionCard y sus componentes */
            QFrame#TrainingOptionCard {{
                background-color: {QColor(self.color_central_bg).lighter(120).name()}; /* Ajustado para ser un poco más claro */
                border-radius: 15px;
                border: 2px solid {QColor(self.color_central_bg).lighter(135).name()}; /* Borde ajustado para mejor contraste */
                padding: 20px;
            }}
            QFrame#TrainingOptionCard:hover {{
                border: 2px solid {self.color_accent_blue_link}; 
                background-color: {QColor(self.color_central_bg).lighter(130).name()}; /* Ajustado para ser más claro que el fondo normal y el borde normal */
            }}
            QLabel#CardModelIndicator {{ /* Estilo para el indicador de imagen/texto si no carga imagen */
                background-color: {QColor(self.color_central_bg).lighter(115).name()};
                color: {self.color_text_light};
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
                padding: 5px;
                min-width: 60px; /* Para asegurar que el texto no se corte */
                min-height: 40px;
            }}
            QPushButton#CardMainButton {{
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 13pt; /* Era 15px */
                font-weight: bold;
                padding: 10px 20px;
            }}
            QPushButton#CardMainButton[optionType="ml"] {{
                background-color: {self.color_accent_blue_link}; /* Era #3498DB */
            }}
            QPushButton#CardMainButton[optionType="dl"] {{
                background-color: {self.color_accent_red}; /* Era #E74C3C */
            }}
            QPushButton#CardMainButton[optionType="ml"]:hover {{
                background-color: {QColor(self.color_accent_blue_link).lighter(120).name()};
            }}
            QPushButton#CardMainButton[optionType="dl"]:hover {{
                background-color: {QColor(self.color_accent_red).lighter(120).name()};
            }}
            QPushButton#CardMainButton[optionType="ml"]:pressed {{
                background-color: {QColor(self.color_accent_blue_link).darker(120).name()};
            }}
            QPushButton#CardMainButton[optionType="dl"]:pressed {{
                background-color: {QColor(self.color_accent_red).darker(120).name()};
            }}
            QPushButton#CardConfigButton {{
                background-color: transparent;
                color: {self.color_text_medium}; /* Era #7F8C8D */
                border: 1px solid {self.color_text_medium}; /* Era #7F8C8D */
                border-radius: 18px;
                font-size: 11pt; /* Era 13px */
                padding: 8px 18px;
            }}
            QPushButton#CardConfigButton:hover {{
                color: {self.color_text_light}; /* Era #BDC3C7 */
                border: 1px solid {self.color_text_light}; /* Era #BDC3C7 */
            }}
            QLabel#CardDescriptionLabel {{
                color: {QColor(self.color_text_medium).lighter(110).name()}; /* Era #95A5A6 */
                font-size: 10pt; /* Era 12px */
                margin: 10px 0px;
                padding: 8px;
            }}
            /* Fin de estilos para TrainingOptionCard */

            QFrame[objectName^="NotifFrame_"] {{ border-radius: 5px; padding: 12px; }}
            QFrame#NotifFrame_critical {{ background-color: {color_notif_critical_bg}; border: 1px solid {color_notif_critical_border}; }}
            QFrame#NotifFrame_critical QLabel[objectName!="NotifTextLabel"] {{ color: {color_notif_critical_icon}; }} 
            QFrame#NotifFrame_critical QLabel#NotifTextLabel {{ color: {self.color_text_light}; }} 
            QFrame#NotifFrame_warning {{ background-color: {color_notif_warning_bg}; border: 1px solid {color_notif_warning_border}; }}
            QFrame#NotifFrame_warning QLabel[objectName!="NotifTextLabel"] {{ color: {color_notif_warning_icon}; }}
            QFrame#NotifFrame_warning QLabel#NotifTextLabel {{ color: {self.color_text_light}; }}
            QFrame#NotifFrame_info {{ background-color: {color_notif_info_bg}; border: 1px solid {color_notif_info_border}; }}
            QFrame#NotifFrame_info QLabel[objectName!="NotifTextLabel"] {{ color: {color_notif_info_icon}; }}
            QFrame#NotifFrame_info QLabel#NotifTextLabel {{ color: {self.color_text_light}; }}
            QFrame#NotifFrame_success {{ background-color: {color_notif_success_bg}; border: 1px solid {color_notif_success_border}; }}
            QFrame#NotifFrame_success QLabel[objectName!="NotifTextLabel"] {{ color: {color_notif_success_icon}; }}
            QFrame#NotifFrame_success QLabel#NotifTextLabel {{ color: {self.color_text_light}; }}

            QPushButton[objectName^="NotifActionButton_"] {{ background-color: {color_notif_action_button_bg}; color: white; border-radius: 3px; font-size: 10pt; padding: 5px 10px; }}
            QPushButton[objectName^="NotifActionButton_"]:hover {{ background-color: {color_notif_action_button_hover_bg}; }}
            QPushButton#NotifActionButton_critical {{ background-color: {self.color_accent_red}; }}
            QPushButton#NotifActionButton_critical:hover {{ background-color: {QColor(self.color_accent_red).lighter(110).name()}; }}
        """)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName("ClasificaTalento PRO") 
    app.setOrganizationName("TalentHunter") 
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())