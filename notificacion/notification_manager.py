# notification_manager.py
import sys
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QGraphicsOpacityEffect,
                             QApplication, QMainWindow, QGridLayout, QSizePolicy) # <- CORREGIDO
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt6.QtGui import QFont
from enum import Enum


class NotificationType(Enum):
    """Tipos de notificación disponibles"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    QUESTION = "question"


class NotificationWidget(QFrame):
    """Widget de notificación individual con diseño moderno"""
    
    closed = pyqtSignal()
    action_clicked = pyqtSignal(str)
    
    def __init__(self, title, message, notification_type=NotificationType.INFO, 
                 duration=5000, actions=None, parent=None):
        super().__init__(parent)
        self.notification_type = notification_type
        self.duration = duration
        self.actions = actions or []
        
        # Ajuste para un aspecto más cuadrado
        self.setFixedHeight(110)
        self.setMinimumWidth(360)
        self.setMaximumWidth(450)
        
        # Configurar estilo base
        self.setup_style()
        
        # Configurar UI
        self.setup_ui(title, message)
        
        # Configurar animaciones
        self.setup_animations()
        
        # Auto-cerrar si tiene duración
        if self.duration > 0:
            QTimer.singleShot(self.duration, self.close_notification)
    
    def setup_style(self):
        """Configura el nuevo estilo limpio y cuadrado"""
        styles = {
            NotificationType.SUCCESS: {
                'indicator': '#2ECC71', # Verde
                'icon': '✔'
            },
            NotificationType.ERROR: {
                'indicator': '#E74C3C', # Rojo
                'icon': '✖'
            },
            NotificationType.WARNING: {
                'indicator': '#F1C40F', # Amarillo
                'icon': '!'
            },
            NotificationType.INFO: {
                'indicator': '#3498DB', # Azul
                'icon': 'i'
            },
            NotificationType.QUESTION: {
                'indicator': '#9B59B6', # Morado
                'icon': '?'
            }
        }
        
        style = styles.get(self.notification_type, styles[NotificationType.INFO])
        indicator_color = style['indicator']
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #2C3E50; /* Fondo oscuro principal */
                border-radius: 8px;
                /* Borde izquierdo de color que indica el tipo */
                border-left: 6px solid {indicator_color}; 
            }}
            QLabel#titleLabel {{
                color: #ECF0F1; /* Color de texto claro */
                font-size: 11pt;
                font-weight: bold;
                background: transparent;
                border: none;
            }}
            QLabel#messageLabel {{
                color: #BDC3C7; /* Color de texto más suave */
                font-size: 9pt;
                background: transparent;
                border: none;
            }}
            /* Botones de acción (Confirmar, Cancelar, etc.) */
            QPushButton#actionButton {{
                background-color: #34495E;
                color: #ECF0F1;
                border: none;
                border-radius: 4px;
                padding: 8px 14px;
                font-size: 8pt;
                font-weight: bold;
            }}
            QPushButton#actionButton:hover {{
                background-color: #4E657E;
            }}
            QPushButton#actionButton:pressed {{
                background-color: #2C3E50;
            }}
            /* Botón de cerrar (X) */
            QPushButton#closeButton {{
                background-color: transparent;
                color: #7F8C8D;
                border: none;
                border-radius: 12px;
                font-size: 14pt;
                font-weight: bold;
            }}
            QPushButton#closeButton:hover {{
                background-color: #34495E;
                color: #BDC3C7;
            }}
        """)
        
        self.style_data = style
    
    def setup_ui(self, title, message):
        """Configura la interfaz de usuario con el nuevo diseño"""
        layout = QHBoxLayout(self)
        # Margen izquierdo 0 para que el borde de color se pegue al borde
        layout.setContentsMargins(0, 10, 10, 10) 
        layout.setSpacing(15)
        
        # Icono
        icon_label = QLabel(self.style_data['icon'])
        icon_label.setFixedSize(40, 40)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet(f"""
            background-color: #34495E;
            color: #ECF0F1;
            font-size: 18pt;
            font-weight: bold;
            border-radius: 20px; /* Círculo perfecto */
        """)
        
        # Contenido (Título y Mensaje)
        content_layout = QVBoxLayout()
        content_layout.setSpacing(2)
        
        title_label = QLabel(title)
        title_label.setObjectName("titleLabel") # ID para QSS
        title_label.setWordWrap(True)
        
        message_label = QLabel(message)
        message_label.setObjectName("messageLabel") # ID para QSS
        message_label.setWordWrap(True)
        # Permite que el texto se ajuste verticalmente si crece
        message_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # <- CORREGIDO
        
        content_layout.addWidget(title_label)
        content_layout.addWidget(message_label)
        content_layout.addStretch()
        
        # Botones (Cerrar y Acciones)
        actions_column_layout = QVBoxLayout()
        actions_column_layout.setSpacing(8)

        # Botón cerrar en la esquina superior derecha
        top_bar_layout = QHBoxLayout()
        top_bar_layout.addStretch()
        close_btn = QPushButton("×")
        close_btn.setObjectName("closeButton") # ID para QSS
        close_btn.setFixedSize(24, 24)
        close_btn.clicked.connect(self.close_notification)
        top_bar_layout.addWidget(close_btn)
        actions_column_layout.addLayout(top_bar_layout)
        
        actions_column_layout.addStretch()

        # Botones de acción personalizados en la parte inferior
        if self.actions:
            actions_row_layout = QHBoxLayout()
            actions_row_layout.setSpacing(8)
            actions_row_layout.addStretch() # Alinea los botones a la derecha
            for action_text, action_id in self.actions:
                action_btn = QPushButton(action_text)
                action_btn.setObjectName("actionButton") # ID para QSS
                action_btn.clicked.connect(lambda _, aid=action_id: self.on_action_clicked(aid))
                actions_row_layout.addWidget(action_btn)
            actions_column_layout.addLayout(actions_row_layout)
        
        # Ensamblaje final
        layout.addSpacing(15) # Espacio entre el borde y el icono
        layout.addWidget(icon_label)
        layout.addLayout(content_layout, 1) # El '1' le da prioridad para expandirse
        layout.addLayout(actions_column_layout)

    def setup_animations(self):
        """Configura las animaciones de entrada y salida"""
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        
        self.fade_in_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_in_animation.setDuration(300)
        self.fade_in_animation.setStartValue(0.0)
        self.fade_in_animation.setEndValue(1.0)
        self.fade_in_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        self.fade_out_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_out_animation.setDuration(300)
        self.fade_out_animation.setStartValue(1.0)
        self.fade_out_animation.setEndValue(0.0)
        self.fade_out_animation.setEasingCurve(QEasingCurve.Type.InCubic)
        self.fade_out_animation.finished.connect(self.on_fade_out_finished)
        
        self.fade_in_animation.start()
    
    def on_action_clicked(self, action_id):
        """Maneja el clic en botones de acción"""
        self.action_clicked.emit(action_id)
        self.close_notification()
    
    def close_notification(self):
        """Cierra la notificación con animación"""
        self.fade_out_animation.start()
    
    def on_fade_out_finished(self):
        """Se ejecuta cuando termina la animación de salida"""
        self.closed.emit()
        self.deleteLater()


class NotificationManager(QWidget):
    """Gestor de notificaciones que se superpone a la ventana principal"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.raise_()
        self.setWindowOpacity(1.0)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(10)
        self.layout.addStretch()
        
        self.active_notifications = []
        self.main_window_ref = None

        if parent:
            self.set_main_window(parent.window())
        
    def set_main_window(self, main_window):
        self.main_window_ref = main_window
        if self.main_window_ref:
            self.main_window_ref.moveEvent = self.update_position
            self.main_window_ref.resizeEvent = self.update_position
            self.update_position()

    def update_position(self, event=None):
        if self.main_window_ref and self.main_window_ref.isVisible():
            window_rect = self.main_window_ref.geometry()
            self.setGeometry(window_rect)
            self.raise_()
        else:
            screen = QApplication.primaryScreen().geometry()
            self.setGeometry(screen)

    def show_notification(self, title, message, notification_type=NotificationType.INFO,
                          duration=5000, actions=None):
        self.update_position()
        notification = NotificationWidget(
            title, message, notification_type, duration, actions, self
        )
        notification.closed.connect(lambda: self.remove_notification(notification))
        notification.action_clicked.connect(self.on_notification_action)
        self.layout.insertWidget(self.layout.count() - 1, notification, 0,
                                 Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        self.active_notifications.append(notification)
        if not self.isVisible():
            self.show()
        self.raise_()
        return notification
    
    def remove_notification(self, notification):
        if notification in self.active_notifications:
            self.active_notifications.remove(notification)
        if not self.active_notifications:
            self.hide()
    
    def on_notification_action(self, action_id):
        print(f"Acción '{action_id}' seleccionada.")
    
    def clear_all(self):
        for notification in self.active_notifications.copy():
            notification.close_notification()


_global_manager = None

def get_notification_manager(parent=None):
    global _global_manager
    if _global_manager is None:
        app_instance = QApplication.instance()
        if not app_instance:
            app_instance = QApplication(sys.argv)
        _global_manager = NotificationManager(None)

    if parent:
        main_window = parent.window()
        if _global_manager.main_window_ref != main_window:
            _global_manager.set_main_window(main_window)
    return _global_manager

def show_success(title, message, duration=4000, actions=None, parent=None):
    manager = get_notification_manager(parent)
    return manager.show_notification(title, message, NotificationType.SUCCESS, duration, actions)

def show_error(title, message, duration=6000, actions=None, parent=None):
    manager = get_notification_manager(parent)
    return manager.show_notification(title, message, NotificationType.ERROR, duration, actions)

def show_warning(title, message, duration=5000, actions=None, parent=None):
    manager = get_notification_manager(parent)
    return manager.show_notification(title, message, NotificationType.WARNING, duration, actions)

def show_info(title, message, duration=4000, actions=None, parent=None):
    manager = get_notification_manager(parent)
    return manager.show_notification(title, message, NotificationType.INFO, duration, actions)

def show_question(title, message, actions=None, parent=None):
    manager = get_notification_manager(parent)
    return manager.show_notification(title, message, NotificationType.QUESTION, 0, actions)


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Demo del Gestor de Notificaciones")
        self.setGeometry(100, 100, 500, 400)
        self.setStyleSheet("background-color: #F0F2F5;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20,20,20,20)

        button_style = """
            QPushButton {
                font-size: 11pt;
                padding: 12px;
                border-radius: 8px;
                background-color: #FFFFFF;
                border: 1px solid #D9D9D9;
                color: #333;
            }
            QPushButton:hover {
                background-color: #F8F9FA;
                border-color: #3498DB;
            }
            QPushButton:pressed {
                background-color: #E9ECEF;
            }
        """

        btn_success = QPushButton("Éxito (Success)")
        btn_success.setStyleSheet(button_style)
        btn_success.clicked.connect(self.show_success_notification)
        layout.addWidget(btn_success, 0, 0)

        btn_error = QPushButton("Error")
        btn_error.setStyleSheet(button_style)
        btn_error.clicked.connect(self.show_error_notification)
        layout.addWidget(btn_error, 0, 1)

        btn_warning = QPushButton("Advertencia (Warning)")
        btn_warning.setStyleSheet(button_style)
        btn_warning.clicked.connect(self.show_warning_notification)
        layout.addWidget(btn_warning, 1, 0)

        btn_info = QPushButton("Información (Info)")
        btn_info.setStyleSheet(button_style)
        btn_info.clicked.connect(self.show_info_notification)
        layout.addWidget(btn_info, 1, 1)

        btn_question = QPushButton("Pregunta (Question)")
        btn_question.setStyleSheet(button_style)
        btn_question.clicked.connect(self.show_question_notification)
        layout.addWidget(btn_question, 2, 0, 1, 2)
        
        self.result_label = QLabel("La respuesta a la pregunta aparecerá aquí.")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 11pt; color: #555; margin-top: 10px;")
        layout.addWidget(self.result_label, 3, 0, 1, 2)

    def show_success_notification(self):
        show_success("Éxito", "La operación se completó correctamente.", parent=self)

    def show_error_notification(self):
        show_error("Error Crítico", "No se pudo conectar con el servidor. Por favor, revise su conexión a internet.", parent=self)

    def show_warning_notification(self):
        show_warning("Advertencia", "El espacio en disco es bajo.", parent=self)

    def show_info_notification(self):
        show_info("Información", "Hay una nueva actualización disponible para la aplicación.", parent=self)

    def show_question_notification(self):
        actions = [("Confirmar", "confirm"), ("Cancelar", "cancel")]
        notification = show_question("Confirmación", "¿Está seguro que desea eliminar este elemento?", 
                                     actions=actions, parent=self)
        if notification:
            notification.action_clicked.connect(self.handle_question_action)

    def handle_question_action(self, action_id):
        if action_id == "confirm":
            self.result_label.setText("Respuesta: ¡Elemento eliminado!")
            self.result_label.setStyleSheet("color: #E74C3C; font-weight: bold;")
        elif action_id == "cancel":
            self.result_label.setText("Respuesta: Operación cancelada.")
            self.result_label.setStyleSheet("color: #3498DB; font-weight: bold;")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())