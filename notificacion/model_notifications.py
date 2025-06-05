# model_notifications.py
from .notification_manager import (show_success, show_error, show_warning, 
                                  show_info, show_question, NotificationType)


class ModelNotifications:
    """Notificaciones específicas para operaciones de modelos"""
    
    @staticmethod
    def model_loaded_success(model_name, parent=None):
        """Notificación de modelo cargado exitosamente"""
        return show_success(
            "Modelo Cargado",
            f"El modelo '{model_name}' se ha cargado correctamente y está listo para usar.",
            duration=3000,
            parent=parent
        )
    
    @staticmethod
    def model_load_error(model_name, error_msg, parent=None):
        """Notificación de error al cargar modelo"""
        return show_error(
            "Error al Cargar Modelo",
            f"No se pudo cargar '{model_name}': {error_msg}",
            duration=6000,
            parent=parent
        )
    
    @staticmethod
    def model_exported_success(model_name, file_path, parent=None):
        """Notificación de modelo exportado exitosamente"""
        return show_success(
            "Modelo Exportado",
            f"'{model_name}' se ha exportado correctamente a {file_path}",
            duration=4000,
            actions=[("Abrir Carpeta", "open_folder")],
            parent=parent
        )
    
    @staticmethod
    def model_export_error(model_name, error_msg, parent=None):
        """Notificación de error al exportar modelo"""
        return show_error(
            "Error de Exportación",
            f"No se pudo exportar '{model_name}': {error_msg}",
            duration=6000,
            parent=parent
        )
    
    @staticmethod
    def model_imported_success(model_name, parent=None):
        """Notificación de modelo importado exitosamente"""
        return show_success(
            "Modelo Importado",
            f"El modelo '{model_name}' se ha importado correctamente y está disponible.",
            duration=4000,
            parent=parent
        )
    
    @staticmethod
    def model_import_error(error_msg, parent=None):
        """Notificación de error al importar modelo"""
        return show_error(
            "Error de Importación",
            f"No se pudo importar el modelo: {error_msg}",
            duration=6000,
            parent=parent
        )
    
    @staticmethod
    def confirm_model_deletion(model_name, parent=None):
        """Notificación de confirmación para eliminar modelo"""
        return show_question(
            "Confirmar Eliminación",
            f"¿Estás seguro de que quieres eliminar '{model_name}'? Esta acción no se puede deshacer.",
            actions=[
                ("Eliminar", "confirm_delete"),
                ("Cancelar", "cancel_delete")
            ],
            parent=parent
        )
    
    @staticmethod
    def model_deleted_success(model_name, parent=None):
        """Notificación de modelo eliminado exitosamente"""
        return show_success(
            "Modelo Eliminado",
            f"El modelo '{model_name}' ha sido eliminado correctamente.",
            duration=3000,
            parent=parent
        )
    
    @staticmethod
    def model_delete_error(model_name, error_msg, parent=None):
        """Notificación de error al eliminar modelo"""
        return show_error(
            "Error al Eliminar",
            f"No se pudo eliminar '{model_name}': {error_msg}",
            duration=6000,
            parent=parent
        )
    
    @staticmethod
    def model_backup_success(count, location, parent=None):
        """Notificación de backup exitoso"""
        return show_success(
            "Backup Completado",
            f"Se han respaldado {count} modelos en {location}",
            duration=4000,
            actions=[("Ver Carpeta", "open_backup_folder")],
            parent=parent
        )
    
    @staticmethod
    def model_backup_error(error_msg, parent=None):
        """Notificación de error en backup"""
        return show_error(
            "Error en Backup",
            f"No se pudo completar el backup: {error_msg}",
            duration=6000,
            parent=parent
        )
    
    @staticmethod
    def no_models_available(parent=None):
        """Notificación cuando no hay modelos disponibles"""
        return show_info(
            "Sin Modelos",
            "No hay modelos disponibles. Entrena un modelo primero o importa uno existente.",
            duration=4000,
            actions=[("Ir a Entrenamiento", "go_training")],
            parent=parent
        )
    
    @staticmethod
    def model_validation_warning(model_name, issues, parent=None):
        """Notificación de advertencia sobre validación de modelo"""
        return show_warning(
            "Advertencia de Modelo",
            f"'{model_name}' tiene problemas de validación: {issues}",
            duration=5000,
            parent=parent
        )
    
    @staticmethod
    def model_update_available(model_name, parent=None):
        """Notificación de actualización disponible para modelo"""
        return show_info(
            "Actualización Disponible",
            f"Hay una nueva versión disponible para '{model_name}'",
            duration=5000,
            actions=[("Actualizar", "update_model"), ("Más Tarde", "dismiss")],
            parent=parent
        )
    
    @staticmethod
    def model_training_complete(model_name, accuracy, parent=None):
        """Notificación de entrenamiento completado"""
        return show_success(
            "Entrenamiento Completado",
            f"'{model_name}' se ha entrenado exitosamente con {accuracy:.1%} de precisión.",
            duration=5000,
            actions=[("Ver Detalles", "view_details")],
            parent=parent
        )
    
    @staticmethod
    def model_training_failed(model_name, error_msg, parent=None):
        """Notificación de fallo en entrenamiento"""
        return show_error(
            "Entrenamiento Fallido",
            f"El entrenamiento de '{model_name}' falló: {error_msg}",
            duration=6000,
            actions=[("Ver Log", "view_log"), ("Reintentar", "retry_training")],
            parent=parent
        )
    
    @staticmethod
    def model_prediction_ready(model_name, parent=None):
        """Notificación de modelo listo para predicción"""
        return show_info(
            "Modelo Listo",
            f"'{model_name}' está cargado y listo para clasificar CVs.",
            duration=3000,
            actions=[("Ir a Clasificar", "go_classify")],
            parent=parent
        )
    
    @staticmethod
    def invalid_model_format(file_name, parent=None):
        """Notificación de formato de modelo inválido"""
        return show_warning(
            "Formato Inválido",
            f"'{file_name}' no es un archivo .senati válido o está corrupto.",
            duration=5000,
            parent=parent
        )
    
    @staticmethod
    def model_compatibility_warning(model_name, version, parent=None):
        """Notificación de advertencia de compatibilidad"""
        return show_warning(
            "Compatibilidad",
            f"'{model_name}' fue creado con una versión diferente ({version}). Puede haber problemas de compatibilidad.",
            duration=6000,
            actions=[("Continuar", "continue_anyway"), ("Cancelar", "cancel_load")],
            parent=parent
        )
    
    @staticmethod
    def storage_space_warning(space_needed, space_available, parent=None):
        """Notificación de advertencia de espacio en disco"""
        return show_warning(
            "Espacio Insuficiente",
            f"Se necesitan {space_needed} MB pero solo hay {space_available} MB disponibles.",
            duration=6000,
            actions=[("Liberar Espacio", "clean_space"), ("Continuar", "continue_anyway")],
            parent=parent
        )


# Funciones de conveniencia para uso directo
def notify_model_loaded(model_name, parent=None):
    return ModelNotifications.model_loaded_success(model_name, parent)

def notify_model_error(model_name, error_msg, parent=None):
    return ModelNotifications.model_load_error(model_name, error_msg, parent)

def notify_model_exported(model_name, file_path, parent=None):
    return ModelNotifications.model_exported_success(model_name, file_path, parent)

def notify_model_imported(model_name, parent=None):
    return ModelNotifications.model_imported_success(model_name, parent)

def confirm_delete_model(model_name, parent=None):
    return ModelNotifications.confirm_model_deletion(model_name, parent)

def notify_model_deleted(model_name, parent=None):
    return ModelNotifications.model_deleted_success(model_name, parent)
