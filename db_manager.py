import sqlite3
import os
from datetime import datetime
import logging
from src.config import logging_config  # noqa: F401
from src.config.settings import Settings

logger = logging.getLogger(__name__)

DATABASE_NAME = "postulantes.db"
# Ubicación de la base de datos. Puede sobreescribirse con la variable de entorno
# ``DB_PATH``. Por defecto se crea en la carpeta raíz del proyecto definida en
# ``Settings.BASE_DIR``.
DATABASE_PATH = os.getenv(
    "DB_PATH",
    str(Settings.BASE_DIR / DATABASE_NAME)
)

def init_db():
    """Inicializa la base de datos y crea la tabla de aplicaciones si no existe."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS aplicaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            telefono TEXT NOT NULL,
            correo TEXT NOT NULL UNIQUE, 
            nombre_cv TEXT NOT NULL,
            tipo_cv TEXT,
            cv_base64 TEXT NOT NULL,
            fecha_aplicacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        # Se añadió UNIQUE al correo para evitar duplicados de postulaciones por el mismo email.
        # Considerar si esto es deseable o si una persona puede postular varias veces.
        # Por ahora, un correo único por postulación.
        
        conn.commit()
        logger.info("Base de datos inicializada correctamente en %s", DATABASE_PATH)
    except sqlite3.Error as e:
        logger.error("Error al inicializar la base de datos: %s", e)
    finally:
        if conn:
            conn.close()

def add_application(nombre, telefono, correo, nombre_cv, tipo_cv, cv_base64):
    """
    Añade una nueva aplicación a la base de datos.
    Devuelve un mensaje de éxito o error.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO aplicaciones (nombre, telefono, correo, nombre_cv, tipo_cv, cv_base64, fecha_aplicacion)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (nombre, telefono, correo, nombre_cv, tipo_cv, cv_base64, datetime.now()))
        
        conn.commit()
        logger.info("Aplicación de %s (%s) guardada correctamente", nombre, correo)
        return "Postulación enviada con éxito."
    except sqlite3.IntegrityError:
        # Esto ocurrirá si el correo ya existe, debido a la restricción UNIQUE
        logger.warning("El correo %s ya ha sido registrado", correo)
        return f"Error: El correo electrónico '{correo}' ya ha sido registrado con una postulación."
    except sqlite3.Error as e:
        logger.error("Error al guardar la aplicación para %s: %s", correo, e)
        return f"Error al procesar la postulación: {e}"
    finally:
        if conn:
            conn.close()

def get_all_applications():
    """Recupera todas las aplicaciones de la base de datos."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, nombre, telefono, correo, nombre_cv, tipo_cv, fecha_aplicacion FROM aplicaciones ORDER BY fecha_aplicacion DESC")
        applications = cursor.fetchall()
        return applications
    except sqlite3.Error as e:
        logger.error("Error al obtener aplicaciones: %s", e)
        return []
    finally:
        if conn:
            conn.close()

def get_application_cv_by_id(app_id):
    """Recupera el nombre_cv y cv_base64 de una aplicación específica por su ID."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT nombre_cv, cv_base64, tipo_cv FROM aplicaciones WHERE id = ?", (app_id,))
        result = cursor.fetchone()
        return result # (nombre_cv, cv_base64, tipo_cv)
    except sqlite3.Error as e:
        logger.error("Error al obtener CV para aplicación ID %s: %s", app_id, e)
        return None
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # Esto se ejecutará si el script se corre directamente.
    # Útil para crear la base de datos por primera vez.
    logger.info("Intentando inicializar la base de datos en %s", DATABASE_PATH)
    init_db()
    
    # Ejemplo de cómo añadir una aplicación (descomentar para probar)
