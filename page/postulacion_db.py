import sqlite3
import os
from datetime import datetime

# Configuración de la base de datos
DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
DATABASE_NAME = "postulaciones.db"
DATABASE_PATH = os.path.join(DATABASE_DIR, DATABASE_NAME)

def ensure_database_directory():
    """Asegura que el directorio de la base de datos exista."""
    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)
        print(f"Directorio de base de datos creado: {DATABASE_DIR}")

def init_database():
    """Inicializa la base de datos y crea las tablas necesarias."""
    try:
        ensure_database_directory()
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Crear tabla de postulaciones
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS postulaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            dni TEXT NOT NULL UNIQUE,
            telefono TEXT NOT NULL,
            correo TEXT NOT NULL UNIQUE,
            cv_filename TEXT NOT NULL,
            cv_data BLOB NOT NULL,
            cv_size INTEGER NOT NULL,
            fecha_postulacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            estado TEXT DEFAULT 'pendiente'
        )
        """)
        
        # Crear índices para mejorar el rendimiento
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dni ON postulaciones(dni)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_correo ON postulaciones(correo)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fecha ON postulaciones(fecha_postulacion)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_estado ON postulaciones(estado)")
        
        conn.commit()
        print(f"Base de datos inicializada correctamente en: {DATABASE_PATH}")
        return True
        
    except sqlite3.Error as e:
        print(f"Error al inicializar la base de datos: {e}")
        return False
    finally:
        if conn:
            conn.close()

def add_postulacion(nombre, dni, telefono, correo, cv_filename, cv_data):
    """
    Añade una nueva postulación a la base de datos.

    Args:
        nombre (str): Nombre completo del postulante
        dni (str): Documento Nacional de Identidad
        telefono (str): Número de teléfono
        correo (str): Correo electrónico
        cv_filename (str): Nombre del archivo CV
        cv_data (bytes): Datos binarios del archivo CV

    Returns:
        dict: Resultado de la operación con 'success' y 'message'
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cv_size = len(cv_data)
        
        cursor.execute("""
        INSERT INTO postulaciones (nombre, dni, telefono, correo, cv_filename, cv_data, cv_size, fecha_postulacion)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (nombre, dni, telefono, correo, cv_filename, cv_data, cv_size, datetime.now()))
        
        conn.commit()
        postulacion_id = cursor.lastrowid
        
        print(f"Postulación de {nombre} ({correo}) guardada con ID: {postulacion_id}")
        return {
            'success': True,
            'message': 'Postulación enviada exitosamente.',
            'id': postulacion_id
        }
        
    except sqlite3.IntegrityError:
        print(f"Error: El correo {correo} ya está registrado.")
        return {
            'success': False,
            'message': f'El correo electrónico "{correo}" ya está registrado con una postulación anterior.'
        }
    except sqlite3.Error as e:
        print(f"Error al guardar la postulación: {e}")
        return {
            'success': False,
            'message': f'Error al procesar la postulación: {str(e)}'
        }
    finally:
        if conn:
            conn.close()

def get_all_postulaciones():
    """
    Recupera todas las postulaciones de la base de datos.
    
    Returns:
        list: Lista de tuplas con los datos de las postulaciones
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT id, nombre, dni, telefono, correo, cv_filename, cv_size,
               fecha_postulacion, estado, puesto_clasificacion, porcentaje_clasificacion, modelo_clasificacion
        FROM postulaciones
        ORDER BY fecha_postulacion DESC
        """)
        
        postulaciones = cursor.fetchall()
        return postulaciones
        
    except sqlite3.Error as e:
        print(f"Error al obtener postulaciones: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_postulacion_by_id(postulacion_id):
    """
    Recupera una postulación específica por su ID.
    
    Args:
        postulacion_id (int): ID de la postulación
    
    Returns:
        tuple: Datos de la postulación o None si no existe
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT id, nombre, dni, telefono, correo, cv_filename, cv_data, cv_size,
               fecha_postulacion, estado
        FROM postulaciones
        WHERE id = ?
        """, (postulacion_id,))
        
        result = cursor.fetchone()
        return result
        
    except sqlite3.Error as e:
        print(f"Error al obtener postulación ID {postulacion_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def update_postulacion_estado(postulacion_id, nuevo_estado):
    """
    Actualiza el estado de una postulación.
    
    Args:
        postulacion_id (int): ID de la postulación
        nuevo_estado (str): Nuevo estado ('pendiente', 'revisado', 'aceptado', 'rechazado')
    
    Returns:
        bool: True si se actualizó correctamente, False en caso contrario
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
        UPDATE postulaciones 
        SET estado = ? 
        WHERE id = ?
        """, (nuevo_estado, postulacion_id))
        
        conn.commit()
        
        if cursor.rowcount > 0:
            print(f"Estado de postulación ID {postulacion_id} actualizado a: {nuevo_estado}")
            return True
        else:
            print(f"No se encontró postulación con ID: {postulacion_id}")
            return False
            
    except sqlite3.Error as e:
        print(f"Error al actualizar estado de postulación: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_cv_data(postulacion_id):
    """
    Recupera los datos del CV de una postulación específica.
    
    Args:
        postulacion_id (int): ID de la postulación
    
    Returns:
        tuple: (cv_filename, cv_data) o None si no existe
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT cv_filename, cv_data 
        FROM postulaciones 
        WHERE id = ?
        """, (postulacion_id,))
        
        result = cursor.fetchone()
        return result
        
    except sqlite3.Error as e:
        print(f"Error al obtener CV para postulación ID {postulacion_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_database_stats():
    """
    Obtiene estadísticas de la base de datos.
    
    Returns:
        dict: Estadísticas de la base de datos
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Total de postulaciones
        cursor.execute("SELECT COUNT(*) FROM postulaciones")
        total = cursor.fetchone()[0]
        
        # Postulaciones por estado
        cursor.execute("""
        SELECT estado, COUNT(*) 
        FROM postulaciones 
        GROUP BY estado
        """)
        por_estado = dict(cursor.fetchall())
        
        # Postulaciones del último mes
        cursor.execute("""
        SELECT COUNT(*) 
        FROM postulaciones 
        WHERE fecha_postulacion >= datetime('now', '-30 days')
        """)
        ultimo_mes = cursor.fetchone()[0]
        
        return {
            'total': total,
            'por_estado': por_estado,
            'ultimo_mes': ultimo_mes
        }
        
    except sqlite3.Error as e:
        print(f"Error al obtener estadísticas: {e}")
        return {
            'total': 0,
            'por_estado': {},
            'ultimo_mes': 0
        }
    finally:
        if conn:
            conn.close()

def delete_postulacion_from_db(postulacion_id):
    """
    Elimina una postulación de la base de datos por su ID.

    Args:
        postulacion_id (int): ID de la postulación a eliminar.

    Returns:
        bool: True si la eliminación fue exitosa (se eliminó una fila), False en caso contrario.
    """
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM postulaciones WHERE id = ?", (postulacion_id,))
        conn.commit()
        
        if cursor.rowcount > 0:
            print(f"Postulación con ID {postulacion_id} eliminada correctamente.")
            return True
        else:
            print(f"No se encontró postulación con ID {postulacion_id} para eliminar.")
            return False
            
    except sqlite3.Error as e:
        print(f"Error al eliminar postulación ID {postulacion_id}: {e}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # Inicializar la base de datos si se ejecuta directamente
    print("Inicializando base de datos de postulaciones...")
    if init_database():
        print("Base de datos inicializada correctamente.")
        
        # Mostrar estadísticas
        stats = get_database_stats()
        print(f"Estadísticas actuales:")
        print(f"- Total de postulaciones: {stats['total']}")
        print(f"- Postulaciones por estado: {stats['por_estado']}")
        print(f"- Postulaciones último mes: {stats['ultimo_mes']}")
    else:
        print("Error al inicializar la base de datos.")
