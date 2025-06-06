import sqlite3
import os

DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
DATABASE_NAME = "postulaciones.db"
DATABASE_PATH = os.path.join(DATABASE_DIR, DATABASE_NAME)

def add_classification_columns():
    """Agrega columnas para clasificación automática si no existen"""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Verificar si la columna 'puesto_clasificacion' existe
        cursor.execute("PRAGMA table_info(postulaciones)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'puesto_clasificacion' not in columns:
            cursor.execute("ALTER TABLE postulaciones ADD COLUMN puesto_clasificacion TEXT DEFAULT ''")
        
        if 'porcentaje_clasificacion' not in columns:
            cursor.execute("ALTER TABLE postulaciones ADD COLUMN porcentaje_clasificacion REAL DEFAULT 0.0")
        
        if 'modelo_clasificacion' not in columns:
            cursor.execute("ALTER TABLE postulaciones ADD COLUMN modelo_clasificacion TEXT DEFAULT ''")

        conn.commit()
        print("Columnas para clasificación automática y modelo de clasificación agregadas o ya existentes.")
    except sqlite3.Error as e:
        print(f"Error al agregar columnas de clasificación: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    add_classification_columns()
