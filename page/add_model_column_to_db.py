import sqlite3
import os

DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
DATABASE_NAME = "postulaciones.db"
DATABASE_PATH = os.path.join(DATABASE_DIR, DATABASE_NAME)

def add_model_column():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        # Verificar si la columna modelo_clasificacion ya existe
        cursor.execute("PRAGMA table_info(postulaciones)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'modelo_clasificacion' not in columns:
            cursor.execute("ALTER TABLE postulaciones ADD COLUMN modelo_clasificacion TEXT DEFAULT ''")
            print("Columna 'modelo_clasificacion' agregada correctamente.")
        else:
            print("La columna 'modelo_clasificacion' ya existe.")
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error al modificar la base de datos: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    add_model_column()
