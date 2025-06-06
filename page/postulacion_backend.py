import os
import re
from postulacion_db import init_database, add_postulacion, get_all_postulaciones, get_postulacion_by_id, get_cv_data, delete_postulacion_from_db

class PostulacionManager:
    """Clase para manejar las operaciones de postulación."""
    
    def __init__(self):
        """Inicializa el manager y la base de datos."""
        self.allowed_extensions = {'.pdf'}  # Solo PDF
        self.max_file_size = 5 * 1024 * 1024  # 5MB

        # Inicializar base de datos
        init_database()
    
    def validate_email(self, email):
        """
        Valida el formato del correo electrónico.
        
        Args:
            email (str): Correo electrónico a validar
        
        Returns:
            bool: True si es válido, False en caso contrario
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_dni(self, dni):
        """
        Valida el formato del DNI peruano.

        Args:
            dni (str): DNI a validar

        Returns:
            bool: True si es válido, False en caso contrario
        """
        # DNI peruano: 8 dígitos
        pattern = r'^\d{8}$'
        return re.match(pattern, dni.strip()) is not None

    def validate_phone(self, phone):
        """
        Valida el formato del teléfono.

        Args:
            phone (str): Número de teléfono a validar

        Returns:
            bool: True si es válido, False en caso contrario
        """
        # Permitir números con o sin espacios, guiones, paréntesis
        pattern = r'^[\d\s\-\(\)\+]{7,15}$'
        return re.match(pattern, phone.strip()) is not None
    
    def validate_file(self, filename, file_data):
        """
        Valida el archivo CV.
        
        Args:
            filename (str): Nombre del archivo
            file_data (bytes): Datos del archivo
        
        Returns:
            dict: Resultado de la validación
        """
        if not filename:
            return {'valid': False, 'error': 'No se ha seleccionado ningún archivo.'}
        
        # Verificar extensión
        file_ext = os.path.splitext(filename.lower())[1]
        if file_ext not in self.allowed_extensions:
            return {
                'valid': False,
                'error': 'Solo se permiten archivos PDF.'
            }
        
        # Verificar tamaño
        if len(file_data) > self.max_file_size:
            return {
                'valid': False, 
                'error': f'El archivo es demasiado grande. Máximo: {self.max_file_size // (1024*1024)}MB'
            }
        
        # Verificar que no esté vacío
        if len(file_data) == 0:
            return {'valid': False, 'error': 'El archivo está vacío.'}
        
        return {'valid': True}
    
    def process_postulacion(self, nombre, dni, telefono, correo, cv_filename, cv_data):
        """
        Procesa una nueva postulación.

        Args:
            nombre (str): Nombre completo
            dni (str): DNI
            telefono (str): Teléfono
            correo (str): Correo electrónico
            cv_filename (str): Nombre del archivo CV
            cv_data (bytes): Datos del archivo CV

        Returns:
            dict: Resultado del procesamiento
        """
        # Validar datos básicos
        if not all([nombre.strip(), dni.strip(), telefono.strip(), correo.strip()]):
            return {
                'success': False,
                'message': 'Todos los campos son obligatorios.'
            }
        
        # Validar nombre (solo letras, espacios y algunos caracteres especiales)
        if not re.match(r'^[a-zA-ZáéíóúÁÉÍÓÚñÑ\s\-\.]{2,50}$', nombre.strip()):
            return {
                'success': False,
                'message': 'El nombre debe contener solo letras y tener entre 2 y 50 caracteres.'
            }

        # Validar DNI
        if not self.validate_dni(dni):
            return {
                'success': False,
                'message': 'El DNI debe tener exactamente 8 dígitos.'
            }

        # Validar teléfono
        if not self.validate_phone(telefono):
            return {
                'success': False,
                'message': 'El formato del teléfono no es válido.'
            }
        
        # Validar correo
        if not self.validate_email(correo):
            return {
                'success': False,
                'message': 'El formato del correo electrónico no es válido.'
            }
        
        # Validar archivo
        file_validation = self.validate_file(cv_filename, cv_data)
        if not file_validation['valid']:
            return {
                'success': False,
                'message': file_validation['error']
            }
        
        # Limpiar datos
        nombre = nombre.strip().title()
        dni = dni.strip()
        telefono = telefono.strip()
        correo = correo.strip().lower()

        # Guardar en base de datos
        result = add_postulacion(nombre, dni, telefono, correo, cv_filename, cv_data)
        
        return result
    
    def get_postulaciones_list(self):
        """
        Obtiene la lista de todas las postulaciones.
        
        Returns:
            list: Lista de postulaciones
        """
        return get_all_postulaciones()
    
    def get_postulacion_details(self, postulacion_id):
        """
        Obtiene los detalles de una postulación específica.
        
        Args:
            postulacion_id (int): ID de la postulación
        
        Returns:
            dict: Detalles de la postulación o None
        """
        postulacion = get_postulacion_by_id(postulacion_id)
        if postulacion:
            return {
                'id': postulacion[0],
                'nombre': postulacion[1],
                'dni': postulacion[2],
                'telefono': postulacion[3],
                'correo': postulacion[4],
                'cv_filename': postulacion[5],
                'cv_size': postulacion[7],
                'fecha_postulacion': postulacion[8],
                'estado': postulacion[9]
            }
        return None
    
    def download_cv(self, postulacion_id):
        """
        Obtiene los datos del CV para descarga.
        
        Args:
            postulacion_id (int): ID de la postulación
        
        Returns:
            tuple: (filename, data) o None si no existe
        """
        return get_cv_data(postulacion_id)

    def delete_postulacion(self, postulacion_id):
        """
        Elimina una postulación de la base de datos.

        Args:
            postulacion_id (int): ID de la postulación a eliminar.

        Returns:
            bool: True si la eliminación fue exitosa, False en caso contrario.
        """
        return delete_postulacion_from_db(postulacion_id)
    
    def export_postulaciones_csv(self):
        """
        Exporta las postulaciones a formato CSV.
        
        Returns:
            str: Contenido CSV
        """
        postulaciones = self.get_postulaciones_list()
        
        csv_content = "ID,Nombre,DNI,Teléfono,Correo,Archivo CV,Tamaño CV (KB),Fecha Postulación,Estado\n"

        for post in postulaciones:
            csv_content += f"{post[0]},{post[1]},{post[2]},{post[3]},{post[4]},{post[5]},{post[6]//1024},{post[7]},{post[8]}\n"
        
        return csv_content

# Funciones de utilidad para integración con PyQt
def create_postulacion_manager():
    """Crea una instancia del manager de postulaciones."""
    return PostulacionManager()

def handle_form_submission(form_data):
    """
    Maneja el envío del formulario desde la interfaz web.
    
    Args:
        form_data (dict): Datos del formulario
    
    Returns:
        dict: Resultado del procesamiento
    """
    manager = PostulacionManager()
    
    try:
        nombre = form_data.get('nombre', '')
        dni = form_data.get('dni', '')
        telefono = form_data.get('telefono', '')
        correo = form_data.get('correo', '')
        cv_filename = form_data.get('cv_filename', '')
        cv_data = form_data.get('cv_data', b'')

        return manager.process_postulacion(nombre, dni, telefono, correo, cv_filename, cv_data)
        
    except Exception as e:
        return {
            'success': False,
            'message': f'Error interno del servidor: {str(e)}'
        }

def get_postulaciones_summary():
    """
    Obtiene un resumen de las postulaciones para mostrar en la interfaz.
    
    Returns:
        dict: Resumen de postulaciones
    """
    manager = PostulacionManager()
    postulaciones = manager.get_postulaciones_list()
    
    total = len(postulaciones)
    pendientes = sum(1 for p in postulaciones if p[7] == 'pendiente')
    
    return {
        'total': total,
        'pendientes': pendientes,
        'recientes': postulaciones[:5]  # Últimas 5 postulaciones
    }

if __name__ == '__main__':
    # Prueba básica del sistema
    print("Probando sistema de postulaciones...")
    
    manager = PostulacionManager()
    
    # Datos de prueba
    test_data = {
        'nombre': 'Juan Pérez',
        'dni': '12345678',
        'telefono': '123-456-7890',
        'correo': 'juan.perez@email.com',
        'cv_filename': 'cv_juan_perez.pdf',
        'cv_data': b'Datos de prueba del CV'
    }

    # Procesar postulación de prueba
    result = manager.process_postulacion(
        test_data['nombre'],
        test_data['dni'],
        test_data['telefono'],
        test_data['correo'],
        test_data['cv_filename'],
        test_data['cv_data']
    )
    
    print(f"Resultado de prueba: {result}")
    
    # Mostrar resumen
    summary = get_postulaciones_summary()
    print(f"Resumen: {summary}")
