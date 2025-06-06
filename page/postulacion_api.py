from flask import Flask, jsonify, request
from postulacion_backend import PostulacionManager
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Permitir CORS para desarrollo local

postulacion_manager = PostulacionManager()

@app.route('/api/postulaciones', methods=['GET', 'POST'])
def handle_postulaciones():
    if request.method == 'POST':
        nombre = request.form.get('nombre')
        dni = request.form.get('dni')
        correo = request.form.get('correo')
        telefono = request.form.get('telefono')
        cv_file = request.files.get('cv')

        if not all([nombre, dni, correo, telefono, cv_file]):
            return jsonify({'success': False, 'message': 'Faltan datos en el formulario.'}), 400

        if not cv_file.filename:
            return jsonify({'success': False, 'message': 'No se ha subido ningún archivo CV.'}), 400

        cv_filename = cv_file.filename
        cv_data = cv_file.read()

        result = postulacion_manager.process_postulacion(nombre, dni, telefono, correo, cv_filename, cv_data)

        if result.get('success'):
            return jsonify(result), 201
        else:
            return jsonify(result), 400
    
    # Manejo para GET
    postulaciones = postulacion_manager.get_postulaciones_list()
    # Convertir a lista de dicts para JSON
    postulaciones_list = []
    for p in postulaciones:
        postulaciones_list.append({
            'id': p[0],
            'nombre': p[1],
            'dni': p[2],
            'telefono': p[3],
            'correo': p[4],
            'cv_filename': p[5],
            'cv_size': p[6],
            'fecha_postulacion': p[7],
            'estado': p[8]
        })
    return jsonify(postulaciones_list)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
