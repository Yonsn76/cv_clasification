from flask import Flask, jsonify, request
from flask_cors import CORS
import threading

from postulacion_backend import PostulacionManager
from postulacion_extension import add_classification_columns
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model_manager import ModelManager
from models.cv_classifier import CVClassifier

app = Flask(__name__)
CORS(app)

# Inicializar base de datos con columnas extendidas
add_classification_columns()

postulacion_manager = PostulacionManager()
model_manager = ModelManager()

# Variable global para modelo activo
active_model_name = None
active_model_is_deep = False
active_classifier = None
active_classifier_lock = threading.Lock()

import os
from src.config.settings import Settings

def load_active_model():
    global active_classifier
    if active_model_name is None:
        active_classifier = None
        return
    # Determinar ruta del modelo según tipo
    model_dir = Settings.DEEP_MODELS_DIR if active_model_is_deep else Settings.MODELS_DIR
    classifier = CVClassifier(model_dir=str(model_dir))
    success = classifier.load_model(active_model_name)
    if success:
        active_classifier = classifier
    else:
        active_classifier = None

import os
import json

@app.route('/api/models', methods=['GET'])
def list_models():
    ml_classifier = CVClassifier()

    ml_models = ml_classifier.list_available_models()

    # List deep learning models by scanning saved_deep_models directory
    dl_models = []
    saved_dl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_deep_models')
    if os.path.exists(saved_dl_dir):
        for model_name in os.listdir(saved_dl_dir):
            model_path = os.path.join(saved_dl_dir, model_name)
            if os.path.isdir(model_path):
                # Try to read metadata file
                metadata_file = None
                for candidate in ['package_info.json', 'senati_info.json']:
                    candidate_path = os.path.join(model_path, candidate)
                    if os.path.isfile(candidate_path):
                        metadata_file = candidate_path
                        break
                if metadata_file:
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        dl_models.append({
                            'name': model_name,
                            'display_name': metadata.get('model_name', model_name),
                            'model_type': 'dl',
                            'creation_date': metadata.get('creation_date', ''),
                            'description': metadata.get('description', ''),
                            'is_deep_learning': True,
                            'num_professions': metadata.get('num_professions', 0),
                            'model_format': metadata.get('format_version', 'unknown'),
                            'num_features': metadata.get('num_features', 'N/A'),
                            'hyperparameters': metadata.get('hyperparameters', {})
                        })
                    except Exception:
                        # Ignore malformed metadata files
                        pass

    combined_models = ml_models + dl_models
    # Sort by creation_date descending if available
    def get_creation_date(model):
        return model.get('creation_date', '') or ''
    combined_models.sort(key=get_creation_date, reverse=True)

    return jsonify(combined_models)

@app.route('/api/models/select', methods=['POST'])
def select_model():
    global active_model_name, active_model_is_deep
    data = request.json
    model_name = data.get('model_name')
    is_deep = data.get('is_deep', False)
    if not model_name:
        return jsonify({'success': False, 'message': 'model_name es requerido'}), 400
    active_model_name = model_name
    active_model_is_deep = is_deep
    load_active_model()
    return jsonify({'success': True, 'message': f'Modelo {model_name} seleccionado'})

@app.route('/api/postulaciones/classify/<int:postulacion_id>', methods=['POST'])
def classify_postulacion(postulacion_id):
    global active_classifier, active_model_name
    if active_classifier is None:
        return jsonify({'success': False, 'message': 'No hay modelo activo cargado'}), 400
    postulacion = postulacion_manager.get_postulacion_details(postulacion_id)
    if not postulacion:
        return jsonify({'success': False, 'message': 'Postulación no encontrada'}), 404
    cv_data = postulacion_manager.download_cv(postulacion_id)
    if not cv_data:
        return jsonify({'success': False, 'message': 'CV no encontrado'}), 404
    cv_filename, cv_bytes = cv_data
    try:
        # Convertir bytes a texto para clasificación (asumiendo texto plano o extraído)
        cv_text = cv_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error decodificando CV: {str(e)}'}), 500
    result = active_classifier.predict_cv(cv_text)
    if result.get('error'):
        return jsonify({'success': False, 'message': result.get('message')}), 500
    # Guardar resultado en base de datos (crear método nuevo en page para esto)
    puesto = result.get('predicted_profession', '')
    porcentaje = result.get('confidence', 0.0)
    # Actualizar base de datos con puesto, porcentaje y modelo
    try:
        update_classification_result(postulacion_id, puesto, porcentaje, active_model_name)
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error guardando resultado: {str(e)}'}), 500
    return jsonify({'success': True, 'puesto': puesto, 'porcentaje': porcentaje, 'modelo': active_model_name})

def update_classification_result(postulacion_id, puesto, porcentaje, modelo):
    import sqlite3
    import os
    DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
    DATABASE_NAME = "postulaciones.db"
    DATABASE_PATH = os.path.join(DATABASE_DIR, DATABASE_NAME)
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE postulaciones
            SET puesto_clasificacion = ?, porcentaje_clasificacion = ?, modelo_clasificacion = ?
            WHERE id = ?
        """, (puesto, porcentaje, modelo, postulacion_id))
        conn.commit()
    finally:
        if conn:
            conn.close()

@app.route('/api/postulaciones', methods=['GET'])
def get_postulaciones():
    postulaciones = postulacion_manager.get_postulaciones_list()
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
            'estado': p[8],
            'puesto_clasificacion': p[9] if len(p) > 9 else '',
            'porcentaje_clasificacion': p[10] if len(p) > 10 else 0.0,
            'modelo_clasificacion': p[11] if len(p) > 11 else ''
        })
    return jsonify(postulaciones_list)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
