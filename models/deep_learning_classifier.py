"""
Clasificador de CVs usando modelos de Deep Learning
Incluye LSTM, BERT y CNN para texto
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
import json
import datetime
import pickle
warnings.filterwarnings('ignore')

from .model_manager import ModelManager

# Verificar disponibilidad de librerías de deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Input
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow disponible")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("❌ TensorFlow no disponible")

try:
    from transformers import AutoTokenizer, TFAutoModel
    import transformers
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers disponible")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ Transformers no disponible")

class DeepLearningClassifier:
    """Clasificador de CVs usando modelos de Deep Learning"""
    
    def __init__(self):
        """Inicializa el clasificador de Deep Learning"""
        self.model = None
        self.model_type = None
        self.is_trained = False
        self.max_length = 512
        self.vocab_size = 10000
        self.tokenizer = None
        self.bert_tokenizer = None
        self.label_encoder = None
        
        # Configuración de BERT
        self.bert_config = {
            'model_name': 'dccuchile/bert-base-spanish-wwm-uncased',
            'version': '1.0',
            'files': {
                'model': 'pytorch_model.bin',  # ~440MB
                'config': 'config.json',       # ~1MB
                'vocab': 'vocab.txt',          # ~1MB
                'tokenizer': 'tokenizer.json', # ~1MB
            }
        }
        
        # Inicializar el gestor de modelos
        self.model_manager = ModelManager()
        self.model_dir = self.model_manager.deep_models_dir
        self.bert_cache_dir = os.path.join(self.model_dir, 'bert_cache')
        os.makedirs(self.bert_cache_dir, exist_ok=True)
    
    def check_dependencies(self, model_type):
        """Verifica y carga las dependencias necesarias según el tipo de modelo"""
        try:
            if model_type == 'bert':
                from transformers import AutoTokenizer, TFAutoModel
                if self.bert_tokenizer is None:
                    print("\n=== Configurando tokenizer BERT ===")
                    print("📥 Cargando tokenizer...")
                    
                    self.bert_tokenizer = AutoTokenizer.from_pretrained(
                        self.bert_config['model_name'],
                        cache_dir=self.bert_cache_dir,
                        do_lower_case=True
                    )
                    print("✅ Tokenizer BERT cargado correctamente")
            else:
                from tensorflow.keras.preprocessing.text import Tokenizer
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                if self.tokenizer is None:
                    print("Inicializando tokenizer tradicional...")
                    self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
            
            print("✅ Dependencias cargadas correctamente")
            return True
        except Exception as e:
            print(f"❌ Error cargando dependencias: {str(e)}")
            raise e
    
    def prepare_data_traditional(self, texts, labels):
        """Prepara los datos para modelos tradicionales (LSTM, CNN)"""
        if self.tokenizer is None:
            raise ValueError("El tokenizer no está inicializado. Llama a check_dependencies primero.")

        # Ajustar tokenizer a los textos
        self.tokenizer.fit_on_texts(texts)
        self.vocab_size = min(self.vocab_size, len(self.tokenizer.word_index) + 1)

        # Convertir textos a secuencias
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        return X, labels
    
    def prepare_data_bert(self, texts, labels):
        """Prepara los datos para el modelo BERT"""
        if self.bert_tokenizer is None:
            raise ValueError("El tokenizer BERT no está inicializado. Llama a check_dependencies primero.")

        print(f"\n=== Preparando datos para BERT ===")
        print(f"Longitud máxima de secuencia: {self.max_length}")
        print(f"Número de textos a procesar: {len(texts)}")

        # Tokenizar textos con BERT y asegurar padding correcto
        encoded = self.bert_tokenizer(
            texts,
            truncation=True,
            padding='max_length',  # Cambiar a max_length para forzar longitud fija
            max_length=self.max_length,
            return_tensors='tf',
            return_attention_mask=True
        )
        
        # Verificar formas de los tensores
        input_shape = encoded['input_ids'].shape
        mask_shape = encoded['attention_mask'].shape
        print(f"Forma del tensor de entrada: {input_shape}")
        print(f"Forma del tensor de máscara: {mask_shape}")
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }, labels
    
    def create_lstm_model(self, num_classes):
        """Crea un modelo LSTM para clasificación de texto"""
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, 128, input_length=self.max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def create_cnn_model(self, num_classes):
        """Crea un modelo CNN para clasificación de texto"""
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, 128, input_length=self.max_length),
            tf.keras.layers.Conv1D(128, 5, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def check_bert_cache(self):
        """Verifica si los archivos de BERT están en caché"""
        try:
            cache_info_file = os.path.join(self.bert_cache_dir, 'cache_info.json')
            
            # Verificar si existe información de caché
            if os.path.exists(cache_info_file):
                with open(cache_info_file, 'r') as f:
                    cache_info = json.load(f)
                
                # Verificar versión y archivos
                if (cache_info.get('version') == self.bert_config['version'] and
                    all(os.path.exists(os.path.join(self.bert_cache_dir, fname))
                        for fname in self.bert_config['files'].values())):
                    print("✅ Archivos BERT encontrados en caché")
                    return True
            
            return False
        except Exception as e:
            print(f"⚠️ Error verificando caché BERT: {str(e)}")
            return False

    def update_bert_cache_info(self):
        """Actualiza la información de caché de BERT"""
        try:
            cache_info = {
                'version': self.bert_config['version'],
                'model_name': self.bert_config['model_name'],
                'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'files': list(self.bert_config['files'].values())
            }
            
            cache_info_file = os.path.join(self.bert_cache_dir, 'cache_info.json')
            with open(cache_info_file, 'w') as f:
                json.dump(cache_info, f, indent=4)
        except Exception as e:
            print(f"⚠️ Error actualizando información de caché: {str(e)}")

    def create_bert_model(self, num_classes):
        """Crea un modelo BERT para clasificación de texto"""
        try:
            print("\n=== Configurando modelo BERT ===")
            
            # Verificar caché
            if not self.check_bert_cache():
                print(f"📥 Descargando componentes de BERT ({self.bert_config['model_name']})...")
                print("Este proceso descargará varios archivos:")
                for file, size in [
                    ('Modelo base', '~440MB'),
                    ('Configuración', '~1MB'),
                    ('Vocabulario', '~1MB'),
                    ('Tokenizer', '~1MB')
                ]:
                    print(f"  - {file}: {size}")
                print("\nLos archivos se guardarán en caché para usos futuros.")
            else:
                print("🔄 Usando archivos BERT desde caché local")
            
            # Cargar modelo base BERT con manejo de caché
            bert_model = TFAutoModel.from_pretrained(
                self.bert_config['model_name'],
                cache_dir=self.bert_cache_dir,
                from_pt=True
            )
            print("✅ Modelo BERT base cargado exitosamente")
            
            # Actualizar información de caché
            self.update_bert_cache_info()
            
            # Crear el modelo
            print("🔧 Construyendo arquitectura del clasificador...")
            input_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
            attention_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
            
            # Obtener embeddings de BERT
            bert_outputs = bert_model(input_ids, attention_mask=attention_mask)
            pooled_output = bert_outputs[1]
            
            # Agregar capas de clasificación
            x = tf.keras.layers.Dense(256, activation='relu')(pooled_output)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            
            model = tf.keras.Model(
                inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
                outputs=outputs
            )
            print("✅ Arquitectura del modelo construida")
            
            # Compilar modelo
            print("⚙️ Compilando modelo...")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("✅ Modelo BERT compilado y listo para entrenar")
            
            return model
            
        except Exception as e:
            print(f"❌ Error creando modelo BERT: {str(e)}")
            raise e

    def train_model(self, data, model_type='lstm', epochs=10, batch_size=32, callbacks=None):
        """Entrena un modelo de Deep Learning con los datos proporcionados"""
        try:
            print(f"\n=== ENTRENAMIENTO DE MODELO {model_type.upper()} ===")
            print(f"Épocas configuradas: {epochs}")
            print(f"Batch size: {batch_size}")
            
            # Verificar dependencias
            self.check_dependencies(model_type)
            
            # Preparar datos
            texts = [item['text'] for item in data]
            labels = [item['profession'] for item in data]
            
            # Codificar etiquetas
            self.label_encoder = LabelEncoder()
            labels_encoded = self.label_encoder.fit_transform(labels)
            
            # Preparar textos según el tipo de modelo
            if model_type == 'bert':
                X, _ = self.prepare_data_bert(texts, labels_encoded)
            else:
                X, _ = self.prepare_data_traditional(texts, labels_encoded)
            
            num_classes = len(self.label_encoder.classes_)
            print(f"Número de clases: {num_classes}")
            print(f"Clases: {self.label_encoder.classes_}")
            
            # Convertir etiquetas a one-hot encoding usando TensorFlow
            y = tf.keras.utils.to_categorical(labels_encoded, num_classes=num_classes)
            
            # Dividir datos asegurando que sean tensores de TensorFlow
            if model_type == 'bert':
                X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
                    X['input_ids'].numpy(), X['attention_mask'].numpy(), y,
                    test_size=0.2, random_state=42, stratify=labels_encoded
                )
                # Convertir de nuevo a tensores
                X_train = {
                    'input_ids': tf.convert_to_tensor(X_train_ids, dtype=tf.int32),
                    'attention_mask': tf.convert_to_tensor(X_train_mask, dtype=tf.int32)
                }
                X_test = {
                    'input_ids': tf.convert_to_tensor(X_test_ids, dtype=tf.int32),
                    'attention_mask': tf.convert_to_tensor(X_test_mask, dtype=tf.int32)
                }
            else:
                X = tf.convert_to_tensor(X, dtype=tf.float32)
                X_train, X_test, y_train, y_test = train_test_split(
                    X.numpy(), y, test_size=0.2, random_state=42, stratify=labels_encoded
                )
                # Convertir de nuevo a tensores
                X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
                X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
            
            # Convertir etiquetas a tensores
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
            y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
            
            # Crear modelo
            print(f"Creando modelo {model_type.upper()}...")
            if model_type == 'lstm':
                self.model = self.create_lstm_model(num_classes)
            elif model_type == 'cnn':
                self.model = self.create_cnn_model(num_classes)
            elif model_type == 'bert':
                self.model = self.create_bert_model(num_classes)
            else:
                raise ValueError(f"Tipo de modelo no soportado: {model_type}")
            
            self.model_type = model_type
            
            # Callbacks base
            training_callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.model_dir, 'best_model.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(self.model_dir, 'logs'),
                    histogram_freq=1
                )
            ]
            
            # Agregar callbacks adicionales si se proporcionan
            if callbacks:
                if isinstance(callbacks, list):
                    training_callbacks.extend(callbacks)
                else:
                    training_callbacks.append(callbacks)
            
            # Entrenar modelo
            print(f"\nIniciando entrenamiento...")
            print(f"Épocas: {epochs}")
            print(f"Batch size: {batch_size}")
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=training_callbacks,
                verbose=1
            )
            
            # Evaluar modelo
            print(f"\nEvaluando modelo...")
            y_pred = self.model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_test_classes = np.argmax(y_test.numpy(), axis=1)
            
            accuracy = accuracy_score(y_test_classes, y_pred_classes)
            
            print(f"\n=== RESULTADOS DEL ENTRENAMIENTO ===")
            print(f"Precisión: {accuracy:.3f}")
            print(f"Épocas completadas: {len(history.history['loss'])}")
            
            # Reporte detallado
            report = classification_report(
                y_test_classes, y_pred_classes,
                target_names=self.label_encoder.classes_,
                zero_division=0
            )
            print(f"\nReporte de clasificación:")
            print(report)
            
            self.is_trained = True
            
            return {
                'success': True,
                'accuracy': accuracy,
                'model_type': model_type,
                'epochs_trained': len(history.history['loss']),
                'num_classes': num_classes,
                'history': history.history
            }
            
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_cv(self, text):
        """Predice la profesión de un CV"""
        if not self.is_trained:
            return {'error': True, 'message': 'Modelo no entrenado'}
        
        try:
            # Preparar texto
            if self.model_type == 'bert':
                encoded = self.bert_tokenizer(
                    [text],
                    truncation=True,
                    padding='max_length',  # Usar max_length para consistencia
                    max_length=self.max_length,
                    return_tensors='tf',
                    return_attention_mask=True
                )
                X = {
                    'input_ids': encoded['input_ids'],
                    'attention_mask': encoded['attention_mask']
                }
            else:
                sequence = self.tokenizer.texts_to_sequences([text])
                X = pad_sequences(sequence, maxlen=self.max_length, padding='post', truncating='post')
            
            # Predecir
            prediction = self.model.predict(X, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
            
            # Obtener nombre de la profesión
            profession = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Ranking de todas las profesiones
            ranking = []
            for i, prob in enumerate(prediction[0]):
                prof_name = self.label_encoder.inverse_transform([i])[0]
                ranking.append({
                    'profession': prof_name,
                    'probability': float(prob),
                    'percentage': f"{float(prob)*100:.1f}%"
                })

            # Ordenar por probabilidad
            ranking.sort(key=lambda x: x['probability'], reverse=True)

            # Determinar nivel de confianza
            if confidence > 0.8:
                confidence_level = "Alta"
            elif confidence > 0.6:
                confidence_level = "Media"
            else:
                confidence_level = "Baja"

            return {
                'error': False,
                'predicted_profession': profession,
                'confidence': confidence,
                'confidence_percentage': f"{confidence*100:.1f}%",
                'confidence_level': confidence_level,
                'profession_ranking': ranking
            }
            
        except Exception as e:
            return {'error': True, 'message': str(e)}
    
    def save_model(self, model_name='deep_cv_classifier'):
        """Guarda el modelo entrenado y sus componentes"""
        if not self.is_trained:
            print("❌ El modelo no está entrenado")
            return False

        try:
            print(f"\n=== Guardando modelo '{model_name}' ===")
            
            # Crear directorio para el modelo
            model_folder = os.path.join(self.model_dir, model_name)
            os.makedirs(model_folder, exist_ok=True)
            print(f"📁 Directorio del modelo: {model_folder}")

            # Crear metadatos
            metadata = {
                'model_type': self.model_type,
                'max_length': self.max_length,
                'vocab_size': self.vocab_size,
                'num_classes': len(self.label_encoder.classes_),
                'classes': list(self.label_encoder.classes_),
                'saved_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'is_deep_learning': True
            }

            # Guardar modelo en formato H5
            model_path = os.path.join(model_folder, 'model.h5')
            self.model.save(model_path)
            print("✅ Modelo guardado")

            # Guardar tokenizer
            if self.tokenizer:
                joblib.dump(self.tokenizer, os.path.join(model_folder, 'tokenizer.pkl'))
            elif self.bert_tokenizer:
                self.bert_tokenizer.save_pretrained(os.path.join(model_folder, 'bert_tokenizer'))
            print("✅ Tokenizer guardado")

            # Guardar encoder
            joblib.dump(self.label_encoder, os.path.join(model_folder, 'encoder.pkl'))
            print("✅ Encoder guardado")

            # Guardar metadatos
            joblib.dump(metadata, os.path.join(model_folder, 'metadata.pkl'))
            print("✅ Metadatos guardados")

            print(f"\n✅ Modelo '{model_name}' guardado exitosamente")
            return True

        except Exception as e:
            print(f"❌ Error guardando modelo: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load_model(self, model_name):
        """Carga un modelo entrenado y sus componentes"""
        try:
            print(f"\n=== Cargando modelo '{model_name}' ===")
            
            # Verificar que existe el directorio del modelo
            model_folder = os.path.join(self.model_dir, model_name)
            if not os.path.exists(model_folder):
                print(f"❌ No se encontró el directorio del modelo: {model_folder}")
                return False
            
            # Cargar metadatos
            metadata_path = os.path.join(model_folder, 'metadata.pkl')
            if not os.path.exists(metadata_path):
                print(f"❌ No se encontraron metadatos del modelo en: {metadata_path}")
                return False
            
            metadata = joblib.load(metadata_path)
            print("✅ Metadatos cargados")
            
            # Cargar modelo
            model_path = os.path.join(model_folder, 'model.h5')
            if not os.path.exists(model_path):
                print(f"❌ No se encontró el modelo en: {model_path}")
                return False
            
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Modelo cargado")
            
            # Cargar tokenizer
            if metadata['model_type'].lower() == 'bert':
                tokenizer_path = os.path.join(model_folder, 'bert_tokenizer')
                if os.path.exists(tokenizer_path):
                    self.bert_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                    print("✅ Tokenizer BERT cargado")
            else:
                tokenizer_path = os.path.join(model_folder, 'tokenizer.pkl')
                if os.path.exists(tokenizer_path):
                    self.tokenizer = joblib.load(tokenizer_path)
                    print("✅ Tokenizer cargado")
            
            # Cargar encoder
            encoder_path = os.path.join(model_folder, 'encoder.pkl')
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)
                print("✅ Label encoder cargado")
            
            # Configurar parámetros
            self.model_type = metadata['model_type']
            self.max_length = metadata.get('max_length', self.max_length)
            self.vocab_size = metadata.get('vocab_size', self.vocab_size)
            self.is_trained = True
            
            print(f"\n✅ Modelo '{model_name}' cargado exitosamente")
            print(f"   Tipo: {self.model_type}")
            print(f"   Profesiones: {len(metadata['classes'])}")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False
