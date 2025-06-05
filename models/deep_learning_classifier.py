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
warnings.filterwarnings('ignore')

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
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.is_trained = False
        self.model_type = None
        self.max_length = 512
        self.vocab_size = 10000
        self.embedding_dim = 128
        
        # Directorio para modelos
        self.model_dir = "saved_deep_models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def check_dependencies(self, model_type):
        """Verifica que las dependencias estén disponibles"""
        if model_type in ['lstm', 'cnn'] and not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow no está instalado. Instala con: pip install tensorflow")
        
        if model_type == 'bert' and not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers no está instalado. Instala con: pip install transformers")
        
        return True
    
    def prepare_data_traditional(self, texts, labels):
        """Prepara datos para LSTM y CNN"""
        # Tokenizar textos
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
        
        # Convertir textos a secuencias
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Padding
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        # Codificar etiquetas
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)
        y = to_categorical(y_encoded)
        
        return X, y
    
    def prepare_data_bert(self, texts, labels):
        """Prepara datos para BERT"""
        # Usar tokenizer de BERT
        model_name = 'distilbert-base-uncased'  # Modelo más ligero
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Tokenizar textos
        encoded = self.bert_tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='tf'
        )
        
        # Codificar etiquetas
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)
        y = to_categorical(y_encoded)
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }, y
    
    def create_lstm_model(self, num_classes):
        """Crea modelo LSTM"""
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_cnn_model(self, num_classes):
        """Crea modelo CNN para texto"""
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_bert_model(self, num_classes):
        """Crea modelo BERT"""
        model_name = 'distilbert-base-uncased'
        
        # Cargar modelo pre-entrenado
        bert_model = TFAutoModel.from_pretrained(model_name)
        
        # Inputs
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
        
        # BERT embeddings
        bert_output = bert_model(input_ids, attention_mask=attention_mask)
        
        # Clasificador
        pooled_output = bert_output.last_hidden_state[:, 0, :]  # [CLS] token
        x = Dense(128, activation='relu')(pooled_output)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, data, model_type='lstm', epochs=10, batch_size=32):
        """Entrena el modelo de deep learning"""
        try:
            # Verificar dependencias
            self.check_dependencies(model_type)
            
            print(f"=== INICIANDO ENTRENAMIENTO DEEP LEARNING ===")
            print(f"Modelo: {model_type.upper()}")
            
            # Preparar datos
            texts = [item['text'] for item in data]
            labels = [item['profession'] for item in data]
            
            print(f"Datos preparados: {len(texts)} CVs, {len(set(labels))} profesiones")
            print(f"Profesiones: {set(labels)}")
            
            # Preparar datos según el tipo de modelo
            if model_type == 'bert':
                X, y = self.prepare_data_bert(texts, labels)
            else:
                X, y = self.prepare_data_traditional(texts, labels)
            
            num_classes = len(self.label_encoder.classes_)
            
            # Dividir datos
            if model_type == 'bert':
                X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
                    X['input_ids'], X['attention_mask'], y, test_size=0.2, random_state=42, stratify=labels
                )
                X_train = {'input_ids': X_train_ids, 'attention_mask': X_train_mask}
                X_test = {'input_ids': X_test_ids, 'attention_mask': X_test_mask}
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=labels
                )
            
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
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            
            # Entrenar modelo
            print(f"Entrenando modelo...")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Evaluar modelo
            print(f"Evaluando modelo...")
            y_pred = self.model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
            
            accuracy = accuracy_score(y_test_classes, y_pred_classes)
            
            print(f"\n=== RESULTADOS DEL ENTRENAMIENTO ===")
            print(f"Precisión: {accuracy:.3f}")
            print(f"Épocas entrenadas: {len(history.history['loss'])}")
            
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
                    padding=True,
                    max_length=self.max_length,
                    return_tensors='tf'
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
        """Guarda el modelo entrenado en formato H5 dentro de una carpeta específica"""
        if not self.is_trained:
            raise ValueError("No hay modelo entrenado para guardar")

        try:
            # Crear carpeta específica para el modelo
            model_folder = os.path.join(self.model_dir, model_name)
            os.makedirs(model_folder, exist_ok=True)
            print(f"📁 Creando carpeta del modelo DL: {model_folder}")

            # Guardar modelo de TensorFlow en formato H5
            model_path = os.path.join(model_folder, 'model.h5')
            self.model.save(model_path, save_format='h5')
            print(f"✅ Modelo guardado en formato H5: {model_path}")

            # Guardar tokenizer y label encoder
            if self.model_type == 'bert':
                tokenizer_path = os.path.join(model_folder, 'bert_tokenizer')
                self.bert_tokenizer.save_pretrained(tokenizer_path)
                print(f"✅ Tokenizer BERT guardado: {tokenizer_path}")
            else:
                tokenizer_path = os.path.join(model_folder, 'tokenizer.pkl')
                joblib.dump(self.tokenizer, tokenizer_path)
                print(f"✅ Tokenizer guardado: {tokenizer_path}")

            encoder_path = os.path.join(model_folder, 'encoder.pkl')
            joblib.dump(self.label_encoder, encoder_path)
            print(f"✅ Label encoder guardado: {encoder_path}")

            # Guardar metadatos
            metadata = {
                'model_name': model_name,
                'model_type': f'Deep Learning - {self.model_type.upper()}',
                'professions': list(self.label_encoder.classes_),
                'max_length': self.max_length,
                'vocab_size': self.vocab_size,
                'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'num_professions': len(self.label_encoder.classes_),
                'deep_learning': True,
                'model_format': 'h5'
            }

            metadata_path = os.path.join(model_folder, 'metadata.pkl')
            joblib.dump(metadata, metadata_path)
            print(f"✅ Metadatos guardados: {metadata_path}")

            print(f"✅ Modelo Deep Learning '{model_name}' guardado completamente en {model_folder}/")
            print(f"   - model.h5")
            print(f"   - encoder.pkl")
            print(f"   - tokenizer.pkl o bert_tokenizer/")
            print(f"   - metadata.pkl")

            return True

        except Exception as e:
            print(f"❌ Error guardando modelo: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load_model(self, model_name='deep_cv_classifier'):
        """Carga un modelo guardado desde formato H5 en su carpeta específica"""
        try:
            # Carpeta específica del modelo
            model_folder = os.path.join(self.model_dir, model_name)

            if not os.path.exists(model_folder):
                print(f"❌ No se encontró la carpeta del modelo: {model_folder}")
                return False

            print(f"📁 Cargando modelo DL desde: {model_folder}")

            # Cargar metadatos
            metadata_path = os.path.join(model_folder, 'metadata.pkl')
            if not os.path.exists(metadata_path):
                print(f"❌ No se encontraron metadatos: {metadata_path}")
                return False

            metadata = joblib.load(metadata_path)
            self.model_type = metadata['model_type'].split(' - ')[1].lower()
            print(f"📋 Cargando modelo tipo: {self.model_type}")

            # Verificar dependencias
            self.check_dependencies(self.model_type)

            # Cargar modelo desde archivo H5
            model_path = os.path.join(model_folder, 'model.h5')
            if not os.path.exists(model_path):
                print(f"❌ No se encontró el archivo del modelo: {model_path}")
                return False

            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Modelo H5 cargado: {model_path}")

            # Cargar tokenizer
            if self.model_type == 'bert':
                tokenizer_path = os.path.join(model_folder, 'bert_tokenizer')
                if os.path.exists(tokenizer_path):
                    self.bert_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                    print(f"✅ Tokenizer BERT cargado: {tokenizer_path}")
                else:
                    print(f"❌ No se encontró tokenizer BERT: {tokenizer_path}")
                    return False
            else:
                tokenizer_path = os.path.join(model_folder, 'tokenizer.pkl')
                if os.path.exists(tokenizer_path):
                    self.tokenizer = joblib.load(tokenizer_path)
                    print(f"✅ Tokenizer cargado: {tokenizer_path}")
                else:
                    print(f"❌ No se encontró tokenizer: {tokenizer_path}")
                    return False

            # Cargar label encoder
            encoder_path = os.path.join(model_folder, 'encoder.pkl')
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)
                print(f"✅ Label encoder cargado: {encoder_path}")
            else:
                print(f"❌ No se encontró label encoder: {encoder_path}")
                return False

            # Restaurar configuración
            self.max_length = metadata.get('max_length', 512)
            self.vocab_size = metadata.get('vocab_size', 10000)

            self.is_trained = True
            print(f"✅ Modelo Deep Learning '{model_name}' cargado exitosamente")
            print(f"   - Profesiones: {list(self.label_encoder.classes_)}")
            print(f"   - Tipo: {self.model_type}")
            print(f"   - Max length: {self.max_length}")

            return True

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False
