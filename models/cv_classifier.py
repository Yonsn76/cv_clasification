# -*- coding: utf-8 -*-
"""
Clasificador de CVs por profesiones - Versión simplificada
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

class CVClassifier:
    """Clasificador simplificado de CVs por profesiones"""
    
    def __init__(self, model_dir='saved_models'):
        self.model_dir = model_dir
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.is_trained = False

        # Crear directorio de modelos
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_training_data(self, cv_data):
        """Prepara los datos para entrenamiento"""
        if not cv_data:
            raise ValueError("No hay datos de CVs para entrenar")
        
        # Convertir a DataFrame
        df = pd.DataFrame(cv_data)
        
        # Filtrar solo CVs procesados exitosamente
        df = df[df['status'] == 'success'].copy()
        
        if len(df) == 0:
            raise ValueError("No hay CVs procesados exitosamente")
        
        # Preparar textos y etiquetas
        texts = df['text'].tolist()
        professions = df['profession'].tolist()
        
        print(f"Datos preparados: {len(texts)} CVs, {len(set(professions))} profesiones")
        print(f"Profesiones: {set(professions)}")
        
        return texts, professions
    
    def train_model(self, cv_data, test_size=0.2, model_type='random_forest'):
        """Entrena el modelo de clasificación"""
        print("=== INICIANDO ENTRENAMIENTO ===")
        
        # Preparar datos
        texts, professions = self.prepare_training_data(cv_data)
        
        if len(set(professions)) < 2:
            raise ValueError("Se necesitan al menos 2 profesiones diferentes para entrenar")
        
        # Vectorizar textos
        print("Vectorizando textos...")

        # Ajustar parámetros según el tamaño del dataset
        min_df = 1 if len(texts) < 10 else 2
        max_features = min(5000, len(texts) * 100)

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=None,  # Mantenemos todas las palabras para español
            ngram_range=(1, 2),  # Unigramas y bigramas
            min_df=min_df,  # Ajustado según tamaño del dataset
            max_df=0.95  # Máximo 95% de documentos
        )
        
        X = self.vectorizer.fit_transform(texts)
        
        # Codificar etiquetas
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(professions)
        
        print(f"Características extraídas: {X.shape[1]}")
        print(f"Clases: {self.label_encoder.classes_}")
        
        # Dividir datos
        if len(texts) > 4:  # Solo dividir si hay suficientes datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            # Con pocos datos, usar todo para entrenamiento
            X_train, X_test = X, X
            y_train, y_test = y, y
            print("⚠️ Pocos datos: usando todo el dataset para entrenamiento y prueba")
        
        # Entrenar modelo
        print(f"Entrenando modelo {model_type}...")

        if model_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=2
            )
        elif model_type == 'logistic_regression':
            self.classifier = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
        elif model_type == 'svm':
            self.classifier = SVC(
                kernel='rbf',
                random_state=42,
                probability=True,  # Necesario para predict_proba
                C=1.0,
                gamma='scale'
            )
        elif model_type == 'naive_bayes':
            self.classifier = MultinomialNB(
                alpha=1.0  # Suavizado de Laplace
            )
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")

        # Nota: Para SVM y Naive Bayes, asegurar que los datos sean no negativos
        if model_type in ['svm', 'naive_bayes']:
            # TF-IDF ya produce valores no negativos, así que está bien
            pass

        self.classifier.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n=== RESULTADOS DEL ENTRENAMIENTO ===")
        print(f"Precisión: {accuracy:.3f}")
        print(f"Datos de entrenamiento: {X_train.shape[0]}")
        print(f"Datos de prueba: {X_test.shape[0]}")
        
        # Reporte detallado
        if len(set(y_test)) > 1:  # Solo si hay múltiples clases en test
            report = classification_report(
                y_test, y_pred, 
                target_names=self.label_encoder.classes_,
                zero_division=0
            )
            print("\nReporte de clasificación:")
            print(report)
        
        self.is_trained = True
        
        return {
            'success': True,
            'accuracy': accuracy,
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'features': X.shape[1],
            'classes': list(self.label_encoder.classes_)
        }
    
    def predict_cv(self, cv_text):
        """Predice la profesión más adecuada para un CV"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        if not cv_text or cv_text.strip() == "":
            return {
                'error': True,
                'message': 'El texto del CV está vacío'
            }
        
        try:
            # Vectorizar texto
            X = self.vectorizer.transform([cv_text])
            
            # Predecir
            prediction = self.classifier.predict(X)[0]
            probabilities = self.classifier.predict_proba(X)[0]
            
            # Obtener nombre de la profesión
            profession = self.label_encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))
            
            # Crear ranking de profesiones
            profession_ranking = []
            for i, prob in enumerate(probabilities):
                prof_name = self.label_encoder.inverse_transform([i])[0]
                profession_ranking.append({
                    'profession': prof_name,
                    'probability': float(prob),
                    'percentage': f"{prob*100:.1f}%"
                })
            
            # Ordenar por probabilidad
            profession_ranking.sort(key=lambda x: x['probability'], reverse=True)
            
            # Determinar nivel de confianza
            if confidence > 0.8:
                confidence_level = 'Alta'
            elif confidence > 0.6:
                confidence_level = 'Media'
            else:
                confidence_level = 'Baja'
            
            return {
                'predicted_profession': profession,
                'confidence': confidence,
                'confidence_level': confidence_level,
                'confidence_percentage': f"{confidence*100:.1f}%",
                'profession_ranking': profession_ranking,
                'error': False
            }
            
        except Exception as e:
            return {
                'error': True,
                'message': f'Error en la predicción: {str(e)}'
            }
    
    def save_model(self, model_name='cv_classifier'):
        """Guarda el modelo entrenado en una carpeta específica con el nombre del modelo"""
        if not self.is_trained:
            raise ValueError("No hay modelo entrenado para guardar")

        try:
            # Crear carpeta específica para el modelo
            model_folder = os.path.join(self.model_dir, model_name)
            os.makedirs(model_folder, exist_ok=True)
            print(f"📁 Creando carpeta del modelo: {model_folder}")

            # Guardar vectorizador
            vectorizer_path = os.path.join(model_folder, 'vectorizer.pkl')
            joblib.dump(self.vectorizer, vectorizer_path)
            print(f"✅ Vectorizador guardado: {vectorizer_path}")

            # Guardar clasificador
            classifier_path = os.path.join(model_folder, 'classifier.pkl')
            joblib.dump(self.classifier, classifier_path)
            print(f"✅ Clasificador guardado: {classifier_path}")

            # Guardar codificador de etiquetas
            encoder_path = os.path.join(model_folder, 'encoder.pkl')
            joblib.dump(self.label_encoder, encoder_path)
            print(f"✅ Codificador guardado: {encoder_path}")

            # Obtener nombre amigable del algoritmo
            algorithm_names = {
                'RandomForestClassifier': 'Random Forest',
                'LogisticRegression': 'Logistic Regression',
                'SVC': 'Support Vector Machine (SVM)',
                'MultinomialNB': 'Naive Bayes'
            }

            model_type_name = algorithm_names.get(
                type(self.classifier).__name__,
                type(self.classifier).__name__
            )

            # Guardar metadatos del modelo
            metadata = {
                'model_name': model_name,
                'model_type': model_type_name,
                'professions': list(self.label_encoder.classes_),
                'num_features': self.vectorizer.max_features,
                'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'num_professions': len(self.label_encoder.classes_),
                'model_format': 'sklearn_pkl'
            }

            metadata_path = os.path.join(model_folder, 'metadata.pkl')
            joblib.dump(metadata, metadata_path)
            print(f"✅ Metadatos guardados: {metadata_path}")

            print(f"✅ Modelo ML '{model_name}' guardado completamente en {model_folder}/")
            print(f"   - vectorizer.pkl")
            print(f"   - classifier.pkl")
            print(f"   - encoder.pkl")
            print(f"   - metadata.pkl")

            return True

        except Exception as e:
            print(f"❌ Error guardando modelo: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load_model(self, model_name='cv_classifier'):
        """Carga un modelo previamente entrenado desde su carpeta específica"""
        try:
            # Carpeta específica del modelo
            model_folder = os.path.join(self.model_dir, model_name)

            if not os.path.exists(model_folder):
                print(f"❌ No se encontró la carpeta del modelo: {model_folder}")
                return False

            print(f"📁 Cargando modelo desde: {model_folder}")

            # Cargar vectorizador
            vectorizer_path = os.path.join(model_folder, 'vectorizer.pkl')
            if not os.path.exists(vectorizer_path):
                print(f"❌ No se encontró vectorizer.pkl en {model_folder}")
                return False
            self.vectorizer = joblib.load(vectorizer_path)
            print(f"✅ Vectorizador cargado")

            # Cargar clasificador
            classifier_path = os.path.join(model_folder, 'classifier.pkl')
            if not os.path.exists(classifier_path):
                print(f"❌ No se encontró classifier.pkl en {model_folder}")
                return False
            self.classifier = joblib.load(classifier_path)
            print(f"✅ Clasificador cargado")

            # Cargar codificador de etiquetas
            encoder_path = os.path.join(model_folder, 'encoder.pkl')
            if not os.path.exists(encoder_path):
                print(f"❌ No se encontró encoder.pkl en {model_folder}")
                return False
            self.label_encoder = joblib.load(encoder_path)
            print(f"✅ Codificador cargado")

            # Cargar metadatos si existen
            metadata_path = os.path.join(model_folder, 'metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                print(f"✅ Metadatos cargados")
                print(f"   Tipo de modelo: {metadata.get('model_type', 'Unknown')}")
                print(f"   Fecha de creación: {metadata.get('creation_date', 'Unknown')}")

            self.is_trained = True

            print(f"✅ Modelo '{model_name}' cargado exitosamente")
            print(f"   Profesiones disponibles: {list(self.label_encoder.classes_)}")

            return True

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False
    
    def get_model_info(self):
        """Retorna información sobre el modelo cargado"""
        if not self.is_trained:
            return None

        # Obtener nombre amigable del algoritmo
        algorithm_names = {
            'RandomForestClassifier': 'Random Forest',
            'LogisticRegression': 'Logistic Regression',
            'SVC': 'Support Vector Machine (SVM)',
            'MultinomialNB': 'Naive Bayes'
        }

        model_type_name = algorithm_names.get(
            type(self.classifier).__name__,
            type(self.classifier).__name__
        ) if self.classifier else 'Unknown'

        return {
            'is_trained': self.is_trained,
            'professions': list(self.label_encoder.classes_),
            'num_professions': len(self.label_encoder.classes_),
            'num_features': self.vectorizer.max_features if self.vectorizer else 0,
            'model_type': model_type_name
        }

    def list_available_models(self):
        """Lista todos los modelos disponibles (tradicionales y Deep Learning)"""
        models = []

        # Modelos tradicionales (en carpetas)
        if os.path.exists(self.model_dir):
            # Buscar carpetas de modelos
            for item in os.listdir(self.model_dir):
                item_path = os.path.join(self.model_dir, item)
                if os.path.isdir(item_path):
                    model_name = item
                    try:
                        metadata_path = os.path.join(item_path, 'metadata.pkl')

                        if os.path.exists(metadata_path):
                            metadata = joblib.load(metadata_path)

                            # Verificar que todos los archivos del modelo existen
                            required_files = ['vectorizer.pkl', 'classifier.pkl', 'encoder.pkl']

                            all_files_exist = all(
                                os.path.exists(os.path.join(item_path, f))
                                for f in required_files
                            )

                            if all_files_exist:
                                models.append({
                                    'name': model_name,
                                    'display_name': metadata.get('model_name', model_name),
                                    'model_type': metadata.get('model_type', 'Unknown'),
                                    'professions': metadata.get('professions', []),
                                    'num_professions': metadata.get('num_professions', 0),
                                    'creation_date': metadata.get('creation_date', 'Unknown'),
                                    'num_features': metadata.get('num_features', 0),
                                    'is_deep_learning': False,
                                    'model_format': metadata.get('model_format', 'sklearn_pkl')
                                })

                    except Exception as e:
                        print(f"Error leyendo metadatos de {model_name}: {e}")
                        continue

        # Modelos de Deep Learning (en carpetas)
        deep_models_dir = "saved_deep_models"
        if os.path.exists(deep_models_dir):
            for item in os.listdir(deep_models_dir):
                item_path = os.path.join(deep_models_dir, item)
                if os.path.isdir(item_path):
                    model_name = item
                    try:
                        metadata_path = os.path.join(item_path, 'metadata.pkl')

                        if os.path.exists(metadata_path):
                            metadata = joblib.load(metadata_path)

                            # Verificar que el modelo de Deep Learning existe (formato H5)
                            model_path = os.path.join(item_path, 'model.h5')
                            encoder_path = os.path.join(item_path, 'encoder.pkl')

                            if os.path.exists(model_path) and os.path.exists(encoder_path):
                                models.append({
                                    'name': model_name,
                                    'display_name': metadata.get('model_name', model_name),
                                    'model_type': metadata.get('model_type', 'Deep Learning'),
                                    'professions': metadata.get('professions', []),
                                    'num_professions': metadata.get('num_professions', 0),
                                    'creation_date': metadata.get('creation_date', 'Unknown'),
                                    'num_features': metadata.get('max_length', 0),
                                    'is_deep_learning': True,
                                    'model_format': metadata.get('model_format', 'h5')
                                })

                    except Exception as e:
                        print(f"Error leyendo metadatos DL de {model_name}: {e}")
                        continue

        return sorted(models, key=lambda x: x['creation_date'], reverse=True)

    def delete_model(self, model_name, is_deep_learning=False):
        """Elimina un modelo y toda su carpeta (tradicional o Deep Learning)"""
        try:
            import shutil

            if is_deep_learning:
                # Eliminar carpeta completa del modelo de Deep Learning
                deep_models_dir = "saved_deep_models"
                model_folder = os.path.join(deep_models_dir, model_name)

                if os.path.exists(model_folder):
                    shutil.rmtree(model_folder)
                    print(f"✅ Modelo Deep Learning '{model_name}' eliminado")
                    print(f"   Carpeta eliminada: {model_folder}")
                    return True
                else:
                    print(f"⚠️ No se encontró la carpeta del modelo DL '{model_name}'")
                    return False
            else:
                # Eliminar carpeta completa del modelo tradicional
                model_folder = os.path.join(self.model_dir, model_name)

                if os.path.exists(model_folder):
                    shutil.rmtree(model_folder)
                    print(f"✅ Modelo tradicional '{model_name}' eliminado")
                    print(f"   Carpeta eliminada: {model_folder}")
                    return True
                else:
                    print(f"⚠️ No se encontró la carpeta del modelo '{model_name}'")
                    return False

        except Exception as e:
            print(f"❌ Error eliminando modelo '{model_name}': {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False
