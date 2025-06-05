"""
Gestor centralizado de modelos para CV Classifier v2.0
Maneja el ciclo de vida completo de modelos ML y DL
"""

import os
import json
import datetime
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from src.config.settings import Settings


@dataclass
class ModelMetadata:
    """Metadatos estructurados para modelos"""
    name: str
    display_name: str
    model_type: str
    creation_date: str
    last_modified: str
    version: str = "1.0.0"
    is_deep_learning: bool = False
    professions: List[str] = None
    num_features: int = 0
    num_professions: int = 0
    accuracy: float = 0.0
    training_samples: int = 0
    test_samples: int = 0
    hyperparameters: Dict[str, Any] = None
    performance_metrics: Dict[str, float] = None
    tags: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.professions is None:
            self.professions = []
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.tags is None:
            self.tags = []


class ModelManager:
    """Gestor centralizado de modelos siguiendo patrones del sistema"""
    
    def __init__(self):
        self.models_dir = Settings.MODELS_DIR
        self.deep_models_dir = Settings.DEEP_MODELS_DIR
        
        # Asegurar que los directorios existan
        Settings.ensure_directories()
    
    def create_model_metadata(self, 
                            name: str,
                            model_type: str,
                            professions: List[str],
                            is_deep_learning: bool = False,
                            **kwargs) -> ModelMetadata:
        """Crea metadatos para un nuevo modelo siguiendo convenciones del sistema"""
        
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        display_name = kwargs.get('display_name', name.replace('_', ' ').title())
        
        metadata = ModelMetadata(
            name=name,
            display_name=display_name,
            model_type=model_type,
            creation_date=current_time,
            last_modified=current_time,
            is_deep_learning=is_deep_learning,
            professions=professions,
            num_professions=len(professions),
            **kwargs
        )
        
        return metadata
    
    def save_model_metadata(self, metadata: ModelMetadata) -> bool:
        """Guarda metadatos del modelo en formato JSON"""
        try:
            model_dir = self.deep_models_dir if metadata.is_deep_learning else self.models_dir
            metadata_path = model_dir / f"{metadata.name}_metadata.json"
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, indent=4, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"❌ Error guardando metadatos del modelo {metadata.name}: {e}")
            return False
    
    def load_model_metadata(self, model_name: str, is_deep_learning: bool = False) -> Optional[ModelMetadata]:
        """Carga metadatos de un modelo"""
        try:
            model_dir = self.deep_models_dir if is_deep_learning else self.models_dir
            metadata_path = model_dir / f"{model_name}_metadata.json"
            
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return ModelMetadata(**data)
            
        except Exception as e:
            print(f"❌ Error cargando metadatos del modelo {model_name}: {e}")
            return None
    
    def list_available_models(self, include_deep_learning: bool = True) -> List[ModelMetadata]:
        """Lista todos los modelos disponibles con sus metadatos"""
        models = []
        
        # Modelos tradicionales
        for metadata_file in self.models_dir.glob("*_metadata.json"):
            try:
                model_name = metadata_file.stem.replace('_metadata', '')
                metadata = self.load_model_metadata(model_name, is_deep_learning=False)
                if metadata:
                    models.append(metadata)
            except Exception as e:
                print(f"⚠️ Error cargando modelo {metadata_file}: {e}")
        
        # Modelos de Deep Learning
        if include_deep_learning:
            for metadata_file in self.deep_models_dir.glob("*_metadata.json"):
                try:
                    model_name = metadata_file.stem.replace('_metadata', '')
                    metadata = self.load_model_metadata(model_name, is_deep_learning=True)
                    if metadata:
                        models.append(metadata)
                except Exception as e:
                    print(f"⚠️ Error cargando modelo DL {metadata_file}: {e}")
        
        # Ordenar por fecha de creación (más recientes primero)
        models.sort(key=lambda x: x.creation_date, reverse=True)
        return models
    
    def delete_model(self, model_name: str, is_deep_learning: bool = False) -> bool:
        """Elimina un modelo y todos sus archivos asociados"""
        try:
            model_dir = self.deep_models_dir if is_deep_learning else self.models_dir
            
            # Archivos a eliminar
            files_to_delete = [
                model_dir / f"{model_name}_metadata.json",
                model_dir / f"{model_name}.pkl",  # Modelo tradicional
                model_dir / f"{model_name}_vectorizer.pkl",
                model_dir / f"{model_name}_encoder.pkl",
            ]
            
            # Para modelos de Deep Learning
            if is_deep_learning:
                files_to_delete.extend([
                    model_dir / f"{model_name}_model",  # Directorio del modelo TensorFlow
                    model_dir / f"{model_name}_tokenizer.pkl",
                    model_dir / f"{model_name}_bert_tokenizer",
                ])
            
            deleted_count = 0
            for file_path in files_to_delete:
                if file_path.exists():
                    if file_path.is_dir():
                        import shutil
                        shutil.rmtree(file_path)
                    else:
                        file_path.unlink()
                    deleted_count += 1
            
            print(f"✅ Modelo {model_name} eliminado ({deleted_count} archivos)")
            return True
            
        except Exception as e:
            print(f"❌ Error eliminando modelo {model_name}: {e}")
            return False
    
    def get_model_performance_summary(self, metadata: ModelMetadata) -> Dict[str, str]:
        """Genera resumen de rendimiento del modelo para mostrar en GUI"""
        summary = {
            'Tipo': metadata.model_type,
            'Precisión': f"{metadata.accuracy:.1%}" if metadata.accuracy > 0 else "N/A",
            'Profesiones': str(metadata.num_professions),
            'Muestras Entrenamiento': str(metadata.training_samples),
            'Características': str(metadata.num_features),
            'Creado': metadata.creation_date,
        }
        
        if metadata.is_deep_learning:
            summary['Tipo DL'] = metadata.model_type
            if 'epochs_trained' in metadata.hyperparameters:
                summary['Épocas'] = str(metadata.hyperparameters['epochs_trained'])
        
        return summary
    
    def validate_model_compatibility(self, metadata: ModelMetadata) -> Dict[str, bool]:
        """Valida la compatibilidad del modelo con el sistema actual"""
        validation = {
            'metadata_valid': True,
            'files_exist': True,
            'dependencies_available': True,
            'version_compatible': True
        }
        
        try:
            # Verificar archivos requeridos
            model_dir = self.deep_models_dir if metadata.is_deep_learning else self.models_dir
            required_files = [f"{metadata.name}_metadata.json"]
            
            if not metadata.is_deep_learning:
                required_files.extend([
                    f"{metadata.name}.pkl",
                    f"{metadata.name}_vectorizer.pkl",
                    f"{metadata.name}_encoder.pkl"
                ])
            
            for file_name in required_files:
                if not (model_dir / file_name).exists():
                    validation['files_exist'] = False
                    break
            
            # Verificar dependencias para Deep Learning
            if metadata.is_deep_learning:
                try:
                    import tensorflow
                    if metadata.model_type.lower() == 'bert':
                        import transformers
                except ImportError:
                    validation['dependencies_available'] = False
            
        except Exception as e:
            print(f"⚠️ Error validando modelo {metadata.name}: {e}")
            validation['metadata_valid'] = False
        
        return validation
