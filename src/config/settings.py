"""
Configuración centralizada del sistema
"""

import os
from pathlib import Path

class Settings:
    """Configuración global del sistema"""
    
    # Directorios base
    BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Directorios de modelos
    MODELS_DIR = BASE_DIR / 'saved_models'
    DEEP_MODELS_DIR = BASE_DIR / 'saved_deep_models'
    
    # Directorios de caché
    CACHE_DIR = BASE_DIR / 'cache'
    BERT_CACHE_DIR = DEEP_MODELS_DIR / 'bert_cache'
    
    @classmethod
    def ensure_directories(cls):
        """Asegura que existan todos los directorios necesarios"""
        directories = [
            cls.MODELS_DIR,
            cls.DEEP_MODELS_DIR,
            cls.CACHE_DIR,
            cls.BERT_CACHE_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        return True 