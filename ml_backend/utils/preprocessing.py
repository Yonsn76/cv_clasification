"""
Funciones para preprocesamiento de datos.
Incluye métodos para limpieza, transformación y optimización de datos, especialmente de PDFs.
"""

import pandas as pd
import numpy as np
import PyPDF2
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder

def extract_text_from_pdf(pdf_path):
    """
    Extrae texto de un archivo PDF.
    
    Args:
        pdf_path (str): Ruta al archivo PDF
        
    Returns:
        str: Texto extraído del PDF
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        print(f"Error al extraer texto del PDF: {e}")
        return ""

def clean_text_data(text):
    """
    Limpia texto extraído de PDFs, eliminando caracteres especiales y normalizando el formato.
    
    Args:
        text (str): Texto a limpiar
        
    Returns:
        str: Texto limpio
    """
    # Eliminar caracteres especiales y múltiples espacios
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    # Convertir a minúsculas para normalización
    text = text.lower().strip()
    return text

def optimize_data(data, is_text=False):
    """
    Optimiza los datos antes del entrenamiento.
    Si los datos provienen de texto (PDF), realiza limpieza adicional.
    
    Args:
        data: Datos a optimizar (puede ser str si es texto o DataFrame si ya está estructurado)
        is_text (bool): Indica si los datos son texto sin procesar
        
    Returns:
        DataFrame: Datos optimizados en formato tabular
    """
    if is_text:
        # Si es texto, limpiar y convertir a un formato estructurado
        cleaned_text = clean_text_data(data)
        # Aquí se puede implementar lógica para estructurar el texto en un DataFrame
        # Por ejemplo, asumir que el texto tiene un formato específico y extraer características
        # Esto es solo un placeholder, debe ajustarse según el formato real de los datos
        data_dict = {"texto": [cleaned_text]}
        df = pd.DataFrame(data_dict)
    else:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    
    # Eliminar valores nulos
    df = df.dropna()
    
    # Convertir columnas categóricas a numéricas si las hay
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
    
    return df

def scale_features(X):
    """
    Escala las características para que tengan media 0 y varianza 1.
    
    Args:
        X: Datos de características (DataFrame o array)
        
    Returns:
        Array: Características escaladas
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)
