# utils.py
"""
Funciones auxiliares para la app Streamlit de recomendaciones de recetas.
- Cargar dataset
- Preprocesamiento simple de texto
- Cargar o entrenar TF-IDF
- Cargar embeddings (si existen)
- Calcular similitud y recomendar
- Guardar logs y encuestas
"""

import os
import time
import csv
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from scipy import sparse
import nltk
import re

# Si no has descargado punkt: descomenta la siguiente línea (solo la primera vez)
# nltk.download('punkt')

# === Ajustar aquí si tu CSV tiene otros nombres de columna ===
COLUMN_ID = "id"               # ID esperado. Si no existe, se genera.
COLUMN_TITLE = "Title"         # nombre del título
COLUMN_CLEANED_ING = "Cleaned_Ingredients"  # ingredientes ya preprocesados
COLUMN_INSTRUCTIONS = "Instructions"        # instrucciones
COLUMN_CATEGORY = "Category"   # opcional
COLUMN_TAGS = "Tags"           # opcional
# =========================================================

# Rutas por defecto
DATA_DIR = "data"
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "cleaned_recipes.csv")
VECTORIZER_PATH = os.path.join(DATA_DIR, "vectorizer.pkl")
TFIDF_PATH = os.path.join(DATA_DIR, "tfidf.npz")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
EMB_IDS_PATH = os.path.join(DATA_DIR, "emb_ids.npy")
LOGS_CSV = os.path.join(DATA_DIR, "logs.csv")
SURVEY_CSV = os.path.join(DATA_DIR, "survey_responses.csv")

def cargar_dataset(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Carga el CSV y asegura que tenga las columnas requeridas."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Archivo no encontrado en {path}. Coloca cleaned_recipes.csv allí o cambia la ruta.")
    df = pd.read_csv(path)
    # Normalizar nombres de columnas si necesario (intentar bajar a los esperados)
    # Si columna ID no existe, crear una basada en el índice
    if COLUMN_ID not in df.columns:
        df[COLUMN_ID] = df.index.astype(int)
    # Asegurarse de columnas esperadas existan; si no, lanzar aviso
    missing = []
    for col in [COLUMN_TITLE, COLUMN_CLEANED_ING, COLUMN_INSTRUCTIONS]:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise KeyError(f"Faltan columnas esperadas en el CSV: {missing}. Edita los nombres en utils.py si difieren.")
    # Rellenar NaNs en columnas textuales con string vacío
    for col in [COLUMN_TITLE, COLUMN_CLEANED_ING, COLUMN_INSTRUCTIONS]:
        df[col] = df[col].fillna("").astype(str)
    return df

# Preprocesamiento simple: limpieza mínima (opcional ampliar con lematización)
def preprocesar_texto(text: str) -> str:
    text = text.lower()
    # eliminar contenido entre corchetes, parentesis
    text = re.sub(r"\[.*?\]|\(.*?\)", " ", text)
    # eliminar caracteres no alfanum (preservar comas y espacios)
    text = re.sub(r"[^a-z0-9,áéíóúüñ\s]", " ", text)
    # normalizar espacios
    text = re.sub(r"\s+", " ", text).strip()
    return text

def cargar_o_entrenar_tfidf(df: pd.DataFrame,
                           text_col: str = COLUMN_CLEANED_ING,
                           vectorizer_path: str = VECTORIZER_PATH,
                           tfidf_path: str = TFIDF_PATH,
                           force_retrain: bool = False) -> Tuple[TfidfVectorizer, sparse.spmatrix]:
    """
    Carga vectorizer.pkl y tfidf.npz si existen. Si no, entrena TF-IDF sobre text_col y guarda artefactos.
    Retorna (vectorizer, tfidf_matrix)
    """
    # Si existe y no forzar recálculo -> cargar
    if (os.path.exists(vectorizer_path) and os.path.exists(tfidf_path)) and not force_retrain:
        vectorizer = joblib.load(vectorizer_path)
        tfidf = sparse.load_npz(tfidf_path)
        return vectorizer, tfidf

    # Entrenar TF-IDF
    texts = df[text_col].fillna("").astype(str).apply(preprocesar_texto).tolist()
    # Parámetros básicos; puedes ajustarlos
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
    tfidf = vectorizer.fit_transform(texts)
    # Guardar
    joblib.dump(vectorizer, vectorizer_path)
    sparse.save_npz(tfidf_path, tfidf)
    return vectorizer, tfidf

def cargar_embeddings(emb_path: str = EMBEDDINGS_PATH, emb_ids_path: str = EMB_IDS_PATH) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Carga embeddings.npy y emb_ids.npy si existen. Retorna (embeddings, ids) o None."""
    if os.path.exists(emb_path) and os.path.exists(emb_ids_path):
        emb = np.load(emb_path)
        ids = np.load(emb_ids_path)
        return emb, ids
    return None

def calcular_similitud_tfidf(query: str, vectorizer: TfidfVectorizer, tfidf_matrix, top_k: int = 5) -> List[Tuple[int, float]]:
    """Calcula similitud coseno entre query y la matriz TF-IDF. Retorna lista de (index, score)."""
    q = preprocesar_texto(query)
    q_vec = vectorizer.transform([q])
    # similitud coseno
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]

def calcular_similitud_embeddings(query: str, embeddings: np.ndarray, emb_ids: np.ndarray,
                                  embed_model_func, top_k: int = 5) -> List[Tuple[int, float]]:
    """
    Si tienes embeddings precomputados y una función para crear embedding del query,
    calcula similitud coseno y devuelve (id_index, score). emb_ids sirve para mapear índices del array.
    embed_model_func debe aceptar una lista de strings y retornar un numpy array (n, dim).
    """
    q_emb = embed_model_func([query])  # shape (1, dim)
    # normalizar
    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    q_norm = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    scores = np.dot(emb_norm, q_norm.T).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(emb_ids[i]), float(scores[i])) for i in top_idx]

def recomendar_recetas(ingredientes_usuario: str,
                        df: pd.DataFrame,
                        method: str = 'auto',
                        k: int = 5,
                        vectorizer: TfidfVectorizer = None,
                        tfidf_matrix = None,
                        embeddings_tuple: Tuple[np.ndarray, np.ndarray] = None,
                        embed_model_func = None) -> List[Dict]:
    """
    Interfaz principal para recomendar recetas.
    - method: 'auto', 'tfidf', 'embeddings'
    - embeddings_tuple: (embeddings_array, emb_ids_array) si existen
    - embed_model_func: función para convertir query -> embedding (opcional)
    Retorna lista de dicts: {id, Title, score, snippet_ingredientes, instructions}
    """
    # elegir método
    if method == 'auto':
        if embeddings_tuple is not None and embed_model_func is not None:
            selected = 'embeddings'
        else:
            selected = 'tfidf'
    else:
        selected = method

    results = []
    if selected == 'embeddings':
        emb, emb_ids = embeddings_tuple
        if embed_model_func is None:
            raise ValueError("No se proporcionó función para crear embeddings del query.")
        # calcular similitud
        top = calcular_similitud_embeddings(ingredientes_usuario, emb, emb_ids, embed_model_func, top_k=k)
        for idx, score in top:
            row = df[df[COLUMN_ID] == idx].iloc[0]
            results.append({
                "id": int(row[COLUMN_ID]),
                "Title": row[COLUMN_TITLE],
                "score": float(score),
                "snippet_ingredientes": row[COLUMN_CLEANED_ING][:300],
                "instructions": row[COLUMN_INSTRUCTIONS][:500]
            })
    else:  # tfidf
        if vectorizer is None or tfidf_matrix is None:
            raise ValueError("TF-IDF no cargado. Ejecuta cargar_o_entrenar_tfidf primero.")
        top = calcular_similitud_tfidf(ingredientes_usuario, vectorizer, tfidf_matrix, top_k=k)
        for idx_rel, score in top:
            # idx_rel corresponde al índice en df (posicional). Si tu df no está ordenado, confirma
            row = df.iloc[idx_rel]
            results.append({
                "id": int(row[COLUMN_ID]),
                "Title": row[COLUMN_TITLE],
                "score": float(score),
                "snippet_ingredientes": row[COLUMN_CLEANED_ING][:300],
                "instructions": row[COLUMN_INSTRUCTIONS][:500]
            })
    return results

def guardar_log(timestamp: Optional[str], user_input: str, recommended_ids: List[int], scores: List[float], feedback: Optional[str], method: str, path: str = LOGS_CSV):
    """Guardar una fila de log en CSV (anexa)."""
    header = ["timestamp", "user_input", "recommended_ids", "scores", "feedback", "method"]
    row = {
        "timestamp": timestamp or datetime.utcnow().isoformat(),
        "user_input": user_input,
        "recommended_ids": json.dumps(recommended_ids),
        "scores": json.dumps(scores),
        "feedback": feedback or "",
        "method": method
    }
    file_exists = os.path.exists(path)
    with open(path, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def guardar_encuesta(responses: dict, path: str = SURVEY_CSV):
    """
    Guardar respuestas de encuesta (formulario). responses es dict de preguntas->respuesta.
    """
    header = ["timestamp"] + list(responses.keys())
    row = {"timestamp": datetime.utcnow().isoformat()}
    row.update(responses)
    file_exists = os.path.exists(path)
    with open(path, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
