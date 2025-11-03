# app.py
"""
Streamlit app: Chatbot de recomendaciones de recetas usando TF-IDF o Embeddings.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import time
import json
from typing import List
from sentence_transformers import SentenceTransformer

from utils import (
    cargar_dataset,
    cargar_o_entrenar_tfidf,
    cargar_embeddings,
    recomendar_recetas,
    preprocesar_texto,
    guardar_log,
    guardar_encuesta,
    DEFAULT_DATA_PATH,
    VECTORIZER_PATH,
    TFIDF_PATH,
    EMBEDDINGS_PATH,
    LOGS_CSV,
    SURVEY_CSV
)

# Hardcoded admin password (solo para pruebas)
ADMIN_PASSWORD = "admin123"

st.set_page_config(page_title="Chatbot Recetas (Streamlit)", layout="wide")

# --- Sidebar: configuración ---
st.sidebar.title("Configuración")
data_path = st.sidebar.text_input("Ruta dataset (CSV)", DEFAULT_DATA_PATH)
method_pref = st.sidebar.selectbox("Método preferido", ["auto", "tfidf", "embeddings"])
k = st.sidebar.number_input("Número de recetas (k)", min_value=1, max_value=20, value=5)
force_recalc = st.sidebar.checkbox("Forzar recálculo TF-IDF", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("**Archivos generados / esperados**")
st.sidebar.text(f"vectorizer: {VECTORIZER_PATH}")
st.sidebar.text(f"tfidf: {TFIDF_PATH}")
st.sidebar.text(f"embeddings: {EMBEDDINGS_PATH}")
st.sidebar.markdown("---")
st.sidebar.markdown("Instrucciones rápidas:")
st.sidebar.write("Coloca cleaned_recipes.csv en la carpeta 'data' y reinicia la app. Si quieres recalcular TF-IDF, marca 'Forzar recálculo'.")

# --- Cargar dataset ---
st.title("Chatbot de recetas — Streamlit")
try:
    df = cargar_dataset(data_path)
    st.success(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
except Exception as e:
    st.error(f"No se pudo cargar dataset: {e}")
    st.stop()

# --- Cargar o entrenar artefactos (TF-IDF y Embeddings) ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

with st.spinner("Cargando artefactos (TF-IDF / Embeddings) ..."):
    try:
        vectorizer, tfidf_matrix = cargar_o_entrenar_tfidf(df, force_retrain=force_recalc)
    except Exception as e:
        st.error(f"Error TF-IDF: {e}")
        vectorizer, tfidf_matrix = None, None

    embeddings_loaded = cargar_embeddings()
    if embeddings_loaded is not None:
        embeddings, emb_ids = embeddings_loaded
        st.info(f"Embeddings cargados: {embeddings.shape}")
        embed_model = load_embedding_model()
    else:
        embeddings, emb_ids = None, None
        embed_model = None
        st.warning("No se encontraron embeddings precomputados. Usando TF-IDF si está disponible.")

def embed_query_func(texts: List[str]):
    if embed_model is None:
        raise RuntimeError("El modelo de embeddings no está cargado.")
    return embed_model.encode(texts, convert_to_numpy=True)

# --- Chat UI ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.subheader("Chat - Busca recetas por ingredientes")
col1, col2 = st.columns([3,1])

with col1:
    user_input = st.text_area("Ingresa ingredientes o una descripción (lenguaje natural):", height=120)
    send = st.button("Enviar")
with col2:
    st.write("Controles rápidos")
    if st.button("Limpiar chat"):
        st.session_state.chat_history = []

if send and user_input.strip():
    method_to_use = method_pref
    if method_pref == 'auto':
        method_to_use = 'embeddings' if embeddings is not None else 'tfidf'
    
    try:
        with st.spinner("Buscando recomendaciones..."):
            recs = recomendar_recetas(user_input, df,
                                      method=method_to_use,
                                      k=int(k),
                                      vectorizer=vectorizer,
                                      tfidf_matrix=tfidf_matrix,
                                      embeddings_tuple=(embeddings, emb_ids) if embeddings is not None else None,
                                      embed_model_func=(embed_query_func if embeddings is not None else None))
        st.session_state.chat_history.append((user_input, recs, method_to_use))
        
        try:
            recommended_ids = [r["id"] for r in recs]
            scores = [r["score"] for r in recs]
            guardar_log(None, user_input, recommended_ids, scores, feedback=None, method=method_to_use)
        except Exception as e:
            st.warning(f"No se pudo guardar log: {e}")
            
    except Exception as e:
        st.error(f"Error al generar recomendaciones: {e}")
        recs = []


# Mostrar historial
for i, (u, recs, method_used) in enumerate(reversed(st.session_state.chat_history)):
    st.markdown(f"**Usuario:** {u}")
    if not recs:
        st.info("No se encontraron recomendaciones.")
        continue
    
    for r in recs:
        st.markdown(f"- **{r['Title']}**  (score: {r['score']:.4f})")
        st.markdown(f"  - Ingredientes: {r['snippet_ingredientes']}")
        with st.expander("Ver instrucciones"):
            st.write(r['instructions'])
        
        fb_col1, fb_col2 = st.columns([1,5])
        with fb_col1:
            if st.button(f"👍 Like {r['id']}_{i}", key=f"like_{r['id']}_{i}"):
                guardar_log(None, u, [r['id']], [r['score']], feedback="like", method=method_used)
                st.success("Gracias por tu feedback 👍")
        with fb_col2:
            if st.button(f"👎 Dislike {r['id']}_{i}", key=f"dis_{r['id']}_{i}"):
                guardar_log(None, u, [r['id']], [r['score']], feedback="dislike", method=method_used)
                st.warning("Feedback registrado 👎")
    st.markdown("---")

# --- Sección Evaluación (FASE 5 parcial) ---
st.subheader("Evaluación — encuesta rápida")
with st.form("survey_form"):
    st.write("Por favor evalúa la experiencia (1=Muy en desacuerdo, 5=Muy de acuerdo)")
    clarity = st.slider("Claridad de las recomendaciones", 1, 5, 4)
    usefulness = st.slider("Utilidad de las recomendaciones", 1, 5, 4)
    ease = st.slider("Facilidad de uso de la interfaz", 1, 5, 4)
    satisfaction = st.slider("Satisfacción general", 1, 5, 4)
    comments = st.text_area("Comentarios adicionales (opcional)")
    consent = st.checkbox("Consiento que mis respuestas sean usadas para mejorar el sistema", value=True)
    submitted = st.form_submit_button("Enviar evaluación")
    if submitted:
        if not consent:
            st.error("El consentimiento es requerido para guardar la encuesta.")
        else:
            responses = {
                "clarity": clarity,
                "usefulness": usefulness,
                "ease": ease,
                "satisfaction": satisfaction,
                "comments": comments,
                "consent": consent
            }
            guardar_encuesta(responses)
            st.success("Gracias por tu evaluación. ❤️")

# --- Admin (protección simple) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Admin (solo pruebas)")
admin_pw = st.sidebar.text_input("Password admin", type="password")
if admin_pw == ADMIN_PASSWORD:
    st.sidebar.success("Acceso admin concedido")
    st.sidebar.markdown("**Métricas**")
    try:
        logs_df = pd.read_csv(LOGS_CSV) if os.path.exists(LOGS_CSV) else pd.DataFrame()
        survey_df = pd.read_csv(SURVEY_CSV) if os.path.exists(SURVEY_CSV) else pd.DataFrame()
        
        st.sidebar.markdown(f"- Total interacciones: {len(logs_df)}")
        if not survey_df.empty:
            st.sidebar.markdown(f"- Promedio satisfacción: {survey_df['satisfaction'].astype(float).mean():.2f}")
            
        if not logs_df.empty:
            logs_df['recommended_ids'] = logs_df['recommended_ids'].fillna("[]")
            logs_df['recommended_ids_list'] = logs_df['recommended_ids'].apply(lambda s: json.loads(s) if isinstance(s, str) and s.startswith('[') else [])
            all_ids = sum(logs_df['recommended_ids_list'].tolist(), [])
            from collections import Counter
            top = Counter(all_ids).most_common(10)
            st.sidebar.markdown("Top recetas por impresiones:")
            for t in top:
                st.sidebar.markdown(f"  - ID {t[0]}: {t[1]} veces")
        
        if not logs_df.empty:
            st.sidebar.markdown("Descargar logs:")
            with open(LOGS_CSV, "rb") as fp:
                st.sidebar.download_button("Descargar logs.csv", data=fp, file_name="logs.csv")
        if not survey_df.empty:
            st.sidebar.markdown("Descargar encuestas:")
            with open(SURVEY_CSV, "rb") as fp:
                st.sidebar.download_button("Descargar survey_responses.csv", data=fp, file_name="survey_responses.csv")

    except Exception as e:
        st.sidebar.error(f"Error admin: {e}")
elif admin_pw:
    st.sidebar.error("Password incorrecta")

# --- Mensajes de ayuda ---
st.markdown("**Ayuda / Tips**")
st.write("""
- Ingresa ingredientes separados por comas o en lenguaje natural, por ejemplo: "pollo, ajo, limón, romero".
- Si no encuentras resultados buenos, intenta usar sinónimos o describir el plato (ej. "pollo asado con verduras").
- Para usar embeddings, asegúrate que los archivos `embeddings.npy` y `emb_ids.npy` estén en la carpeta `data`.
- Para FORZAR recálculo de TF-IDF marca la opción en la sidebar y recarga la app.
""")