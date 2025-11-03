import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import time
import json
from typing import List
from sentence_transformers import SentenceTransformer
import recommender
import json

# Use recommender's processed data dir for generated artefacts
DATA_DIR = recommender.DATA_DIR
VECTORIZER_PATH = getattr(recommender, 'VECTORIZER_PATH', os.path.join(DATA_DIR, 'tfidf_vectorizer.joblib'))
TFIDF_PATH = getattr(recommender, 'TFIDF_PATH', os.path.join(DATA_DIR, 'tfidf_model.joblib'))
EMBEDDINGS_PATH = getattr(recommender, 'EMBED_PATH', os.path.join(DATA_DIR, 'embeddings.npy'))

# Logs / survey files stored in processed data folder
LOGS_CSV = os.path.join(DATA_DIR, 'logs.csv')
SURVEY_CSV = os.path.join(DATA_DIR, 'survey_responses.csv')


def guardar_log(user_id, query_text, recommended_ids, scores, feedback=None, method=None):
    """Append an interaction row to LOGS_CSV.

    Columns: timestamp, user_id, query, recommended_ids (json), scores (json), feedback, method
    """
    row = {
        'timestamp': time.time(),
        'user_id': user_id,
        'query': query_text,
        'recommended_ids': json.dumps(recommended_ids),
        'scores': json.dumps(scores),
        'feedback': feedback,
        'method': method,
    }
    df_row = pd.DataFrame([row])
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LOGS_CSV):
        df_row.to_csv(LOGS_CSV, index=False)
    else:
        df_row.to_csv(LOGS_CSV, mode='a', header=False, index=False)


def guardar_encuesta(responses: dict):
    """Append survey response to SURVEY_CSV."""
    row = responses.copy()
    row['timestamp'] = time.time()
    df_row = pd.DataFrame([row])
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SURVEY_CSV):
        df_row.to_csv(SURVEY_CSV, index=False)
    else:
        df_row.to_csv(SURVEY_CSV, mode='a', header=False, index=False)

# Hardcoded admin password (solo para pruebas)
ADMIN_PASSWORD = "admin123"

st.set_page_config(page_title="Recechat", layout="wide")

# --- Inyectar CSS personalizado para mejorar la apariencia ---
CUSTOM_CSS = '''
<style>
    /* Container / layout */
    div[data-testid="stApp"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        max-width: 1100px;
        margin: 0 auto;
        padding: 18px 12px;
    }

    /* Recipe card */
    .rc-card {
        background: linear-gradient(180deg, #ffffff, #fbfbfb);
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 6px 18px rgba(22,27,29,0.06);
        margin-bottom: 12px;
    }
    .rc-title {
        font-weight: 700;
        font-size: 18px;
        color: #0f1724;
        margin-bottom: 6px;
    }
    .rc-score { color: #6b7280; font-weight: 600; font-size: 13px; margin-left: 8px; }
    .rc-ingredients { color: #374151; font-size: 14px; margin-bottom: 8px; }
    .rc-instructions summary { cursor: pointer; font-weight:600; color:#0f1724; }
    .rc-instructions div { margin-top:8px; color:#374151; font-size:14px; }

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] .css-1d391kg { padding-top: 6px; }

    /* Make like/dislike buttons more visible (affects raw button elements) */
    button {
        border-radius: 8px !important;
    }

    /* Responsive tweaks */
    @media (max-width: 600px) {
        div[data-testid="stApp"] { padding: 8px; }
        .rc-title { font-size: 16px; }
    }
</style>
'''

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Sidebar: configuración ---
st.sidebar.title("Settings")
data_path = st.sidebar.text_input("Dataset path (CSV)", getattr(recommender, 'CLEANED_CSV', os.path.join(DATA_DIR, 'cleaned_recipes.csv')))
method_pref = st.sidebar.selectbox("Preferred method", ["auto", "tfidf", "embeddings"])
k = st.sidebar.number_input("NNumber of recipes (k)", min_value=1, max_value=20, value=5)
force_recalc = st.sidebar.checkbox("Force TF-IDF recalculation", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("**Generated / Expected Files**")
st.sidebar.text(f"vectorizer: {VECTORIZER_PATH}")
st.sidebar.text(f"tfidf: {TFIDF_PATH}")
st.sidebar.text(f"embeddings: {EMBEDDINGS_PATH}")
st.sidebar.markdown("---")
st.sidebar.markdown("Quick Instructions:")
st.sidebar.write("Place cleaned_recipes.csv in the 'data' folder and restart the app. If you want to recalculate TF-IDF, check 'Force recalculation'.")

# --- Cargar dataset ---
st.title("Recechat")
try:
    # recommender already loads cleaned df at import time
    df = getattr(recommender, 'df', None)
    df_full = getattr(recommender, 'df_full', None)
    if df is None:
        # Fallback: try reading cleaned csv path
        df = pd.read_csv(data_path)
    st.success(f"Loading Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# --- Cargar o entrenar artefactos (TF-IDF y Embeddings) ---
with st.spinner("Charging (TF-IDF / Embeddings) ..."):
    # recommender module loads tfidf_vectorizer, tfidf_matrix, embeddings and embed_model
    try:
        vectorizer = getattr(recommender, 'tfidf_vectorizer', None)
        tfidf_matrix = getattr(recommender, 'tfidf_matrix', None)
        embeddings = getattr(recommender, 'embeddings', None)
        embed_model = getattr(recommender, 'embed_model', None)

        if vectorizer is not None and tfidf_matrix is not None:
            st.info("Loaded TF-IDF")
        else:
            st.warning("TF-IDF not available in recommender module.")

        if embeddings is not None:
            st.info(f"Loaded embeddings: {embeddings.shape}")
        else:
            st.warning("Embeddings not available.")

    except Exception as e:
        st.error(f"Error loading artifacts from recommender: {e}")
        vectorizer, tfidf_matrix, embeddings, embed_model = None, None, None, None


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
            # Use recommender.recomendar_recetas which returns list of tuples (index, title, score)
            resultados = recommender.recomendar_recetas(user_input, top_k=int(k))

            # Convert to the dict format expected by the UI
            recs = []
            for item in resultados:
                idx, title, score = item
                try:
                    rid = int(idx)
                except Exception:
                    rid = idx

                # Try to fetch ingredients/instructions from the original full df when available
                snippet = ''
                instructions = ''
                try:
                    if df_full is not None and rid in df_full.index:
                        snippet = df_full.loc[rid, 'Ingredients'] if 'Ingredients' in df_full.columns else ''
                        instructions = df_full.loc[rid, 'Instructions'] if 'Instructions' in df_full.columns else ''
                    elif 'Cleaned_Ingredients' in df.columns and rid in df.index:
                        snippet = df.loc[rid, 'Cleaned_Ingredients']
                except Exception:
                    pass

                recs.append({
                    'id': rid,
                    'Title': title,
                    'score': float(score),
                    'snippet_ingredientes': str(snippet),
                    'instructions': str(instructions)
                })
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