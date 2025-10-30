import os
import ast
import joblib
import numpy as np
import pandas as pd
import random
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import memory_manager as mm   
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprimir logs de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desactivar optimizaciones OneDNN


# ---------- Config ----------
CLEANED_CSV = 'cleaned_recipes.csv'
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K = 8
CACHE_INFO = 'cache_info.txt'
TFIDF_PATH = 'tfidf_model.joblib'
VECTORIZER_PATH = 'tfidf_vectorizer.joblib'
EMBED_PATH = 'embeddings.npy'
COMMON_ING_PATH = 'common_ingredients.txt'

# Utilidades para evitar recálculos innecesarios de embeddings y TF-IDF
def is_dataset_updated(csv_path, cache_info_path):
    if not os.path.exists(cache_info_path):
        return True
    with open(cache_info_path, 'r') as f:
        cached_date = f.read().strip()
    current_date = str(os.path.getmtime(csv_path))
    return cached_date != current_date

def save_common_ingredients(ingredients):
    with open(COMMON_ING_PATH, 'w') as f:
        f.write('\n'.join(sorted(set(ingredients))))

def load_common_ingredients():
    if not os.path.exists(COMMON_ING_PATH):
        base_common = ['salt', 'water', 'oil', 'sugar', 'egg', 'flour', 'milk', 'butter']
        save_common_ingredients(base_common)
        return base_common
    with open(COMMON_ING_PATH, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# ---------- Carga de datos ----------
df = pd.read_csv(CLEANED_CSV)
if df['Cleaned_Ingredients'].dtype == object and df['Cleaned_Ingredients'].str.startswith('[').any():
    try:
        df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].apply(lambda x: ' '.join(ast.literal_eval(x)))
    except Exception:
        df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].astype(str)
else:
    df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].astype(str)

corpus = df['Cleaned_Ingredients'].tolist()
titles = df['Title'].fillna('No title').tolist()

# ---------- Cache de modelos ya calculados ----------
need_recompute = is_dataset_updated(CLEANED_CSV, CACHE_INFO)
if not need_recompute and os.path.exists(EMBED_PATH) and os.path.exists(VECTORIZER_PATH):
    print("Cargando modelos y embeddings desde cache...")
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
    tfidf_matrix = joblib.load(TFIDF_PATH)
    embeddings = np.load(EMBED_PATH)
else:
    print("Recalculando TF-IDF y Embeddings...")
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=30000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    joblib.dump(tfidf_vectorizer, VECTORIZER_PATH)
    joblib.dump(tfidf_matrix, TFIDF_PATH)

    print("Cargando modelo de embeddings:", EMBED_MODEL_NAME)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = embed_model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
    np.save(EMBED_PATH, embeddings)

    with open(CACHE_INFO, 'w') as f:
        f.write(str(os.path.getmtime(CLEANED_CSV)))

print("Modelos cargados correctamente.")

# ---------- Estado global ----------
interaction_count = 0
common_ingredients = load_common_ingredients()

# ---------- Funciones principales ----------


def es_consulta_comida(text):
    palabras_clave = ['recipe', 'cook', 'food', 'ingredient', 'bake', 'meal', 'dish', 'soup', 'salad']
    return any(word in text.lower() for word in palabras_clave)

def recomendar_recetas_tfidf(query_text, top_k=TOP_K):
    q_vec = tfidf_vectorizer.transform([query_text])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(-sims)[:top_k]
    return [(int(idx), titles[idx], float(sims[idx])) for idx in top_idx]

def build_list_text(resultados, n=3):
    simple = []
    for idx, title, score in resultados[:n]:
        simple.append(f"{title} ({score:.2f})")
    return "; ".join(simple)

def mostrar_respuesta_amigable(resultados, query_text, offset=0):
    """Genera una respuesta textual con tono humano, pero pide a memory_manager por variantes."""
    global common_ingredients
    # build base textual response
    if not es_consulta_comida(query_text):
        base = random.choice([
            "Hmm... that doesn't sound like a recipe, but I can cook up an idea!",
            "That doesn't seem like food, but interesting question.",
            "If I had digital ingredients I'd make a bytes-salad!"
        ])
        # register template shown
        mm.register_interaction(query_text, base, "shown")
        # ask memory manager for alternative template suited for query
        tpl = mm.get_best_template_for_query(query_text)
        final = mm.render_template(tpl, list_text="")
        return final

    # If low similarity, suggest common ingredients
    if not resultados or resultados[0][2] < 0.1:
        sugeridas = ", ".join(random.sample(common_ingredients, min(5, len(common_ingredients))))
        base = f"No similar recipes found, maybe try these: {sugeridas}"
        mm.register_interaction(query_text, base, "shown")
        tpl = mm.get_best_template_for_query(query_text)
        final = mm.render_template(tpl, list_text=sugeridas)
        return final

    # normal case: create base listing
    list_text = build_list_text(resultados[offset:offset+3], n=3)
    base = "Here are some recipes you might like: {list_text}"
    # ask memory manager to pick/modify a template
    tpl = mm.get_best_template_for_query(query_text)
    final = mm.render_template(tpl, list_text=list_text)
    mm.register_interaction(query_text, final, "shown")
    return final

def mostrar_mas_recetas(resultados, start_idx=3):
    extra = resultados[start_idx:start_idx+3]
    if not extra:
        base = "I've shown all the closest recipes."
        mm.register_interaction("", base, "shown")
        return base
    list_text = build_list_text(extra, n=3)
    base = "Here are some other options: {list_text}"
    tpl = mm.get_best_template_for_query(list_text)
    final = mm.render_template(tpl, list_text=list_text)
    mm.register_interaction(list_text, final, "shown")
    return final

def actualizar_ingredientes_comunes(query_text):
    global common_ingredients
    ingredientes = re.findall(r'\b[a-zA-Z]+\b', query_text.lower())
    common_ingredients.extend(ingredientes)
    common_ingredients = list(set(common_ingredients))
    save_common_ingredients(common_ingredients)

def update_common_ingredients(user_input, accepted, path=COMMON_ING_PATH):
    """Actualiza los ingredientes comunes basados en consultas aceptadas."""
    if not accepted:
        return
    common = set(load_common_ingredients())
    for ing in re.findall(r'\b[a-zA-Z]+\b', user_input.lower()):
        if ing:
            common.add(ing)
    save_common_ingredients(common)

# ---------- Ciclo de evaluación (cada 30 interacciones) ----------
def evaluar_modelo_periodico():
    """
    Llamada que hace 2 cosas:
    - la evaluación ligera que ya tenías
    - y lanza memory_manager.evaluate_and_improve() para producir nuevas respuestas
    """
    print("\nPeriodic Evaluation: checking health of recommender and response memory...")
    # quick recommender check (same as earlier)
    muestras = random.sample(corpus, min(10, len(corpus)))
    scores = []
    for m in muestras:
        res = recomendar_recetas_tfidf(m)
        if res:
            scores.append(res[0][2])
    if scores:
        avg_score = np.mean(scores)
        print(f"Average similarity top-1: {avg_score:.3f}")
    else:
        print("No valid samples.")

    # now: automatic improvement of responses
    res = mm.evaluate_and_improve()
    print("MemoryManager:", res)

# ---------- Loop de interacción ----------
def interactuar_con_usuario(query_text):
    global interaction_count
    interaction_count += 1

    
    resultados = recomendar_recetas_tfidf(query_text)
    offset = 0
    respuesta = mostrar_respuesta_amigable(resultados, query_text, offset=offset)
    print("\n" + respuesta)

    # user loop
    while True:
        opcion = input("\n¿Did you like any recipe? (yes/no/more/exit): ").strip().lower()
        if opcion in ("yes", "y"):
            idx = input("Which number did you like?: ").strip()
            try:
                chosen = int(idx) - 1
                chosen_title = resultados[offset + chosen][1]
                print(f"Great choice! '{chosen_title}' 😋")
                mm.register_interaction(query_text, respuesta, "accepted")
                update_common_ingredients(query_text, accepted=True)
            except Exception:
                print("Invalid number.")
            break
        elif opcion in ("no", "n"):
            print("Ok, I'll try with other options.")
            mm.register_interaction(query_text, respuesta, "rejected")
            break
        elif opcion == "more":
            offset += 3
            more_text = mostrar_mas_recetas(resultados, start_idx=offset)
            print("\n" + more_text)
            mm.register_interaction(query_text, more_text, "more")
        elif opcion == "exit":
            mm.register_interaction(query_text, respuesta, "rejected")
            print("Goodbye 👋")
            return
        else:
            print("Please respond with: yes / no / more / exit")

    # every 30 interactions run periodic evaluation + memory improvement
    if interaction_count % 30 == 0:
        evaluar_modelo_periodico()

# ---------- Main ----------
if __name__ == "__main__":
    print("Starting recommender + memory manager. Type 'exit' to quit.")
    while True:
        user_input = input("\n¿ What ingredients do you have today? (o 'exit'): ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        interactuar_con_usuario(user_input)
