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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

MAX_MORE_REQUESTS = 2

CLEANED_CSV = 'cleaned_recipes.csv'
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K = 12
CACHE_INFO = 'cache_info.txt'
TFIDF_PATH = 'tfidf_model.joblib'
VECTORIZER_PATH = 'tfidf_vectorizer.joblib'
EMBED_PATH = 'embeddings.npy'
COMMON_ING_PATH = 'common_ingredients.txt'

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

df = pd.read_csv(CLEANED_CSV)
df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].astype(str)
corpus = df['Cleaned_Ingredients'].tolist()
titles = df['Title'].fillna('No title').tolist()

need_recompute = is_dataset_updated(CLEANED_CSV, CACHE_INFO)
if not need_recompute and os.path.exists(EMBED_PATH) and os.path.exists(VECTORIZER_PATH):
    print("Loading cached embeddings & TF-IDF...")
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
    tfidf_matrix = joblib.load(TFIDF_PATH)
    embeddings = np.load(EMBED_PATH)
else:
    print("Recomputing TF-IDF + Embeddings...")
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=25000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    joblib.dump(tfidf_vectorizer, VECTORIZER_PATH)
    joblib.dump(tfidf_matrix, TFIDF_PATH)

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = embed_model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
    np.save(EMBED_PATH, embeddings)

    with open(CACHE_INFO, 'w') as f:
        f.write(str(os.path.getmtime(CLEANED_CSV)))

print("Models ready ✅")

interaction_count = 0
common_ingredients = load_common_ingredients()

def recomendar_recetas_tfidf(query_text, top_k=TOP_K):
    q_vec = tfidf_vectorizer.transform([query_text])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    idx_top = np.argsort(-sims)[:top_k]
    return [(i, titles[i], float(sims[i])) for i in idx_top]

def format_recipes_for_display(resultados, offset=0, limit=3):
    formatted = []
    for pos, (_, title, score) in enumerate(resultados[offset:offset+limit], start=offset+1):
        formatted.append(f"{pos}. {title} ({score:.2f})")
    return formatted

def mostrar_respuesta_amigable(resultados, query_text, offset=0):
    if resultados and resultados[0][2] >= 0.05:
        lista = format_recipes_for_display(resultados, offset, 3)
        salida = "Here are some recipes you might like:\n" + "\n".join(lista)
        mm.register_interaction(query_text, salida, "shown")
        return salida

    # Si pobre similitud, sugerir ingredientes comunes
    sugeridas = ", ".join(random.sample(common_ingredients, min(5, len(common_ingredients))))
    salida = (
        "I couldn't find anything very similar 😅\n"
        f"Try other ingredients like: {sugeridas}"
    )
    mm.register_interaction(query_text, salida, "shown")
    return salida

def mostrar_mas_recetas(resultados, offset, more_count):
    if more_count >= MAX_MORE_REQUESTS:
        return "That’s all the best options for now! 😄", offset

    offset += 3
    lista = format_recipes_for_display(resultados, offset, 3)

    if not lista:
        return "No more recipes available!", offset

    tpl = mm.get_best_template_for_query("more")
    salida = mm.render_template(tpl, list="\n".join(lista))
    mm.register_interaction("more", salida, "more")
    return salida, offset

def evaluar_modelo_periodico():
    """
    Evaluación automática del recomendador y mejora automática del generador de plantillas.
    Se ejecuta cada 30 interacciones.
    """
    print("\n🔍 Running periodic model evaluation...")

    try:
        # — Comprobamos cómo está funcionando el recomendador —
        muestras = random.sample(corpus, min(10, len(corpus)))
        scores = []
        for m in muestras:
            res = recomendar_recetas_tfidf(m, top_k=3)
            if res:
                scores.append(res[0][2])

        if scores:
            avg_score = np.mean(scores)
            print(f"📊 Avg similarity top-1: {avg_score:.3f}")
        else:
            print("⚠️ Not enough samples for evaluation.")

        # ✅ Esto hace que el modelo aprenda nuevas respuestas automáticamente
        result = mm.evaluate_and_improve()
        print("🧠 Memory Manager:", result)

    except Exception as e:
        print("⚠️ Evaluation error:", e)


# ✅ MAIN INTERACTION LOOP ✅
def interactuar_con_usuario(query_text):
    global interaction_count
    interaction_count += 1

    more_count = 0
    offset = 0

    resultados = recomendar_recetas_tfidf(query_text)
    respuesta = mostrar_respuesta_amigable(resultados, query_text, offset)
    print("\n" + respuesta)

    while True:
        opcion = input("\nDid you like any recipe? (yes/no/more/exit): ").strip().lower()

        if opcion in ("yes", "y"):
            idx = input("Which number did you like?: ").strip()
            if not idx.isdigit():
                print("Please enter a valid number.")
                continue
            idx = int(idx) - 1
            if idx < 0 or idx >= len(resultados):
                print("That number is not in the list.")
                continue
            chosen_title = resultados[idx][1]
            print(f"Great choice! '{chosen_title}' 😋")
            mm.register_interaction(query_text, respuesta, "accepted")
            break

        elif opcion in ("no", "n"):
            print("I'll keep improving!")
            mm.register_interaction(query_text, respuesta, "rejected")
            break

        elif opcion == "more":
            more_response, offset = mostrar_mas_recetas(resultados, offset, more_count)
            more_count += 1
            print("\n" + more_response)

        elif opcion == "exit":
            print("Goodbye 👋")
            return

        else:
            print("Options: yes / no / more / exit")

    # ✅ Evaluación automática cada 30 interacciones
    if interaction_count % 30 == 0:
        evaluar_modelo_periodico()


if __name__ == "__main__":
    print("🍳 Recipe AI Recommender (type 'exit' to stop)")
    while True:
        user_input = input("\nWhat ingredients do you have today? ").strip()
        if user_input.lower() == "exit":
            print("Goodbye! 👋")
            break
        interactuar_con_usuario(user_input)
