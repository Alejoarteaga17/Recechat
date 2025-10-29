"""
recommender.py
Persona B - Vectorización y Motor de Recomendación
Fase 3 & Fase 5 (parcial)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ast
import random
import joblib

# ---------- Config ----------
CLEANED_CSV = 'cleaned_recipes.csv'  # asumiendo que está en la misma carpeta
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'  # ligero y bueno para similitud semántica
N_SAMPLE_QUERIES = 200  # cuántas consultas sintéticas para evaluar
TOP_K = 5  # k para Precision@k / Recall@k
SHARE_THRESHOLD = 0.4  # umbral (fracción) para considerar 'relevante' por overlap de ingredientes

# ---------- Carga ----------
df = pd.read_csv(CLEANED_CSV)
# Nos aseguramos de tener una columna con ingredientes tokenizados/limpios (string)
# Si Cleaned_Ingredients está en formato lista guardada como string, intentar parsearla
if df['Cleaned_Ingredients'].dtype == object and df['Cleaned_Ingredients'].str.startswith('[').any():
    try:
        df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].apply(lambda x: ' '.join(ast.literal_eval(x)))
    except Exception:
        df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].astype(str)
else:
    df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].astype(str)

corpus = df['Cleaned_Ingredients'].tolist()
titles = df['Title'].fillna('No title').tolist()

# ---------- TF-IDF Vectorization ----------
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=30000)  # ajustar si es necesario
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)  # sparse matrix (n_recipes x features)


# ---------- Sentence-Transformers Embeddings ----------
print("Cargando modelo de embeddings:", EMBED_MODEL_NAME)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
embeddings = embed_model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)  # (n_recipes, dim)

# ---------- Funciones de recomendación ----------
def recomendar_recetas_tfidf(query_text, top_k=TOP_K):
    """
    Usa TF-IDF + coseno.
    query_text: string (ingredientes del usuario)
    devuelve: list de tuples (index, title, score)
    """
    q_vec = tfidf_vectorizer.transform([query_text])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(-sims)[:top_k]
    return [(int(idx), titles[idx], float(sims[idx])) for idx in top_idx]

def recomendar_recetas_embed(query_text, top_k=TOP_K):
    """
    Usa sentence-transformers embeddings + coseno.
    """
    q_emb = embed_model.encode([query_text], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings).flatten()
    top_idx = np.argsort(-sims)[:top_k]
    return [(int(idx), titles[idx], float(sims[idx])) for idx in top_idx]

# ---------- Funciones de evaluación práctica ----------
def ingredient_overlap_fraction(query_ingredients_list, recipe_ingredients_list):
    """
    Ambos argumentos son listas de tokens/ingredientes (strings).
    Retorna fracción de tokens del query que aparecen en la receta.
    """
    if not query_ingredients_list:
        return 0.0
    query_set = set([x.strip() for x in query_ingredients_list if x.strip()])
    recipe_set = set([x.strip() for x in recipe_ingredients_list if x.strip()])
    if len(query_set) == 0:
        return 0.0
    overlap = len(query_set & recipe_set)
    return overlap / len(query_set)

def parse_ingredient_string(s):
    """
    Convierte una cadena limpia (p. ej. "olive oil tomato onion") en lista de tokens simples.
    """
    toks = [t.strip() for t in s.split() if t.strip()]
    return toks

def evaluate_recommender(sample_queries, recommender_func, top_k=TOP_K, share_threshold=SHARE_THRESHOLD):
    """
    sample_queries: lista de strings (consultas)
    recommender_func: función que recibe query_text y devuelve top_k recomendaciones [(idx,title,score),...]
    Devuelve: dict con avg_cosine_similarity, avg_precision_at_k, avg_recall_at_k
    """
    precisions = []
    recalls = []
    cosine_avgs = []

    for q in sample_queries:
        recs = recommender_func(q, top_k=top_k)
        # avg cosine score among top_k
        cosine_avgs.append(np.mean([r[2] for r in recs]) if recs else 0.0)

        # calcular relevancia por overlap token
        query_toks = parse_ingredient_string(q)
        # relevancia: receta es 'relevante' si overlap_fraction >= share_threshold
        relevant_count = 0
        # para recall: necesitamos número de "relevantes totales" en corpus; aproximamos: contar cuántas recetas en corpus cumplen threshold
        total_relevants_in_corpus = 0
        for idx_row, ing_str in enumerate(corpus):
            recipe_toks = parse_ingredient_string(ing_str)
            frac = ingredient_overlap_fraction(query_toks, recipe_toks)
            if frac >= share_threshold:
                total_relevants_in_corpus += 1

        for (idx, title, score) in recs:
            recipe_toks = parse_ingredient_string(corpus[idx])
            frac = ingredient_overlap_fraction(query_toks, recipe_toks)
            if frac >= share_threshold:
                relevant_count += 1

        precision_k = relevant_count / top_k
        recall_k = (relevant_count / total_relevants_in_corpus) if total_relevants_in_corpus > 0 else 0.0

        precisions.append(precision_k)
        recalls.append(recall_k)

    return {
        'avg_cosine_topk': float(np.mean(cosine_avgs)),
        'avg_precision_at_k': float(np.mean(precisions)),
        'avg_recall_at_k': float(np.mean(recalls))
    }

# ---------- Generar consultas sintéticas (si no hay etiquetas) ----------
def generate_synthetic_queries(df, n=N_SAMPLE_QUERIES, min_tokens=2, max_tokens=6):
    """
    Construye consultas simuladas a partir de las listas de ingredientes existentes.
    Estrategia: tomar una receta al azar y muestrear una fracción de sus ingredientes como 'lo que tiene el usuario'.
    """
    queries = []
    for _ in range(n):
        row = df.sample(1).iloc[0]
        ing_str = row['Cleaned_Ingredients']
        toks = parse_ingredient_string(ing_str)
        if not toks:
            continue
        # sample between min_tokens and min(max_tokens, len(toks))
        k = random.randint(min_tokens, min(max_tokens, max(1, len(toks))))
        sampled = random.sample(toks, k) if len(toks) >= k else toks
        queries.append(' '.join(sampled))
    return queries

# ---------- Comparación TF-IDF vs Embeddings ----------
def compare_methods(sample_queries):
    print("Evaluando TF-IDF...")
    tfidf_res = evaluate_recommender(sample_queries, recomendar_recetas_tfidf)
    print("Evaluando Embeddings...")
    embed_res = evaluate_recommender(sample_queries, recomendar_recetas_embed)
    return tfidf_res, embed_res

# ---------- Rutina principal (ejecución) ----------
if __name__ == '__main__':
    print("Generando consultas sintéticas...")
    sample_queries = generate_synthetic_queries(df, n=N_SAMPLE_QUERIES)
    print(f"{len(sample_queries)} consultas generadas para evaluación.")

    tfidf_metrics, embed_metrics = compare_methods(sample_queries)

    print("\n--- RESULTADOS COMPARATIVOS ---")
    print("TF-IDF metrics:", tfidf_metrics)
    print("Embeddings metrics:", embed_metrics)

    # Ejemplos de prueba visual
    test_examples = [
        "oat milk banana almond cookie",    # ejemplo tipo desayuno
        "rice tomato chicken garlic",       # ejemplo almuerzo
        "flour sugar butter egg",           # repostería
    ]
    print("\n--- EJEMPLOS DE RECOMENDACIÓN (TF-IDF) ---")
    for q in test_examples:
        print("\nQuery:", q)
        for idx, title, score in recomendar_recetas_tfidf(q, top_k=5):
            print(f"  - [{idx}] {title} (score={score:.4f})")

    print("\n--- EJEMPLOS DE RECOMENDACIÓN (EMBEDDINGS) ---")
    for q in test_examples:
        print("\nQuery:", q)
        for idx, title, score in recomendar_recetas_embed(q, top_k=5):
            print(f"  - [{idx}] {title} (score={score:.4f})")

# --- MODO INTERACTIVO ---
    print("\n --- MODO INTERACTIVO ---")
    print("Escribe los ingredientes que tienes y te sugeriré recetas. Escribe 'salir' para terminar.")
    while True:
        query = input("\n Ingredientes: ")
        if query.lower() == "salir":
            print("👋 ¡Hasta luego!")
            break
        print("\n🔍 Recomendaciones basadas en embeddings:\n")
        for idx, title, score in recomendar_recetas_embed(query, top_k=5):
            print(f"   {title} (score={score:.3f})")