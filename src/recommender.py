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
import json
import nltk
from nltk.corpus import stopwords
import faiss


nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

MAX_MORE_REQUESTS = 2

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# Use processed data folder for all data files
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
CLEANED_CSV = os.path.join(DATA_DIR, 'cleaned_recipes.csv')
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K = 12
CACHE_INFO = os.path.join(DATA_DIR, 'cache_info.txt')
TFIDF_PATH = os.path.join(DATA_DIR, 'tfidf_model.joblib')
VECTORIZER_PATH = os.path.join(DATA_DIR, 'tfidf_vectorizer.joblib')
EMBED_PATH = os.path.join(DATA_DIR, 'embeddings.npy')
COMMON_ING_PATH = os.path.join(DATA_DIR, 'common_ingredients.txt')
FAISS_INDEX_PATH = os.path.join(DATA_DIR, 'faiss_index.index')

# Load ingredient synonyms mapping (for normalization)
SYN_PATH = os.path.join(DATA_DIR, 'synonyms.json')
try:
    with open(SYN_PATH, 'r', encoding='utf-8') as _f:
        INGREDIENT_SYNONYMS = json.load(_f)
except Exception:
    INGREDIENT_SYNONYMS = {}


def normalize_query(text):
    """Replace ingredient variants in text using INGREDIENT_SYNONYMS mapping."""
    if not text:
        return text
    txt = text.lower()
    for k in sorted(INGREDIENT_SYNONYMS.keys(), key=lambda x: -len(x)):
        v = INGREDIENT_SYNONYMS[k]
        txt = re.sub(r'\b' + re.escape(k.lower()) + r'\b', v.lower(), txt)
    return txt

#  Load Full Original Data for User Display
# Locate original full dataset: prefer Recechat/ then parent
candidate_full_1 = os.path.join(BASE_DIR, 'Food Ingredients and Recipe Dataset.csv')
candidate_full_2 = os.path.join(BASE_DIR, '..', 'Food Ingredients and Recipe Dataset.csv')
if os.path.exists(candidate_full_1):
    FULL_CSV = candidate_full_1
elif os.path.exists(candidate_full_2):
    FULL_CSV = candidate_full_2
else:
    FULL_CSV = 'Food Ingredients and Recipe Dataset.csv'  # fallback to cwd lookup

df_full = pd.read_csv(FULL_CSV)

# Ensure fields are text
df_full['Ingredients'] = df_full['Ingredients'].astype(str)
df_full['Instructions'] = df_full['Instructions'].astype(str)
df_full['Title'] = df_full['Title'].astype(str)


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

# Cargar o crear el modelo de embeddings
print("Loading SentenceTransformer model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

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

    embeddings = embed_model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
    np.save(EMBED_PATH, embeddings)

    with open(CACHE_INFO, 'w') as f:
        f.write(str(os.path.getmtime(CLEANED_CSV)))

print("Models ready ✅")

interaction_count = 0
common_ingredients = load_common_ingredients()


def load_or_create_faiss_index(embeddings, dim):
    """Carga o crea el índice FAISS para búsquedas rápidas."""
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
            print("✅ FAISS index loaded from cache.")
            return index
        except Exception as e:
            print(f"⚠️ Error loading FAISS index: {e}, rebuilding...")

    print("🔧 Building FAISS index (cosine-normalized)...")
    # Normalizamos los embeddings para usar producto interno como coseno
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings) # type: ignore
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("✅ FAISS index created and saved.")
    return index


# Crear o cargar índice FAISS al iniciar
faiss_index = load_or_create_faiss_index(embeddings, embeddings.shape[1])


def get_embeddings_similarity(query_text, top_k=TOP_K):
    """Calculate semantic similarity using FAISS (fallback to cosine if needed)."""
    try:
        query_embedding = embed_model.encode([query_text], show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        D, I = faiss_index.search(query_embedding, top_k) # type: ignore
        sims = np.zeros(len(embeddings))
        sims[I[0]] = D[0]
        return sims
    except Exception as e:
        print(f"⚠️ FAISS search failed, fallback to cosine: {e}")
        query_embedding = embed_model.encode([query_text], show_progress_bar=False, convert_to_numpy=True)
        sims = cosine_similarity(query_embedding, embeddings).flatten()
        return sims


def calcular_peso_adaptativo(query_text):
    """Determina el peso TF-IDF según tipo y complejidad del texto."""
    text = query_text.lower().strip()
    tokens = re.findall(r'\w+', text)
    n_tokens = len(tokens)
    
    # Heurística por separadores: lista corta = más TF-IDF
    if re.search(r'[,\n;]', text):
        base_weight = 0.75  # más TF-IDF
    else:
        base_weight = 0.5   # neutro

    # Ajuste por longitud
    if n_tokens <= 4:
        base_weight += 0.15  # corto → TF-IDF más peso
    elif n_tokens >= 10:
        base_weight -= 0.15  # largo → Embeddings más peso

    # Ajuste por diversidad léxica
    unique_tokens = len(set(tokens))
    if unique_tokens / (n_tokens + 1e-5) > 0.8:
        base_weight -= 0.05  # mucha diversidad → embeddings

    # Asegurar rango válido
    return float(np.clip(base_weight, 0.2, 0.8))

def recomendar_recetas(query_text, top_k=TOP_K, w_tfidf=None):
    """
    Recommend recipes using both TF-IDF and semantic embeddings.
    The TF-IDF weight is adaptive if not provided.
    """
    norm_query = normalize_query(query_text)

    if w_tfidf is None:
        w_tfidf = calcular_peso_adaptativo(norm_query)

    q_vec = tfidf_vectorizer.transform([norm_query])
    sims_tfidf = cosine_similarity(q_vec, tfidf_matrix).flatten()
    sims_emb = get_embeddings_similarity(norm_query)

    sims = w_tfidf * sims_tfidf + (1 - w_tfidf) * sims_emb

    idx_top = np.argsort(-sims)[:top_k]
    return [(i, titles[i], float(sims[i]), w_tfidf) for i in idx_top]

def format_recipes_for_display(resultados, offset=0, limit=3):
    formatted = []
    for pos, item in enumerate(resultados[offset:offset+limit], start=offset+1):
        # Handle both 3-tuple and 4-tuple formats (with optional w_tfidf)
        if len(item) == 4:
            _, title, score, _ = item
        else:
            _, title, score = item
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
            res = recomendar_recetas(m, top_k=3)
            if res:
                scores.append(res[0][2])

        if scores:
            avg_score = np.mean(scores)
            print(f"📊 Avg similarity top-1: {avg_score:.3f}")
        else:
            print("⚠️ Not enough samples for evaluation.")

        #  Esto hace que el modelo aprenda nuevas respuestas automáticamente
        result = mm.evaluate_and_improve()
        print("🧠 Memory Manager:", result)

    except Exception as e:
        print("⚠️ Evaluation error:", e)

def clean_ingredient(text):
    """Clean and normalize an ingredient text for comparison."""
    # Apply synonyms normalization first
    try:
        text = normalize_query(text)
    except Exception:
        pass
    # Remove amounts, numbers and parentheses content
    text = re.sub(r'\([^)]*\)', '', text)  # remove (content)
    text = re.sub(r'\d+(/\d+)?', '', text)  # remove numbers like 1, 1/2
    text = re.sub(r'(cup|cups|tablespoon|tablespoons|teaspoon|teaspoons|tbsp|tsp|oz|ounce|ounces|gram|grams|kg|ml|g)\s*', '', text, flags=re.IGNORECASE)
    
    # Basic normalization
    text = text.lower().strip()
    # Remove common adjectives and states
    text = re.sub(r'\b(small|medium|large|fresh|dried|frozen|ripe|raw|cooked|chinese|dwarf)\b', '', text)
    # Convert plural to singular for basic cases
    text = re.sub(r'(\w+)ies$', r'\1y', text)  # berries -> berry
    text = re.sub(r'(\w+)s$', r'\1', text)     # bananas -> banana
    
    return text.strip()

def extract_ingredients_from_text(text):
    """Convert ingredient text to a clean list of ingredients."""
    ingredients_list = re.split(r',|\n|;|•|- ', text)
    return [clean_ingredient(ing) for ing in ingredients_list if ing.strip()]

def compare_ingredients(available_ingredients, recipe_ingredients):
    """Compare available vs needed ingredients and return missing ones."""
    available = set(clean_ingredient(ing) for ing in extract_ingredients_from_text(available_ingredients))
    needed = set(clean_ingredient(ing) for ing in recipe_ingredients)
    
    # Remove empty strings that might have resulted from cleaning
    available = {ing for ing in available if ing}
    needed = {ing for ing in needed if ing}
    
    missing = set()
    for ingredient in needed:
        # Check if any available ingredient contains or is contained in the needed ingredient
        if not any(available_ing in ingredient or ingredient in available_ing 
                for available_ing in available):
            missing.add(ingredient)
    
    return sorted(list(missing))

def mostrar_detalles_receta(idx, resultados, available_ingredients=""):
    recipe_index = resultados[idx][0]

    #  Usamos siempre el dataset ORIGINAL para mostrar información
    title = df_full.loc[recipe_index, "Title"]
    ingredients = df_full.loc[recipe_index, "Ingredients"]
    instructions = df_full.loc[recipe_index, "Instructions"]
    
    # Get clean list of recipe ingredients
    recipe_ingredients = extract_ingredients_from_text(ingredients)

    #  Formato más natural y bonito
    print("\n" + "═" * 50)
    print(f"🍽️ Recipe: {title}")
    print("═" * 50 + "\n")

    while True:
        print("🥕 Ingredients:\n")

        #  Mostrar ingredientes y marcar los que faltan
        if available_ingredients:
            # Mantener ingredientes originales para mostrar pero usar versión limpia para comparar
            missing = compare_ingredients(available_ingredients, extract_ingredients_from_text(ingredients))
            
            # Dividir y mostrar los ingredientes originales
            ingredientes_list = re.split(r',|\n|;|•|- ', ingredients) #type: ignore
            ingredientes_list = [ing.strip() for ing in ingredientes_list if ing.strip()]
            
            for ing_original in ingredientes_list:
                ing_clean = clean_ingredient(ing_original)
                if ing_clean in missing:
                    print(f"• {ing_original}")  # Ingrediente faltante
                else:
                    print(f"• {ing_original}")  # Ingrediente disponible
        else:
            # Si no hay lista de ingredientes disponibles, mostrar normal
            ingredientes_list = re.split(r',|\n|;|•|- ', ingredients) #type: ignore
            ingredientes_list = [ing.strip() for ing in ingredientes_list if ing.strip()]
            print("\n".join(f"• {ing}" for ing in ingredientes_list))

        opcion = input(
            "\nWhat would you like to do?\n"
            " 1️⃣ See instructions\n"
            " 2️⃣ Back to recipe list\n"
            "👉 Choice: "
        ).strip()

        if opcion == "1":
            print("\n👨‍🍳 Instructions:\n")

            #  Dividir texto en pasos numerados
            steps = re.split(r'\d+\)|\d+\.|\.|\n|\r', instructions) #type: ignore
            steps = [step.strip() for step in steps if step.strip()]
            for i, step in enumerate(steps, start=1):
                print(f"{i}. {step}")

            input("\nPress ENTER to go back...")

        elif opcion == "2":
            return  #  Volver a las recetas sin romper flujo

        else:
            print("Invalid option ❌. Try again.\n")

# ✅ MAIN INTERACTION LOOP ✅
def interactuar_con_usuario(query_text):
    global interaction_count
    interaction_count += 1

    more_count = 0
    offset = 0
    
    # Guardamos los ingredientes disponibles para comparar después
    available_ingredients = query_text

    resultados = recomendar_recetas(query_text)
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
            #  Mostrar ingredientes e instrucciones interactivo
            mostrar_detalles_receta(idx, resultados, available_ingredients)
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
