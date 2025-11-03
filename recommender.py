import os
import random
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import memory_manager as mm
from utils import (
    DEFAULT_DATA_PATH,
    VECTORIZER_PATH,
    TFIDF_PATH,
    EMBEDDINGS_PATH,
    LOGS_CSV,
    SURVEY_CSV,
    cargar_dataset,
    cargar_o_entrenar_tfidf
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

MAX_MORE_REQUESTS = 2
TOP_K = 12
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'

# This function is now designed to be called from the CLI main block
def recomendar_recetas_tfidf(query_text, vectorizer, tfidf_matrix, titles, top_k=TOP_K):
    q_vec = vectorizer.transform([query_text])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    idx_top = np.argsort(-sims)[:top_k]
    return [(i, titles[i], float(sims[i])) for i in idx_top]

def format_recipes_for_display(resultados, offset=0, limit=3):
    formatted = []
    for pos, (_, title, score) in enumerate(resultados[offset:offset+limit], start=offset+1):
        formatted.append(f"{pos}. {title} ({score:.2f})")
    return formatted

def mostrar_respuesta_amigable(resultados, query_text, common_ingredients, offset=0):
    if resultados and resultados[0][2] >= 0.05:
        lista = format_recipes_for_display(resultados, offset, 3)
        salida = "Here are some recipes you might like:\n" + "\n".join(lista)
        mm.register_interaction(query_text, salida, "shown")
        return salida

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

def mostrar_detalles_receta(idx, resultados, df):
    recipe_index = resultados[idx][0]
    
    title = df.loc[recipe_index, "Title"]
    ingredients = df.loc[recipe_index, "Cleaned_Ingredients"]
    instructions = df.loc[recipe_index, "Instructions"]

    ingredientes_list = re.split(r',|;|
| - ', ingredients)
    ingredientes_list = [f"- {ing.strip()}" for ing in ingredientes_list if ing.strip()]
    formatted_ingredients = "\n".join(ingredientes_list)

    while True:
        print(f"\n📌 Recipe: {title}")
        print("\n🥕 Ingredients needed:\n")
        print(formatted_ingredients)

        opcion = input("\nWhat would you like to do?\n1 = Show instructions\n2 = Back to recipes\nChoice: ").strip()

        if opcion == "1":
            steps = re.split(r'\.|\n|\r|\d+\)|\d+\.', instructions)
            steps = [step.strip() for step in steps if step.strip()]
            formatted_steps = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])

            print("\n👨‍🍳 Instructions:\n")
            print(formatted_steps)
            input("\nPress ENTER to go back...")
        
        elif opcion == "2":
            return

        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    print("🍳 Recipe AI Recommender (CLI Mode)")
    
    try:
        df = cargar_dataset(DEFAULT_DATA_PATH)
        titles = df['Title'].fillna('No title').tolist()
        print("Dataset loaded.")
        
        vectorizer, tfidf_matrix = cargar_o_entrenar_tfidf(df)
        print("TF-IDF models loaded/trained.")
        
        # For CLI, we can just use a basic list of common ingredients
        common_ingredients = ['salt', 'water', 'oil', 'sugar', 'egg', 'flour', 'milk', 'butter']

    except Exception as e:
        print(f"Failed to initialize models: {e}")
        exit()

    interaction_count = 0
    while True:
        user_input = input("\nWhat ingredients do you have today? (type 'exit' to stop) ").strip()
        if user_input.lower() == "exit":
            print("Goodbye! 👋")
            break
        
        interaction_count += 1
        more_count = 0
        offset = 0

        resultados = recomendar_recetas_tfidf(user_input, vectorizer, tfidf_matrix, titles)
        respuesta = mostrar_respuesta_amigable(resultados, user_input, common_ingredients, offset)
        print("\n" + respuesta)

        while True:
            opcion = input("\nDid you like any recipe? (yes/no/more/exit): ").strip().lower()

            if opcion in ("yes", "y"):
                idx_str = input("Which number did you like?: ").strip()
                if not idx_str.isdigit():
                    print("Please enter a valid number.")
                    continue
                idx = int(idx_str) - 1
                if idx < 0 or idx >= len(resultados):
                    print("That number is not in the list.")
                    continue
                
                mostrar_detalles_receta(idx, resultados, df)
                mm.register_interaction(user_input, respuesta, "accepted")
                break

            elif opcion in ("no", "n"):
                print("I'll keep improving!")
                mm.register_interaction(user_input, respuesta, "rejected")
                break

            elif opcion == "more":
                more_response, offset = mostrar_mas_recetas(resultados, offset, more_count)
                more_count += 1
                print("\n" + more_response)

            elif opcion == "exit":
                break # Exit inner loop to get new ingredients

            else:
                print("Options: yes / no / more / exit")
        
        if opcion == "exit":
            continue