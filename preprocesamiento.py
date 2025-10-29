# =========================
# Preprocesamiento de Recetas
# Persona A - Proyecto Chatbot de Recetas
# =========================

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Descargar recursos de nltk (solo la primera vez)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 1 Cargar dataset
df = pd.read_csv('../Food Ingredients and Recipe Dataset.csv')

# Ver columnas
print("Columnas del dataset:")
print(df.columns)
print("Total de filas:", len(df))

# 2 Seleccionar columnas clave
df = df[['Title', 'Cleaned_Ingredients', 'Instructions']]

# 3 Definir limpieza textual
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()  # Minúsculas
    text = re.sub(r'[^a-z\s]', '', text)  # Eliminar símbolos
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]  # Quitar stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]  # Lematización
    return ' '.join(tokens)

# 4 Aplicar limpieza
df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].astype(str).apply(clean_text)
df['Instructions'] = df['Instructions'].astype(str).apply(clean_text)

# 5 Guardar dataset limpio
df.to_csv('cleaned_recipes.csv', index=False)

print(" Limpieza completada. Archivo guardado como cleaned_recipes.csv")
