import pandas as pd
import os
import pandas as pd
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# 1 Cargar dataset (ruta relativa al archivo actual)
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, 'Food Ingredients and Recipe Dataset.csv')

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError('Food Ingredients and Recipe Dataset.csv not found in src directory')

df = pd.read_csv(DATASET_PATH)

# Ver columnas
print("Columnas del dataset:")
print(df.columns)
print("Total de filas:", len(df))

# 2 Seleccionar columnas clave
df = df[['Title', 'Cleaned_Ingredients', 'Instructions']]

# 3 Definir limpieza textual y estrategia para extraer solo ingredientes
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def basic_tokenize(text):
    text = text.lower() if pd.notnull(text) else ""
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in word_tokenize(text) if t.isalpha()]
    return tokens


def normalize_synonyms(text, synonyms):
    """Replace variant terms in text with canonical synonym values.

    Performs word-boundary replacements using the synonyms mapping.
    """
    if not text:
        return text
    txt = text.lower()
    # Sort keys by length to replace longer phrases first
    for k in sorted(synonyms.keys(), key=lambda x: -len(x)):
        v = synonyms[k]
        # replace only whole-word occurrences
        txt = re.sub(r'\b' + re.escape(k.lower()) + r'\b', v.lower(), txt)
    return txt


def build_ingredient_whitelist(df_series, common_path=None, top_n=500):
    """Build a whitelist of likely ingredient tokens.

    Strategy:
    - Tokenize all ingredient strings and POS-tag them.
    - Keep frequent nouns and any manually provided common ingredients.
    """
    counter = Counter()

    for text in df_series.astype(str):
        tokens = basic_tokenize(text)
        # POS tag and select nouns
        try:
            tags = nltk.pos_tag(tokens)
            nouns = [lemmatizer.lemmatize(w) for w, tag in tags if tag.startswith('NN')]
        except Exception:
            nouns = [lemmatizer.lemmatize(t) for t in tokens]

        counter.update(nouns)

    # Most common nouns across all recipes
    most_common = [w for w, _ in counter.most_common(top_n)]

    whitelist = set(most_common)

    # Load manual common ingredients if provided
    if common_path and os.path.exists(common_path):
        with open(common_path, 'r', encoding='utf-8') as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    whitelist.add(w)

    return whitelist


def clean_text_keep_ingredients(text, whitelist):
    """Keep only tokens that are in the whitelist (after lemmatization).
    Return a space-joined string of ingredient tokens.
    """
    if pd.isnull(text):
        return ""
    tokens = basic_tokenize(text)
    lems = [lemmatizer.lemmatize(t) for t in tokens]
    kept = [t for t in lems if t in whitelist and t not in stop_words]
    return ' '.join(sorted(set(kept)))


def clean_text(text):
    """General cleaning for instructions/text fields: lower, remove non-alpha,
    remove stopwords and lemmatize."""
    if pd.isnull(text):
        return ""
    tokens = basic_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# 4 Aplicar limpieza
common_path = os.path.join(os.path.dirname(__file__), 'common_ingredients.txt')
# Load synonyms mapping to normalize variants in the dataset
syn_path = os.path.join(os.path.dirname(__file__), 'synonyms.json')
if os.path.exists(syn_path):
    with open(syn_path, 'r', encoding='utf-8') as f:
        synonyms = json.load(f)
else:
    synonyms = {}

# Normalize synonyms in the source ingredient text before building whitelist
df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].astype(str).apply(lambda x: normalize_synonyms(x, synonyms))

whitelist = build_ingredient_whitelist(df['Cleaned_Ingredients'], common_path=common_path, top_n=1000)

# Apply cleaning: keep only whitelisted ingredient tokens
df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].astype(str).apply(lambda x: clean_text_keep_ingredients(x, whitelist))
df['Instructions'] = df['Instructions'].astype(str).apply(clean_text)

# 5 Guardar dataset limpio
out_path = 'cleaned_recipes.csv'
df.to_csv(out_path, index=False)

print(f"Limpieza completada. Archivo guardado como {out_path}")
