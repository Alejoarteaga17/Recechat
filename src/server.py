from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
# Import recommender from the src package so it works when running
# uvicorn with the module path (src.server:app)
from src import recommender
import os
import pandas as pd
import json
import time
from typing import Optional, List
from pydantic import BaseModel
from uuid import uuid4

# In-memory session store: session_id -> { resultados, offset, more_count, query }
SESSIONS = {}

app = FastAPI(title="Recechat API")

# Allow CORS for development (change in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from project/static
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if not os.path.exists(STATIC_DIR):
    # also accept a top-level static folder
    STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class Query(BaseModel):
    query: str
    k: int = 5


class Feedback(BaseModel):
    user_id: Optional[str] = None
    query: Optional[str] = None
    recommended_ids: Optional[List[int]] = None
    scores: Optional[List[float]] = None
    feedback: str
    method: Optional[str] = None


LOGS_CSV = os.path.join(getattr(recommender, 'DATA_DIR', os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')), 'logs.csv')


@app.post("/api/feedback")
async def feedback_endpoint(fb: Feedback):
    """Persist feedback to LOGS_CSV (appends row)."""
    os.makedirs(os.path.dirname(LOGS_CSV), exist_ok=True)
    row = {
        'timestamp': time.time(),
        'user_id': fb.user_id,
        'query': fb.query,
        'recommended_ids': json.dumps(fb.recommended_ids) if fb.recommended_ids is not None else json.dumps([]),
        'scores': json.dumps(fb.scores) if fb.scores is not None else json.dumps([]),
        'feedback': fb.feedback,
        'method': fb.method,
    }
    df_row = pd.DataFrame([row])
    if not os.path.exists(LOGS_CSV):
        df_row.to_csv(LOGS_CSV, index=False)
    else:
        df_row.to_csv(LOGS_CSV, mode='a', header=False, index=False)
    return {"status": "ok", "saved": True}


class ChatStart(BaseModel):
    query: str
    k: int = 5


@app.post("/api/chat/start")
async def chat_start(payload: ChatStart):
    """Start a chat session: run recommendation, store resultados and return friendly text + top-k items."""
    q = payload.query
    k = payload.k
    try:
        resultados = recommender.recomendar_recetas(q, top_k=k)
    except Exception as e:
        return {"error": str(e), "results": []}

    # store session
    session_id = str(uuid4())
    SESSIONS[session_id] = {
        'resultados': resultados,
        'offset': 0,
        'more_count': 0,
        'query': q,
    }

    # Friendly response using recommender helper
    try:
        friendly = recommender.mostrar_respuesta_amigable(resultados, q, offset=0)
    except Exception:
        friendly = ''

    # Build compact results list
    compact = []
    df_full = getattr(recommender, 'df_full', None)
    df = getattr(recommender, 'df', None)
    for idx, title, score in resultados:
        try:
            rid = int(idx)
        except Exception:
            rid = idx
        snippet = ''
        try:
            if df_full is not None and rid in df_full.index:
                snippet = str(df_full.loc[rid, 'Ingredients']) if 'Ingredients' in df_full.columns else ''
            elif df is not None and rid in df.index:
                snippet = str(df.loc[rid, 'Cleaned_Ingredients']) if 'Cleaned_Ingredients' in df.columns else ''
        except Exception:
            pass
        compact.append({'id': rid, 'title': title, 'score': float(score), 'ingredients': snippet})

    return {'session_id': session_id, 'friendly': friendly, 'results': compact, 'offset': 0}


class ChatMore(BaseModel):
    session_id: str


@app.post('/api/chat/more')
async def chat_more(payload: ChatMore):
    sess = SESSIONS.get(payload.session_id)
    if not sess:
        return {'error': 'invalid session'}
    resultados = sess['resultados']
    offset = sess.get('offset', 0)
    more_count = sess.get('more_count', 0)

    try:
        out_text, new_offset = recommender.mostrar_mas_recetas(resultados, offset, more_count)
    except Exception as e:
        return {'error': str(e)}

    sess['offset'] = new_offset
    sess['more_count'] = more_count + 1

    # return the textual response (and optionally new items computed from resultados slice)
    # we'll also return the next batch of items if available
    items = []
    df_full = getattr(recommender, 'df_full', None)
    df = getattr(recommender, 'df', None)
    for idx, title, score in resultados[new_offset:new_offset+3]:
        try:
            rid = int(idx)
        except Exception:
            rid = idx
        snippet = ''
        try:
            if df_full is not None and rid in df_full.index:
                snippet = str(df_full.loc[rid, 'Ingredients']) if 'Ingredients' in df_full.columns else ''
            elif df is not None and rid in df.index:
                snippet = str(df.loc[rid, 'Cleaned_Ingredients']) if 'Cleaned_Ingredients' in df.columns else ''
        except Exception:
            pass
        items.append({'id': rid, 'title': title, 'score': float(score), 'ingredients': snippet})

    return {'text': out_text, 'items': items, 'offset': sess['offset']}


class ChatSelect(BaseModel):
    session_id: str
    index: int


@app.post('/api/chat/select')
async def chat_select(payload: ChatSelect):
    sess = SESSIONS.get(payload.session_id)
    if not sess:
        return {'error': 'invalid session'}
    resultados = sess['resultados']
    idx = payload.index
    if idx < 0 or idx >= len(resultados):
        return {'error': 'index out of range'}

    recipe_index = resultados[idx][0]
    # Fetch details from df_full (original dataset)
    df_full = getattr(recommender, 'df_full', None)
    if df_full is not None and recipe_index in df_full.index:
        title = df_full.loc[recipe_index, 'Title']
        ingredients = df_full.loc[recipe_index, 'Ingredients']
        instructions = df_full.loc[recipe_index, 'Instructions']
        return {'id': int(recipe_index), 'title': title, 'ingredients': ingredients, 'instructions': instructions}

    # fallback: try cleaned df
    df = getattr(recommender, 'df', None)
    if df is not None and recipe_index in df.index:
        title = df.loc[recipe_index, 'Title'] if 'Title' in df.columns else ''
        ingredients = df.loc[recipe_index, 'Cleaned_Ingredients'] if 'Cleaned_Ingredients' in df.columns else ''
        return {'id': int(recipe_index), 'title': title, 'ingredients': ingredients, 'instructions': ''}

    return {'error': 'recipe not found'}


class ChatRecipeIngredients(BaseModel):
    session_id: str
    index: int
    available: str


@app.post('/api/chat/recipe_ingredients')
async def chat_recipe_ingredients(payload: ChatRecipeIngredients):
    """Return the ingredient list for a recipe with availability marks based on user's available ingredients text."""
    sess = SESSIONS.get(payload.session_id)
    if not sess:
        return {'error': 'invalid session'}

    resultados = sess['resultados']
    idx = payload.index
    if idx < 0 or idx >= len(resultados):
        return {'error': 'index out of range'}

    recipe_index = resultados[idx][0]
    df_full = getattr(recommender, 'df_full', None)
    if df_full is None or recipe_index not in df_full.index:
        return {'error': 'recipe not found'}

    title = df_full.loc[recipe_index, 'Title']
    ingredients_text = str(df_full.loc[recipe_index, 'Ingredients'])

    # Build original list and compare availability using recommender helpers
    try:
        # Original display list split similar to CLI
        import re as _re
        original_list = [_s.strip() for _s in _re.split(r',|\n|;|•|- ', ingredients_text) if _s.strip()]

        # Cleaned list for comparison
        cleaned_needed = recommender.extract_ingredients_from_text(ingredients_text)
        missing = set(recommender.compare_ingredients(payload.available, cleaned_needed))

        detailed = []
        for ing_original in original_list:
            ing_clean = recommender.clean_ingredient(ing_original)
            detailed.append({'text': ing_original, 'missing': bool(ing_clean in missing)})

        return {'id': int(recipe_index), 'title': title, 'ingredients': detailed}
    except Exception as e:
        return {'error': str(e)}


@app.post("/api/recommend")
async def recommend(q: Query):
    """Return recommendations using recommender.recomendar_recetas."""
    try:
        resultados = recommender.recomendar_recetas(q.query, top_k=q.k)
    except Exception as e:
        return {"error": str(e), "results": []}

    out = []
    df_full = getattr(recommender, "df_full", None)
    df = getattr(recommender, "df", None)

    for idx, title, score in resultados:
        try:
            rid = int(idx)
        except Exception:
            rid = idx

        snippet = ""
        instr = ""
        try:
            if df_full is not None and rid in df_full.index:
                snippet = str(df_full.loc[rid, "Ingredients"]) if "Ingredients" in df_full.columns else ""
                instr = str(df_full.loc[rid, "Instructions"]) if "Instructions" in df_full.columns else ""
            elif df is not None and rid in df.index:
                snippet = str(df.loc[rid, "Cleaned_Ingredients"]) if "Cleaned_Ingredients" in df.columns else ""
        except Exception:
            # ignore snippet extraction errors
            pass

        out.append({
            "id": rid,
            "title": title,
            "score": float(score),
            "ingredients": snippet,
            "instructions": instr,
        })

    return {"query": q.query, "results": out}


@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Recechat API running. Open /static/index.html"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)
