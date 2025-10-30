# memory_manager.py
import os
import json
import random
import re
from collections import defaultdict, Counter
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
RESPONSES_PATH = "responses_memory.json"
FEEDBACK_LOG = "feedback_log.csv"
STATS_PATH = "response_stats.json"
BACKUP_PATH = "responses_memory.bak.json"

# Parameters
MIN_ACCEPT_RATE_FOR_GOOD = 0.3    # acceptance rate threshold
MIN_SHOWS_FOR_EVAL = 3            # minimum times a response must be shown to be evaluated
SYNONYM_DICT = {
    "delicious": ["delicious", "tasty", "yummy", "delish"],
    "try": ["try", "give a shot", "consider", "have a go"],
    "recipe": ["recipe", "dish", "idea"],
    "similar": ["similar", "close", "alike", "matching"]
}

def _ensure_files():
    if not os.path.exists(RESPONSES_PATH):
        default = {
            "templates": [
                "These recipes seem close to what you described: {list}. Would you like more options?",
                "I couldn't find an exact match 😅. Try these common ingredients: {list}.",
                "Hmm... that doesn't sound like a recipe, but I can help with meal ideas!",
                "Here are some recipes you might like:\n{list}\nDo you want more?",
                "No exact match, but how about something with {list}?"
            ]
        }
        with open(RESPONSES_PATH, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
    if not os.path.exists(STATS_PATH):
        with open(STATS_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f)

_ensure_files()

def load_templates():
    with open(RESPONSES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("templates", [])

def save_templates(templates):
    # backup old
    if os.path.exists(RESPONSES_PATH):
        try:
            os.replace(RESPONSES_PATH, BACKUP_PATH)
        except Exception:
            pass
    with open(RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump({"templates": templates}, f, ensure_ascii=False, indent=2)

def load_stats():
    if not os.path.exists(STATS_PATH):
        return {}
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_stats(stats):
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def register_interaction(query, response_text, action):
    """
    Register an interaction.
    action: 'shown' | 'accepted' | 'rejected' | 'more'
    """
    # append to feedback log CSV-like
    line = f"{datetime.now().isoformat()}\t{action}\t{query.replace(chr(9),' ')}\t{response_text.replace(chr(9),' ')}\n"
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(line)

    # update stats in memory file
    stats = load_stats()
    key = response_text.strip()
    if key not in stats:
        stats[key] = {"shown": 0, "accepted": 0, "rejected": 0, "more": 0, "last_shown": None}
    stats[key]["last_shown"] = datetime.now().isoformat()
    if action == "shown":
        stats[key]["shown"] += 1
    elif action == "accepted":
        stats[key]["accepted"] += 1
    elif action == "rejected":
        stats[key]["rejected"] += 1
    elif action == "more":
        stats[key]["more"] += 1
    save_stats(stats)

def _tfidf_similarity_matrix(texts):
    if len(texts) < 2:
        return None, None
    vect = TfidfVectorizer().fit(texts)
    X = vect.transform(texts)
    sim = cosine_similarity(X)
    return sim, vect

def _split_sentences(text):
    # naive split
    parts = re.split(r'(?<=[\.\!\?]\s)|\n', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]

def _combine_texts(base, donor):
    """
    Combine base and donor text to create a new variant. Strategy:
    - split into sentences, pick first part from base, second from donor
    - make a few simple synonym swaps
    """
    b_parts = _split_sentences(base)
    d_parts = _split_sentences(donor)
    # pick fragments
    part_a = b_parts[0] if b_parts else base
    part_b = d_parts[0] if d_parts else donor
    # if both short, just concatenate
    candidate = part_a.rstrip(".!?") + ". " + part_b
    # simple synonym substitutions
    for k, syns in SYNONYM_DICT.items():
        if k in candidate.lower():
            candidate = re.sub(r'\b' + re.escape(k) + r'\b', random.choice(syns), candidate, flags=re.IGNORECASE)
    # ensure punctuation
    if candidate and candidate[-1] not in ".!?":
        candidate += "."
    # minor cleanup
    candidate = re.sub(r'\s+', ' ', candidate).strip()
    return candidate

def evaluate_and_improve(min_shows=MIN_SHOWS_FOR_EVAL, min_accept_rate=MIN_ACCEPT_RATE_FOR_GOOD, top_n_replace=5):
    """
    Evaluate response templates and automatically create improved variants:
    - compute acceptance rates
    - find low-performing templates (shown >= min_shows and accept_rate < threshold)
    - for each low one, find good template(s) and combine to create new variants
    - save new templates appended to responses_memory.json
    """
    stats = load_stats()
    if not stats:
        return {"msg": "no stats available", "changes": 0}

    # compute rates
    rates = []
    for t, s in stats.items():
        shown = s.get("shown", 0)
        accepted = s.get("accepted", 0)
        rejected = s.get("rejected", 0)
        rate = (accepted / shown) if shown > 0 else 0.0
        rates.append((t, shown, accepted, rejected, rate))

    # identify good and bad templates
    good = [r for r in rates if r[1] >= min_shows and r[4] >= min_accept_rate]
    bad = [r for r in rates if r[1] >= min_shows and r[4] < min_accept_rate]

    if not bad:
        return {"msg": "no low-performing templates", "changes": 0}

    # load current templates
    templates = load_templates()
    all_texts = templates + [r[0] for r in rates]
    # compute similarity among all existing responses to find similar good donors
    sim_matrix, vect = _tfidf_similarity_matrix(list(set(all_texts)))
    # map index to text
    unique_texts = list(set(all_texts))
    text_to_index = {t:i for i,t in enumerate(unique_texts)}

    new_variants = []
    for bad_t, shown, accepted, rejected, rate in bad:
        # find donors among good with highest similarity
        donors = []
        if sim_matrix is not None:
            bad_idx = text_to_index.get(bad_t)
            if bad_idx is not None:
                sims = sim_matrix[bad_idx]
                # sort by similarity and keep those corresponding to good templates
                sim_pairs = sorted([(unique_texts[i], sims[i]) for i in range(len(sims)) if unique_texts[i] != bad_t], key=lambda x:-x[1])
                # pick top donors
                for txt, sc in sim_pairs:
                    # prefer donors that are in good templates or in existing templates
                    donors.append((txt, sc))
                    if len(donors) >= 5:
                        break
        # fallback: random sample of templates
        if not donors:
            donors = [(t, 0.0) for t in random.sample(templates, min(3, len(templates)))]

        # combine bad text with top donor(s)
        donor_text = donors[0][0] if donors else random.choice(templates)
        candidate = _combine_texts(bad_t, donor_text)
        # small safeguard: don't add duplicate nearly identical strings
        if candidate and candidate not in templates and candidate not in new_variants:
            new_variants.append(candidate)

    # also generate small paraphrase variants for high-performing templates
    paraphrases = []
    for t, shown, accepted, rejected, rate in good[:top_n_replace]:
        # create 1-2 small paraphrases via synonym swap + reordering
        p = _combine_texts(t, random.choice(templates))
        if p not in templates and p not in new_variants and p not in paraphrases:
            paraphrases.append(p)

    final_new = new_variants + paraphrases
    if final_new:
        templates.extend(final_new)
        save_templates(templates)
        return {"msg": "improved templates added", "changes": len(final_new), "added": final_new}
    return {"msg": "no new variants generated", "changes": 0}

def get_best_template_for_query(query, templates=None):
    """
    Return the most appropriate template for a given query.
    Strategy:
    - compute TF-IDF similarity between query and templates and choose a high-acceptance template
    - otherwise: return a random template
    """
    if templates is None:
        templates = load_templates()
    if not templates:
        return "Sorry, I couldn't think of a reply."

    # simple heuristic: choose template with some randomness but biased by recent acceptance
    stats = load_stats()
    scores = []
    for t in templates:
        stat = stats.get(t, {})
        shown = stat.get("shown", 0)
        accepted = stat.get("accepted", 0)
        accept_rate = (accepted / shown) if shown > 0 else 0.15
        # base score: acceptance rate + small random factor
        score = accept_rate + random.uniform(0, 0.2)
        scores.append((t, score))

    # pick top N by score, then choose the one most semantically relevant via TF-IDF
    scores_sorted = sorted(scores, key=lambda x:-x[1])
    top_candidates = [s[0] for s in scores_sorted[:min(6, len(scores_sorted))]]

    # semantic ranking among top candidates
    try:
        vect = TfidfVectorizer().fit([query] + top_candidates)
        X = vect.transform([query] + top_candidates)
        # use a 2D slice for the query row so we don't index a sparse matrix into a 1D spmatrix
        sims = cosine_similarity(X[0:1], X[1:]).flatten() # type: ignore
        best_idx = int(sims.argmax())
        return top_candidates[best_idx]
    except Exception:
        return random.choice(top_candidates)

# convenience wrapper that formats list placeholders
def render_template(template, list_text=""):
    try:
        return template.format(list=list_text)
    except Exception:
        # fallback simple replacement
        return template.replace("{list}", list_text)
