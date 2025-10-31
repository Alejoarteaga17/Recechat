# ✅ memory_manager.py — Improved with placeholder enforcement ✅

import os
import json
import random
import re
from collections import defaultdict, Counter
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RESPONSES_PATH = "responses_memory.json"
FEEDBACK_LOG = "feedback_log.csv"
STATS_PATH = "response_stats.json"
BACKUP_PATH = "responses_memory.bak.json"

MIN_ACCEPT_RATE_FOR_GOOD = 0.3
MIN_SHOWS_FOR_EVAL = 3
REQUIRED_PLACEHOLDER = "{list}"

SYNONYM_DICT = {
    "delicious": ["delicious", "tasty", "yummy", "delish"],
    "try": ["try", "give it a shot", "consider", "have a go"],
    "recipe": ["recipe", "dish", "meal idea"],
    "similar": ["similar", "close", "alike", "matching"]
}


def _ensure_files():
    if not os.path.exists(RESPONSES_PATH):
        default = {
            "templates": [
                "Here are some recipes you might like: {list}",
                "No exact match 😅 but try these: {list}",
                "These recipes seem close to what you described: {list}",
                "Need more ideas? Try these: {list}",
                "I couldn't find an exact match, but maybe you'd enjoy: {list}"
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
        return json.load(f).get("templates", [])

# ✅ Ensure template file exists and enforce required placeholder if needed


def save_templates(templates):
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
    line = f"{datetime.now().isoformat()}\t{action}\t{query}\t{response_text}\n"
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(line)

    stats = load_stats()
    key = response_text.strip()
    if key not in stats:
        stats[key] = {"shown": 0, "accepted": 0, "rejected": 0, "more": 0}

    if action in stats[key]:
        stats[key][action] += 1

    stats[key]["last_shown"] = datetime.now().isoformat()
    save_stats(stats)


# ✅ Sentence mixing for generating new templates
def _split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+|\n', text)
    return [p.strip() for p in parts if p.strip()]


def _combine_texts(base, donor):
    base_first = _split_sentences(base)[0]
    donor_first = _split_sentences(donor)[0]

    new = f"{base_first}. {donor_first}".strip()

    for k, syns in SYNONYM_DICT.items():
        if k in new.lower():
            new = re.sub(r'\b' + re.escape(k) + r'\b', random.choice(syns), new, flags=re.IGNORECASE)

    # ✅ Enforce placeholder
    if REQUIRED_PLACEHOLDER not in new:
        new += f" {REQUIRED_PLACEHOLDER}"

    return new


def evaluate_and_improve():
    stats = load_stats()
    if not stats:
        return {"msg": "no stats yet"}

    templates = load_templates()

    # Evaluate acceptance rate
    scored = []
    for t, s in stats.items():
        shown = s["shown"]
        acc = s["accepted"]
        if shown >= MIN_SHOWS_FOR_EVAL:
            rate = acc / shown if shown else 0
            scored.append((t, rate))

    if not scored:
        return {"msg": "not enough data"}

    scored.sort(key=lambda x: x[1])
    low = [t for t, r in scored[:3] if r < MIN_ACCEPT_RATE_FOR_GOOD]
    high = [t for t, r in scored[-3:]]

    new_variants = []
    for bad in low:
        for good in high:
            new = _combine_texts(bad, good)
            if new not in templates and new not in new_variants:
                new_variants.append(new)

    if not new_variants:
        return {"msg": "no new templates needed"}

    templates.extend(new_variants)
    templates = list(set(templates))
    save_templates(templates)

    return {"msg": "templates improved", "added": len(new_variants)}


# ✅ Template selection: Weighted by accept rate + semantic matching
def get_best_template_for_query(query, templates=None):
    if templates is None:
        templates = load_templates()

    stats = load_stats()
    weighted = []
    for t in templates:
        st = stats.get(t, {"shown": 0, "accepted": 0})
        shown, acc = st["shown"], st["accepted"]
        score = (acc / shown) if shown else 0.1
        weighted.append((t, score + random.uniform(0, 0.25)))

    weighted.sort(key=lambda x: -x[1])
    top = [t for t, _ in weighted[:5]]

    try:
        vect = TfidfVectorizer().fit([query] + top)
        X = vect.transform([query] + top)
        sims = cosine_similarity(X[0:1], X[1:]).flatten()
        return top[int(sims.argmax())]
    except:
        return random.choice(top)


# ✅ Final templating
def render_template(template, *args, **kwargs):
    """Render a template replacing the {list} placeholder.

    This function keeps backward compatibility with callers that pass the
    replacement as either a positional second argument or as the keyword
    argument named "list". We intentionally avoid using the name
    "list" for the parameter to prevent shadowing the built-in type.

    The replacement may be a string or a list/tuple; lists/tuples are
    joined using "; ". Other types are coerced to str.
    """
    # Accept either keyword 'list' or the first positional argument
    if 'list' in kwargs:
        value = kwargs['list']
    elif args:
        value = args[0]
    else:
        value = ""

    if isinstance(value, (list, tuple)):
        list_str = "; ".join(map(str, value))
    else:
        list_str = str(value)

    return template.replace("{list}", list_str)
