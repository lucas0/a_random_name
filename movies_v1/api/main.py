#!/usr/bin/env python3
import os, sqlite3, numpy as np, faiss, requests
from fastapi import FastAPI, Query, Body, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer

# ---- required envs (no defaults) ----
ENV_VARS = ["DB_PATH_V1", "EMB_MODEL", "FAISS_PATH_V1", "OLLAMA_URL", "LLM_MODEL_NAME"]
missing = [k for k in ENV_VARS if not os.environ.get(k)]
if missing:
    raise RuntimeError(f"Missing env vars: {', '.join(missing)}. Did you `source env.sh`?")

DB_PATH   = os.environ["DB_PATH_V1"]
EMB_MODEL = os.environ["EMB_MODEL"]
FAISS_P   = os.environ["FAISS_PATH_V1"]
OLLAMA_URL= os.environ["OLLAMA_URL"]
LLM_MODEL = os.environ["LLM_MODEL_NAME"]

# ---- load once ----
app   = FastAPI(title="Movie QA API", version="0.1")
conn  = sqlite3.connect(DB_PATH, check_same_thread=False)
index = faiss.read_index(FAISS_P)                 # IndexIDMap2 with movie IDs
model = SentenceTransformer(EMB_MODEL, device="cpu")

# ---- schemas ----
class IdsIn(BaseModel): 
    ids: List[int]
class Movie(BaseModel):
    id: int; 
    title: Optional[str]=None; 
    year: Optional[int]=None
    avg_rating: Optional[float]=None; 
    n_ratings: Optional[int]=None
    genres: Optional[str]=None; 
    director: Optional[str]=None
    cast: Optional[str]=None; 
    overview: Optional[str]=None
class AnswerIn(BaseModel): 
    user_input: str; 
    items: List[Movie]
class AnswerOut(BaseModel): 
    answer: str

# ---- endpoints ----
@app.get("/health") 
def health(): return {"ok": True}

@app.get("/search_ids", response_model=List[int])
def api_search_ids(q: str = Query(..., min_length=1), k: int = 12):
    q_emb = model.encode([q], normalize_embeddings=True).astype("float32")
    sims, labels = index.search(q_emb, k)
    return [int(x) for x in labels[0] if int(x) != -1]

@app.post("/metadata", response_model=List[Movie])
def api_metadata(payload: IdsIn = Body(...)):
    ids = payload.ids
    if not ids: return []
    qs = ",".join("?"*len(ids))
    sql = f"""
      SELECT m.id, m.title AS title, m.year,
             (SELECT ROUND(AVG(r.rating),2) FROM ratings r WHERE r.movie_id=m.id) AS avg_rating,
             (SELECT COUNT(*) FROM ratings r WHERE r.movie_id=m.id) AS n_ratings,
             (SELECT GROUP_CONCAT(g.name, ', ')
                FROM movie_genre mg JOIN genres g ON g.id=mg.genre_id
               WHERE mg.movie_id=m.id) AS genres,
             m.director, m."cast",
             m.overview AS overview
      FROM movies m WHERE m.id IN ({qs});
    """
    cur = conn.execute(sql, ids)
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    order = {mid:i for i, mid in enumerate(ids)}
    rows.sort(key=lambda x: order.get(x["id"], 10**9))
    return rows

@app.post("/answer", response_model=AnswerOut)
def api_answer(payload: AnswerIn = Body(...)):
    if not payload.items:
        raise HTTPException(400, "No items provided for answering.")
    bullets = "\n".join(
        f"- {it.title or ''} ({it.year or '—'}), ⭐ {it.avg_rating or '—'} • {it.genres or ''} • Dir: {it.director or '—'}"
        for it in payload.items[:10]
    )
    prompt = (
        "You are a movie assistant. Use these items as support sources for your answer. Try to drive an answer to the user question based mainly on these sources.\n\n"
        f"User question: {payload.user_input}\n\nItems:\n{bullets}\n\n"
        "Answer briefly (2–5 sentences), citing titles, years, scores, directors and cast when helpful."
    )
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL, 
            "messages":[{"role":"user","content":prompt}],
            "stream": False, 
            "options": {"temperature": 0.2}},
        timeout=120)
    if r.status_code != 200:
        raise HTTPException(502, f"Ollama error: {r.text}")
    text = (r.json().get("message") or {}).get("content", "").strip()
    return {"answer": text or "Sorry—I couldn’t generate an answer."}
