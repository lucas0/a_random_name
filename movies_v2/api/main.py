#!/usr/bin/env python3
import os, json, sqlite3, logging, traceback
import faiss, httpx, numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from string import Template

# ---- env ----
DB_PATH    = os.environ["DB_PATH_V2"]
FAISS_PATH = os.environ.get("FAISS_PATH_V2") or (os.environ["INDEX_V2_DIR"].rstrip("/") + "/movies.faiss")
EMB_MODEL  = os.environ["EMB_MODEL"]
BASE       = os.environ["OLLAMA_URL"].rstrip("/")
CHAT_URL   = BASE + "/api" + "/chat"
GEN_URL    = BASE + "/api" + "/generate"
MODEL      = os.environ["OLLAMA_MODEL"]

# ── logging ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("moviebot.api")

# ---- globals ----
app = FastAPI()
_index = None
_embed = None
_http  = None

# ---- models ----
class SearchIn(BaseModel): query:str; k:int=Field(10, ge=1, le=50)
class SearchOut(BaseModel): ids:list[int]
class MetaIn(BaseModel): ids:list[int]=Field(..., min_items=1)
class MovieRow(BaseModel):
    id:int; title:str; year:int|None=None; avg_rating:float|None=None
    imdb_url:str|None=None; tmdb_title:str|None=None
    tmdb_overview:str|None=None; omdb_overview:str|None=None
    director:str|None=None; cast:str|None=None,
    genres:str|None=None
class MetaOut(BaseModel): movies:list[MovieRow]
class QAIn(BaseModel): query:str; k:int=Field(8, ge=1, le=20)
class QAOut(BaseModel): ids:list[int]; answer:str; used_titles:list[str]=[]

PROMPT = Template(
"""You are MovieBot. Answer using CONTEXT.
- Do not make up facts.
- Prefer exact title (title + year).
- Use names & roles from CONTEXT.
- Recs: top 5 with 1-line reasons. ~180 words max.
- Do not include sources without metadata.

Return JSON: {"answer":"...","used_titles":["Title (Year)"]}

CONTEXT:
$context

QUESTION: $question
"""
)

# ---- tiny utils ----
def ensure_ready():
    global _index, _embed, _http
    if _index is None:
        if not os.path.exists(FAISS_PATH):
            raise HTTPException(500, f"FAISS index not found: {FAISS_PATH}")
        _index = faiss.read_index(FAISS_PATH)
    if _embed is None:
        _embed = SentenceTransformer(EMB_MODEL, device="cpu")
    if _http is None:
        _http = httpx.Client(timeout=300)

def embed(texts:list[str]) -> np.ndarray:
    ensure_ready()
    return _embed.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")

def search_ids(q:str, k:int) -> list[int]:
    ensure_ready()
    _, I = _index.search(embed([q]), k)
    return [int(i) for i in I[0] if i >= 0]

def fetch_metadata(ids: list[int]) -> list[dict]:
    if not ids: return []
    ph = ",".join(["?"] * len(ids))
    order_case = " ".join(f"WHEN ? THEN {i}" for i, _ in enumerate(ids))
    sql = f"""
      SELECT
        m.id, m.title, m.year, m.avg_rating, m.imdb_url,
        m.tmdb_title, m.tmdb_overview, m.omdb_overview,
        m.director, m."cast",
        (SELECT GROUP_CONCAT(g.name, ', ')
           FROM movie_genre mg
           JOIN genres g ON g.id = mg.genre_id
          WHERE mg.movie_id = m.id) AS genres
      FROM movies m
      WHERE m.id IN ({ph})
        AND (
              (m.director IS NOT NULL AND TRIM(m.director) <> '')
           OR (m."cast"  IS NOT NULL AND TRIM(m."cast")  <> '')
           OR (m.tmdb_overview IS NOT NULL AND TRIM(m.tmdb_overview) <> '')
           OR (m.omdb_overview IS NOT NULL AND TRIM(m.omdb_overview) <> '')
        )
      ORDER BY CASE m.id {order_case} END;
    """
    params = ids + ids  # ids for IN (...) + ids for ORDER BY CASE
    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(sql, params).fetchall()]

def build_context(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        t = (r.get("tmdb_overview") or "").strip()
        o = (r.get("omdb_overview") or "").strip()
        parts = [
            f"- {r.get('title')} ({r.get('year')})",
            f"director={r.get('director') or ''}",
            f"cast={r.get('cast') or ''}",
            f"avg_rating={r.get('avg_rating') or ''}",
            f"genres={r.get('genres') or ''}",
        ]
        if t: parts.append(f"tmdb_overview={t}")
        if o and o != t: parts.append(f"omdb_overview={o}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)

def call_ollama(prompt:str) -> str:
    ensure_ready()
    # Try /api/chat first
    resp = _http.post(CHAT_URL, json={
        "model": MODEL,
        "messages":[{"role":"user","content":prompt}],
        "stream": False,
        "options": {"temperature": 0.2}
    })
    if resp.status_code == 404:
        # Fallback to /api/generate (older/newer Ollama variants)
        resp = _http.post(GEN_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        })
    resp.raise_for_status()
    data = resp.json()
    # Support both shapes
    if isinstance(data.get("message"), dict):
        return (data["message"].get("content") or "").strip()
    return (data.get("response") or "").strip()

def parse_json_or_text(s:str) -> dict:
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        try: return json.loads(s)
        except: pass
    a,b = s.find("{"), s.rfind("}")
    if a!=-1 and b!=-1 and b>a:
        try: return json.loads(s[a:b+1])
        except: pass
    return {"answer": s, "used_titles": []}

# ── helpers to return JSON 500 with traceback ─────────────────────────
def _safe(call):
    try:
        return call()
    except Exception as e:
        tb = traceback.format_exc()
        log.error("handler failed: %s", tb)
        return JSONResponse(status_code=500, content={"error": str(e), "trace": tb[-2000:]})

# ── diagnostics ───────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/diag")
def diag():
    return {
        "DB_PATH_V2": DB_PATH,
        "FAISS_PATH": FAISS_PATH,
        "EMB_MODEL": EMB_MODEL,
        "OLLAMA_URL": BASE,
        "OLLAMA_MODEL": MODEL,
    }

# ── routes (wrapped) ─────────────────────────────────────────────────
@app.post("/search", response_model=SearchOut)
def search(inp:SearchIn):
    return _safe(lambda: {"ids": search_ids(inp.query, inp.k)})

@app.post("/metadata", response_model=MetaOut)
def metadata(inp:MetaIn):
    return _safe(lambda: {"movies": fetch_metadata(inp.ids)})

@app.post("/qa", response_model=QAOut)
def qa(inp:QAIn):
    def _run():
        ids  = search_ids(inp.query, inp.k)
        rows = fetch_metadata(ids)
        prompt = PROMPT.substitute(context=build_context(rows), question=inp.query)
        out = parse_json_or_text(call_ollama(prompt))
        return {"ids": ids, "answer": out.get("answer",""), "used_titles": out.get("used_titles", [])}
    return _safe(_run)