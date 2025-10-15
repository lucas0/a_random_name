#!/usr/bin/env python3
import os, sqlite3, numpy as np, faiss
from tqdm import trange
from sentence_transformers import SentenceTransformer

DB_PATH   = os.getenv("DB_PATH_V2", "movies.db")
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_DIR  = os.getenv("INDEX_V2_DIR", "index").rstrip("/")
FAISS_OUT  = f"{INDEX_DIR}/movies.faiss"

SQL = """
SELECT
  m.id,
  COALESCE(m.tmdb_title, m.title) AS title,
  m.tmdb_year,
  (SELECT GROUP_CONCAT(g.name, ', ')
     FROM movie_genre mg JOIN genres g ON g.id=mg.genre_id
    WHERE mg.movie_id=m.id) AS genres,
  m.director, m."cast",
  m.tmdb_overview, m.omdb_overview
FROM movies m
ORDER BY m.id;
"""

def txt(r):
    _, t, ty, gen, d, c, tov, oov = r
    parts = [t or ""]
    if ty: parts.append(f"Release Date: {ty}")
    if gen: parts.append(f"Genres: {gen}")
    if d:   parts.append(f"Director: {d}")
    if c:   parts.append(f"Cast: {c}")
    # keep both overviews (dedupe if identical)
    tov = (tov or "").strip(); oov = (oov or "").strip()
    if tov: parts.append(f"TMDB: {tov}")
    if oov and oov != tov: parts.append(f"OMDB: {oov}")
    return " | ".join(p for p in parts if p)

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    rows = sqlite3.connect(DB_PATH).execute(SQL).fetchall()
    if not rows:
        print("[faiss] no movies"); return
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    model = SentenceTransformer(EMB_MODEL, device="cpu")
    B, vecs = 256, []
    for i in trange(0, len(rows), B, desc="Embedding"):
        batch = [txt(r) for r in rows[i:i+B]]
        vecs.append(model.encode(batch, normalize_embeddings=True, show_progress_bar=False).astype("float32"))
    X = np.vstack(vecs); d = X.shape[1]
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(d))
    index.add_with_ids(X, ids)
    faiss.write_index(index, FAISS_OUT)
    print(f"[faiss] wrote {FAISS_OUT}")

if __name__ == "__main__":
    main()
