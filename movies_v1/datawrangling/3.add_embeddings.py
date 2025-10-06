#!/usr/bin/env python3
import os, sqlite3, numpy as np, faiss
from tqdm import trange
from sentence_transformers import SentenceTransformer

DB_PATH   = os.getenv("DB_PATH_V1", "movies.db")
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SQL = """
SELECT m.id, m.title, m.year,
       (SELECT GROUP_CONCAT(g.name, ', ') FROM movie_genre mg JOIN genres g ON g.id=mg.genre_id WHERE mg.movie_id=m.id),
       m.director, m."cast", m.overview
FROM movies m ORDER BY m.id;
"""

def txt(r):
    _, t, y, gen, d, c, o = r
    parts = [t or ""]
    if y is not None: parts.append(f"({y})")
    if gen: parts.append(f"Genres: {gen}")
    if d: parts.append(f"Director: {d}")
    if c: parts.append(f"Cast: {c}")
    if o: parts.append(o)
    return " | ".join(p for p in parts if p)

def main():
    os.makedirs("index", exist_ok=True)
    rows = sqlite3.connect(DB_PATH).execute(SQL).fetchall()
    if not rows: return print("[faiss] no movies")
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    model = SentenceTransformer(EMB_MODEL, device="cpu")
    B, vecs = 256, []
    for i in trange(0, len(rows), B, desc="Embedding"):
        batch = [txt(r) for r in rows[i:i+B]]
        vecs.append(model.encode(batch, normalize_embeddings=True, show_progress_bar=False).astype("float32"))
    X = np.vstack(vecs); d = X.shape[1]
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(d))
    index.add_with_ids(X, ids)
    faiss.write_index(index, "index/movies.faiss")
    print("[faiss] wrote index/movies.faiss")

if __name__ == "__main__":
    main()
