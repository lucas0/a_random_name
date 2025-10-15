#!/usr/bin/env sh
set -eu

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"

echo "── MovieBot v2 ──"
echo "DB: $DB_PATH_V2"
echo "IDX: $INDEX_V2_DIR"

mkdir -p "$INDEX_V2_DIR"

# helper: does column exist in movies?
have_col() {
  sqlite3 "$DB_PATH_V2" "PRAGMA table_info(movies);" \
  | awk -F'|' -v c="$1" '$2==c{f=1} END{exit f?0:1}'
}

# 1) Create DB if missing
if [ -s "$DB_PATH_V2" ]; then
  echo "[entrypoint] (1) DB exists → skip 1.insert_movielens_into_sqlite.py"
else
  echo "[entrypoint] (1) DB missing → run 1.insert_movielens_into_sqlite.py"
  python /app/movies_v2/datawrangling/1.insert_movielens_into_sqlite.py || { echo "[entrypoint] (1) FAILED"; exit 1; }
  echo "[entrypoint] (1) done."
fi

# 2) TMDB enrichment (adds tmdb_overview, director, cast)
if have_col tmdb_overview; then
  echo "[entrypoint] (2) tmdb_overview present → skip 2.enrich_db_with_tmdb.py"
else
  echo "[entrypoint] (2) tmdb_overview missing → run 2.enrich_db_with_tmdb.py"
  python /app/movies_v2/datawrangling/2.enrich_db_with_tmdb.py || { echo "[entrypoint] (2) FAILED"; exit 1; }
  echo "[entrypoint] (2) done."
fi

# 3) OMDb enrichment (adds omdb_overview; only appends director/cast if empty)
if have_col omdb_overview; then
  echo "[entrypoint] (3) omdb_overview present → skip 3.enrich_db_with_omdb.py"
else
  echo "[entrypoint] (3) omdb_overview missing → run 3.enrich_db_with_omdb.py"
  python /app/movies_v2/datawrangling/3.enrich_db_with_omdb.py || { echo "[entrypoint] (3) FAILED (continuing)"; }
  echo "[entrypoint] (3) done (or best-effort)."
fi

# 4) FAISS index if missing
if [ -s "$INDEX_V2_DIR/movies.faiss" ]; then
  echo "[entrypoint] (4) FAISS exists → skip 4.add_embeddings.py"
else
  echo "[entrypoint] (4) FAISS missing → run 4.add_embeddings.py"
  python /app/movies_v2/datawrangling/4.add_embeddings.py || { echo "[entrypoint] (4) FAILED"; exit 1; }
  [ -s "$INDEX_V2_DIR/movies.faiss" ] && echo "[entrypoint] (4) wrote $INDEX_V2_DIR/movies.faiss" || { echo "[entrypoint] (4) index not found after run"; exit 1; }
fi
echo "API → http://localhost:${API_PORT}"
exec uvicorn movies_v2.api.main:app --host "$API_HOST" --port "$API_PORT" --timeout-keep-alive 300
