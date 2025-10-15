#!/usr/bin/env python3
import os, csv, json, sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import kagglehub
import pandas as pd

# =========================
# Helpers
# =========================

def split_title_year(title: str) -> Tuple[str, Optional[int]]:
    if isinstance(title, str) and title.endswith(")"):
        i = title.rfind("(")
        if i != -1:
            y = title[i+1:-1]
            if y.isdigit():
                return title[:i].strip(), int(y)
    return title, None

def read_genre_index(file_path: Path) -> Dict[int, str]:
    idx2name: Dict[int, str] = {}
    with file_path.open("r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            name, idx = line.split("|", 1)
            if idx.isdigit():
                idx2name[int(idx)] = name
    return idx2name

def read_ml_movies(file_path: Path, idx2name: Dict[int, str]) -> pd.DataFrame:
    """MovieLens u.item with imdb_url and one-hot genre flags → DataFrame
       columns: movieId, title, year, genres(list[str]), imdb_url"""
    rows: List[Dict] = []
    with file_path.open("r", encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="|")
        for r in reader:
            if not r:
                continue
            mid = int(r[0])
            raw_title = r[1]
            imdb_url = r[4] if len(r) > 4 else None
            title, year = split_title_year(raw_title)
            flags = r[-19:]  # last 19 ints indicate genres
            genres = [idx2name[i] for i, flag in enumerate(flags)
                      if flag.isdigit() and int(flag) == 1 and i in idx2name]
            rows.append({
                "movieId": mid,
                "title": title,
                "year": year,
                "genres": genres,
                "imdb_url": imdb_url
            })
    return pd.DataFrame(rows)

def read_ml_ratings(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file_path, sep="\t",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python"
    )
    return df[["userId", "movieId", "rating"]].astype({"userId": int, "movieId": int, "rating": float})

def parse_tmdb_genres(cell: str) -> List[str]:
    """TMDB 5000 genres come as a JSON list of dicts with 'name' keys."""
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        data = json.loads(cell)
        names = [d.get("name") for d in data if isinstance(d, dict) and d.get("name")]
        return [n for n in names if isinstance(n, str)]
    except Exception:
        return []

def year_from_date(date_str: Optional[str]) -> Optional[int]:
    if not date_str or not isinstance(date_str, str):
        return None
    s = date_str.strip()
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return None

def read_tmdb5000_movies(base_dir: Path) -> pd.DataFrame:
    """
    Reads TMDB 5000 movies from common filenames:
    - tmdb_5000_movies.csv  (Kaggle 'TMDB 5000 Movie Dataset')
    or compatible 'movies_metadata' with similar fields.
    Returns DataFrame: columns = title, year, genres(list[str]), imdb_url
    """
    # Common paths
    candidates = [
        base_dir / "tmdb_5000_movies.csv",               # classic dataset
        base_dir / "movies_metadata.csv"                  # alternative dataset name
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Could not find TMDB movies CSV in {base_dir}. "
            "Expected tmdb_5000_movies.csv or movies_metadata.csv"
        )

    df = pd.read_csv(path, low_memory=False)
    # Best-effort column mapping
    # Prefer 'title' else 'original_title'
    title_col = "title" if "title" in df.columns else ("original_title" if "original_title" in df.columns else None)
    if not title_col:
        raise ValueError("TMDB file missing 'title'/'original_title' column.")

    # Genres: JSON in 'genres' if present, else empty
    genres_col = "genres" if "genres" in df.columns else None
    # IMDb id: 'imdb_id' if present
    imdb_col = "imdb_id" if "imdb_id" in df.columns else None
    # Date/year: 'release_date' preferred, else 'year' if numeric
    rdate_col = "release_date" if "release_date" in df.columns else None

    out_rows: List[Dict] = []
    for _, r in df.iterrows():
        title = str(r.get(title_col) or "").strip()
        if not title:
            continue
        # year
        y = year_from_date(str(r.get(rdate_col))) if rdate_col else None
        if y is None and "year" in df.columns:
            try:
                yv = int(r.get("year"))
                y = yv
            except Exception:
                pass
        # genres
        if genres_col:
            gs = parse_tmdb_genres(r.get(genres_col))
        else:
            gs = []
        # imdb_url
        imdb_url = None
        if imdb_col:
            imdb_id = r.get(imdb_col)
            if isinstance(imdb_id, str) and imdb_id.startswith("tt"):
                imdb_url = f"https://www.imdb.com/title/{imdb_id}/"

        out_rows.append({
            # align with ML schema (no source id used for TMDB merge)
            "title": title,
            "year": y,
            "genres": gs,
            "imdb_url": imdb_url
        })

    return pd.DataFrame(out_rows)

def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript("""
        PRAGMA foreign_keys = ON;

        DROP TABLE IF EXISTS movie_genre;
        DROP TABLE IF EXISTS ratings;
        DROP TABLE IF EXISTS genres;
        DROP TABLE IF EXISTS movies;

        CREATE TABLE movies (
            id         INTEGER PRIMARY KEY,
            title      TEXT NOT NULL,
            year       INTEGER,
            avg_rating REAL,
            imdb_url   TEXT
        );

        CREATE TABLE genres (
            id    INTEGER PRIMARY KEY,
            name  TEXT NOT NULL UNIQUE
        );

        CREATE TABLE movie_genre (
            movie_id INTEGER NOT NULL,
            genre_id INTEGER NOT NULL,
            PRIMARY KEY (movie_id, genre_id),
            FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE,
            FOREIGN KEY (genre_id) REFERENCES genres(id) ON DELETE CASCADE
        );

        CREATE TABLE ratings (
            id       INTEGER PRIMARY KEY,
            movie_id INTEGER NOT NULL,
            user_id  INTEGER NOT NULL,
            rating   REAL    NOT NULL,
            FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_movies_title ON movies(title);
        CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movie_id);
    """)
    conn.commit()

def insert_all(
    conn: sqlite3.Connection,
    ml_movies_df: pd.DataFrame,
    ml_ratings_df: pd.DataFrame,
    tmdb_movies_df: pd.DataFrame
) -> None:
    cur = conn.cursor()

    # 1) Pre-compute ML avg ratings by ML movieId
    avg_by_mid = ml_ratings_df.groupby("movieId")["rating"].mean().to_dict()

    # caches
    genre_cache: Dict[str, int] = {}
    ml_src_to_new: Dict[int, int] = {}
    # keep to dedupe TMDB against ML inserts: (title_norm, year) → new_id
    key_to_new: Dict[Tuple[str, Optional[int]], int] = {}

    def get_gid(name: str) -> int:
        if name in genre_cache:
            return genre_cache[name]
        cur.execute("INSERT OR IGNORE INTO genres(name) VALUES (?)", (name,))
        cur.execute("SELECT id FROM genres WHERE name=?", (name,))
        gid = cur.fetchone()[0]
        genre_cache[name] = gid
        return gid

    def norm_key(title: Optional[str], year: Optional[int]) -> Tuple[str, Optional[int]]:
        t = (title or "").strip().lower()
        return (t, year)

    # 2) Insert MovieLens movies (with avg_rating and imdb_url)
    for _, row in ml_movies_df.iterrows():
        avg = float(avg_by_mid.get(int(row["movieId"]), float("nan")))
        cur.execute("""
            INSERT INTO movies(title, year, avg_rating, imdb_url)
            VALUES (?,?,?,?)
        """, (
            row["title"],
            int(row["year"]) if pd.notna(row["year"]) else None,
            (avg if pd.notna(avg) else None),
            row.get("imdb_url")
        ))
        new_id = cur.lastrowid
        ml_src_to_new[int(row["movieId"])] = new_id
        key_to_new[norm_key(row["title"], int(row["year"]) if pd.notna(row["year"]) else None)] = new_id

        for g in (row["genres"] or []):
            gid = get_gid(g)
            cur.execute(
                "INSERT OR IGNORE INTO movie_genre(movie_id, genre_id) VALUES (?,?)",
                (new_id, gid)
            )

    # 3) Insert TMDB movies (no ratings here; avoid duplicates by (title,year))
    inserted_tmdb = 0
    for _, row in tmdb_movies_df.iterrows():
        title = (row.get("title") or "").strip()
        year  = row.get("year")
        k = norm_key(title, int(year) if pd.notna(year) else None)
        if k in key_to_new:
            # Already present from MovieLens; optionally add any new genres from TMDB
            existing_id = key_to_new[k]
            for g in (row.get("genres") or []):
                gid = get_gid(g)
                cur.execute(
                    "INSERT OR IGNORE INTO movie_genre(movie_id, genre_id) VALUES (?,?)",
                    (existing_id, gid)
                )
            continue

        cur.execute("""
            INSERT INTO movies(title, year, avg_rating, imdb_url)
            VALUES (?,?,?,?)
        """, (
            title or None,
            int(year) if pd.notna(year) else None,
            None,  # do not mix TMDB vote_average into avg_rating to keep scale consistent
            row.get("imdb_url")
        ))
        new_id = cur.lastrowid
        key_to_new[k] = new_id
        inserted_tmdb += 1

        for g in (row.get("genres") or []):
            gid = get_gid(g)
            cur.execute(
                "INSERT OR IGNORE INTO movie_genre(movie_id, genre_id) VALUES (?,?)",
                (new_id, gid)
            )

    # 4) Insert MovieLens ratings mapped to new ids
    batch = []
    for _, r in ml_ratings_df.iterrows():
        mid_src = int(r["movieId"])
        if mid_src not in ml_src_to_new:
            continue
        batch.append((ml_src_to_new[mid_src], int(r["userId"]), float(r["rating"])))
        if len(batch) >= 50_000:
            cur.executemany("INSERT INTO ratings(movie_id, user_id, rating) VALUES (?,?,?)", batch)
            batch.clear()
    if batch:
        cur.executemany("INSERT INTO ratings(movie_id, user_id, rating) VALUES (?,?,?)", batch)

    conn.commit()
    print(f"[merge] inserted {len(ml_movies_df)} MovieLens movies, "
          f"added {inserted_tmdb} TMDB movies, total genres={len(genre_cache)}")

# =========================
# Main
# =========================

def main():
    # --- Env / paths ---
    dbpath = Path(os.getenv("DB_PATH_V2", "movies.db"))

    # MovieLens config
    ml_slug   = os.getenv("ML_SLUG", "prajitdatta/movielens-100k-dataset")
    ml_subdir = os.getenv("ML_SUBDIR", "ml-100k")

    # TMDB 5000 config (dataset slug can vary; keep it configurable)
    tmdb_slug = os.getenv("TMDB5000_SLUG", "tmdb/tmdb-movie-metadata")
    tmdb_subdir = os.getenv("TMDB5000_SUBDIR", "")  # often root; allow override

    # --- Download datasets via kagglehub ---
    ml_base   = Path(kagglehub.dataset_download(ml_slug))
    ml_dir    = ml_base / ml_subdir
    print(f"[kagglehub] MovieLens path: {ml_dir}")

    u_item  = ml_dir / "u.item"
    u_data  = ml_dir / "u.data"
    u_genre = ml_dir / "u.genre"
    for p in (u_item, u_data, u_genre):
        if not p.exists():
            raise FileNotFoundError(f"Missing MovieLens file: {p}")

    # TMDB download + locate CSV
    tmdb_base = Path(kagglehub.dataset_download(tmdb_slug))
    tmdb_dir  = tmdb_base / tmdb_subdir if tmdb_subdir else tmdb_base
    print(f"[kagglehub] TMDB path: {tmdb_dir}")

    # --- Read MovieLens ---
    print("[ingest-ml] reading genres …")
    idx2name = read_genre_index(u_genre)

    print("[ingest-ml] reading movies & imdb_url …")
    ml_movies_df = read_ml_movies(u_item, idx2name)
    print(f"[ingest-ml] movies: {len(ml_movies_df)}")

    print("[ingest-ml] reading ratings …")
    ml_ratings_df = read_ml_ratings(u_data)
    print(f"[ingest-ml] ratings: {len(ml_ratings_df)}")

    # --- Read TMDB 5000 movies ---
    print("[ingest-tmdb] reading TMDB 5000 …")
    tmdb_movies_df = read_tmdb5000_movies(tmdb_dir)
    print(f"[ingest-tmdb] movies: {len(tmdb_movies_df)}")

    # --- Write DB ---
    print(f"[db] writing to {dbpath} …")
    conn = sqlite3.connect(dbpath)
    try:
        init_db(conn)
        insert_all(conn, ml_movies_df, ml_ratings_df, tmdb_movies_df)
    finally:
        conn.close()
    print(f"[done] SQLite ready at: {dbpath}")

if __name__ == "__main__":
    main()
