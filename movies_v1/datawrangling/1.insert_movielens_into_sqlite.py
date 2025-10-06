#!/usr/bin/env python3
import os, csv, sqlite3
from pathlib import Path
import kagglehub
import pandas as pd

# ---------- helpers ----------
def split_title_year(title: str):
    if isinstance(title, str) and title.endswith(")"):
        i = title.rfind("(")
        if i != -1:
            y = title[i+1:-1]
            if y.isdigit():
                return title[:i].strip(), int(y)
    return title, None

def read_genre_index(file_path: Path) -> dict[int, str]:
    idx2name: dict[int, str] = {}
    with file_path.open("r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line: continue
            name, idx = line.split("|", 1)
            if idx.isdigit():
                idx2name[int(idx)] = name
    return idx2name

def read_movies_and_url(file_path: Path, idx2name: dict[int, str]) -> pd.DataFrame:
    rows = []
    with file_path.open("r", encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="|")
        for r in reader:
            if not r: continue
            mid = int(r[0])
            raw_title = r[1]
            imdb_url = r[4] if len(r) > 4 else None
            title, year = split_title_year(raw_title)
            flags = r[-19:]
            genres = [idx2name[i] for i, flag in enumerate(flags)
                      if flag.isdigit() and int(flag)==1 and i in idx2name]
            rows.append({
                "movieId": mid,
                "title": title,
                "year": year,
                "genres": genres,
                "imdb_url": imdb_url
            })
    return pd.DataFrame(rows)

def read_ratings(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file_path, sep="\t",
        names=["userId","movieId","rating","timestamp"],
        engine="python"
    )
    return df[["userId","movieId","rating"]].astype({"userId":int,"movieId":int,"rating":float})

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

def insert_all(conn: sqlite3.Connection, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> None:
    cur = conn.cursor()

    # Pre-compute avg rating by source movieId
    avg_by_mid = ratings_df.groupby("movieId")["rating"].mean().to_dict()

    genre_cache: dict[str,int] = {}
    src_to_new: dict[int,int] = {}

    def get_gid(name: str) -> int:
        if name in genre_cache: return genre_cache[name]
        cur.execute("INSERT OR IGNORE INTO genres(name) VALUES (?)", (name,))
        cur.execute("SELECT id FROM genres WHERE name=?", (name,))
        gid = cur.fetchone()[0]
        genre_cache[name] = gid
        return gid

    # Insert movies with avg_rating and imdb_url
    for _, row in movies_df.iterrows():
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
        src_to_new[int(row["movieId"])] = new_id

        for g in (row["genres"] or []):
            gid = get_gid(g)
            cur.execute("INSERT OR IGNORE INTO movie_genre(movie_id,genre_id) VALUES (?,?)", (new_id, gid))

    # Insert ratings mapped to new ids
    batch = []
    for _, r in ratings_df.iterrows():
        mid_src = int(r["movieId"])
        if mid_src not in src_to_new: continue
        batch.append((src_to_new[mid_src], int(r["userId"]), float(r["rating"])))
        if len(batch) >= 50_000:
            cur.executemany("INSERT INTO ratings(movie_id,user_id,rating) VALUES (?,?,?)", batch)
            batch.clear()
    if batch:
        cur.executemany("INSERT INTO ratings(movie_id,user_id,rating) VALUES (?,?,?)", batch)

    conn.commit()

def main():
    slug   = os.getenv("DATASET_SLUG", "prajitdatta/movielens-100k-dataset")
    subdir = os.getenv("DATA_SUBDIR", "ml-100k")
    dbpath = Path(os.getenv("DB_PATH_V1", "movies.db"))

    base = Path(kagglehub.dataset_download(slug))
    data_dir = base / subdir
    print(f"[kagglehub] using: {data_dir}")

    u_item  = data_dir / "u.item"
    u_data  = data_dir / "u.data"
    u_genre = data_dir / "u.genre"
    for p in (u_item, u_data, u_genre):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    print("[ingest] reading genres…")
    idx2name = read_genre_index(u_genre)

    print("[ingest] reading movies & imdb_url…")
    movies_df = read_movies_and_url(u_item, idx2name)
    print(f"[ingest] movies: {len(movies_df)}")

    print("[ingest] reading ratings…")
    ratings_df = read_ratings(u_data)
    print(f"[ingest] ratings: {len(ratings_df)}")

    print(f"[db] writing to {dbpath} …")
    conn = sqlite3.connect(dbpath)
    try:
        init_db(conn)
        insert_all(conn, movies_df, ratings_df)
    finally:
        conn.close()
    print(f"[done] SQLite ready at: {dbpath}")

if __name__ == "__main__":
    main()
