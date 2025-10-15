#!/usr/bin/env python3
# Env: DB_PATH_V2, TMDB_API_KEY
import os, sys, sqlite3, httpx, re
from typing import Optional, Tuple, List
from tqdm import tqdm

DB_PATH  = os.getenv("DB_PATH_V2", "movies.db")
TMDB_KEY = os.getenv("TMDB_API_KEY")
S_URL    = "https://api.themoviedb.org/3/search/movie"
M_URL    = "https://api.themoviedb.org/3/movie/{id}"
YEAR_RE  = re.compile(r"(\d{4})")

if not TMDB_KEY:
    sys.exit("TMDB_API_KEY not set. Add to env.sh and `source env.sh`.")
IS_V4 = TMDB_KEY.startswith("eyJ")  # naive check for v4 Bearer token

def _req(cli: httpx.Client, url: str, params: dict):
    if IS_V4:
        return cli.get(url, params=params, headers={"Authorization": f"Bearer {TMDB_KEY}"}, timeout=20)
    p = dict(params or {}); p["api_key"] = TMDB_KEY
    return cli.get(url, params=p, timeout=20)

def candidates(title: str) -> List[str]:
    t = title.strip()
    out = [t]
    a = t.find("("); b = t.find(")", a+1) if a != -1 else -1
    if a != -1 and b != -1:
        inner = t[a+1:b].strip()
        if inner: out.append(inner)
        outer = t[:a].rstrip()
        if outer: out.append(outer)
    if ", " in t:
        base, suf = t.rsplit(", ", 1)
        if suf: out.append(f"{suf} {base}")
    if ":" in t:
        before, after = t.split(":", 1)
        if before.strip(): out.append(before.strip())
        if after.strip():  out.append(after.strip())
    more = []
    for s in list(out):
        if "&" in s: more.append(s.replace("&", "and"))
        if " and " in s: more.append(s.replace(" and ", " & "))
    out += more
    seen, uniq = set(), []
    for s in out:
        key = s.lower()
        if s and key not in seen:
            seen.add(key); uniq.append(s)
    return uniq

def parse_year_range(s: Optional[str]) -> Optional[Tuple[int,int]]:
    if not s: return None
    ys = [int(x) for x in YEAR_RE.findall(str(s))]
    if not ys: return None
    if len(ys) == 1: return (ys[0], ys[0])
    a, b = ys[0], ys[1]
    return (min(a,b), max(a,b))

def year_dist(release_date: Optional[str], target: Optional[int]) -> int:
    if target is None: return 0
    rng = parse_year_range(release_date)
    if rng is None: return 10**9
    a, b = rng
    if a <= target <= b: return 0
    return min(abs(target - a), abs(target - b))

def search_tmdb(cli: httpx.Client, title: str, year: Optional[int]):
    p = {"query": title, "include_adult": True}
    if year is not None:
        try: p["year"] = int(year)
        except: pass
    r = _req(cli, S_URL, p)
    if r.status_code != 200: return []
    return (r.json() or {}).get("results", []) or []

def details_with_credits(cli: httpx.Client, tmdb_id: int):
    r = _req(cli, M_URL.format(id=tmdb_id),
             {"append_to_response": "credits"})
    return r.json() if r.status_code == 200 else None

def pick_best(results, target_year: Optional[int]):
    if not results: return None
    if target_year is None: return results[0]
    return min(results, key=lambda r: year_dist(r.get("release_date"), target_year))

def extract_director(credits: dict) -> Optional[str]:
    for c in (credits or {}).get("crew", []):
        if c.get("job") == "Director" and c.get("name"):
            return c["name"]
    return None

def cast_with_roles(credits: dict, top_n: int = 10) -> Optional[str]:
    items = []
    for m in (credits or {}).get("cast", [])[:top_n]:
        nm = m.get("name"); ch = m.get("character")
        if nm and ch: items.append(f"{nm} ({ch})")
        elif nm:      items.append(nm)
    return ", ".join(items) if items else None

def main():
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    # Ensure columns exist
    cols = {c[1] for c in cur.execute("PRAGMA table_info(movies);")}
    for col in ("tmdb_id","tmdb_title","tmdb_overview","tmdb_year","director","cast"):
        if col not in cols:
            cur.execute(f'ALTER TABLE movies ADD COLUMN "{col}" TEXT;')
    conn.commit()

    rows = cur.execute("""
      SELECT id, title, year
      FROM movies
      WHERE tmdb_id IS NULL OR tmdb_title IS NULL OR tmdb_overview IS NULL OR tmdb_year IS NULL
            OR director IS NULL OR "cast" IS NULL
      ORDER BY id;
    """).fetchall()

    cli = httpx.Client()
    updated = 0

    with tqdm(total=len(rows), desc="TMDb enrich (candidates)", unit="movie") as bar:
        for mid, title, year in rows:
            yref = int(year) if year is not None else None

            best = None
            for cand in candidates(title):
                res = search_tmdb(cli, cand, yref)
                best = pick_best(res, yref)
                if best: break
            if not best: bar.update(1); continue

            det = details_with_credits(cli, best["id"])
            if not det: bar.update(1); continue

            release_date = det.get("release_date") or None  # store full string
            dir_name     = extract_director(det.get("credits", {}))
            cast_str     = cast_with_roles(det.get("credits", {}))

            cur.execute("""
              UPDATE movies
                 SET tmdb_id       = COALESCE(?, tmdb_id),
                     tmdb_title    = COALESCE(?, tmdb_title),
                     tmdb_overview = COALESCE(?, tmdb_overview),
                     tmdb_year     = COALESCE(?, tmdb_year),
                     director      = COALESCE(?, director),
                     "cast"        = COALESCE(?, "cast")
               WHERE id = ?;
            """, (
                str(det.get("id")) if det.get("id") is not None else None,
                det.get("title"),
                det.get("overview"),
                release_date,
                dir_name,
                cast_str,     # cast includes roles; keeps existing if already present
                mid
            ))
            updated += 1
            if updated % 100 == 0: conn.commit()
            bar.update(1)

    conn.commit(); conn.close()
    print(f"[done] TMDb-enriched {updated} rows (candidates strategy)")

if __name__ == "__main__":
    main()
