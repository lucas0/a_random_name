#!/usr/bin/env python3
# Env: DB_PATH=/path/to/movies.db  OMDB_API_KEY=...
import os, sys, re, sqlite3
from typing import List, Optional, Tuple
import httpx
from tqdm import tqdm

DB_PATH  = os.getenv("DB_PATH_V1", "movies.db")
API_KEY  = os.getenv("OMDB_API_KEY")
OMDB_URL = "https://www.omdbapi.com/"

POSSESSIVE_RE   = re.compile(r"^[A-Za-z .'-]+?'s\s+(.*)$")
YEAR_TOKEN_RE   = re.compile(r"(\d{4})")

def first_parenthetical(t: str) -> Optional[str]:
    a = t.find("("); b = t.find(")", a + 1) if a != -1 else -1
    if a == -1 or b == -1: return None
    s = t[a+1:b].strip()
    return s or None

def outside_before_paren(t: str) -> str:
    a = t.find("(")
    return t if a == -1 else t[:a].rstrip()

def comma_to_prefix(t: str) -> Optional[str]:
    if ", " not in t: return None
    base, suffix = t.rsplit(", ", 1)
    return f"{suffix} {base}" if suffix else None

def prefix_before_colon(t: str) -> Optional[str]:
    if ":" not in t: return None
    s = t.split(":", 1)[0].strip()
    return s or None

def strip_author_possessive(t: str) -> Optional[str]:
    m = POSSESSIVE_RE.match(t)
    return (m.group(1).strip() or None) if m else None

def ampersand_variants(t: str) -> List[str]:
    out = [t]
    if "&" in t: out.append(t.replace("&", "and"))
    if " and " in t: out.append(t.replace(" and ", " & "))
    seen, uniq = set(), []
    for s in out:
        if s and s not in seen:
            seen.add(s); uniq.append(s)
    return uniq

def build_candidates(original: str) -> List[str]:
    bases: List[str] = []
    inner = first_parenthetical(original)
    outer = outside_before_paren(original)

    if inner:
        bases.append(inner)
        cv = comma_to_prefix(inner)
        if cv: bases.append(cv)
    if outer:
        bases.append(outer)
        cv2 = comma_to_prefix(outer)
        if cv2: bases.append(cv2)

    extra: List[str] = []
    for b in list(bases):
        ap = strip_author_possessive(b)
        if ap: extra.append(ap)
        pc = prefix_before_colon(b)
        if pc: extra.append(pc)
    bases.extend(extra)

    # dedupe then expand &/and variants
    seen, uniq = set(), []
    for s in bases:
        if s and s not in seen:
            seen.add(s); uniq.append(s)

    out: List[str] = []
    seen2 = set()
    for s in uniq:
        for v in ampersand_variants(s):
            if v not in seen2:
                seen2.add(v); out.append(v)
    return out or [original]

# ---- Year helpers (handles ranges like "1995–1999" or "1995-1999") ----
def parse_omdb_year_field(y: Optional[str]) -> Optional[Tuple[int, int]]:
    if not y: return None
    toks = YEAR_TOKEN_RE.findall(str(y))
    if not toks: return None
    a = int(toks[0]); b = int(toks[1]) if len(toks) > 1 else a
    if b < a: b = a
    return (a, b)

def year_matches_or_inside_range(omdb_year: Optional[str], target_year: Optional[int]) -> bool:
    rng = parse_omdb_year_field(omdb_year)
    if rng is None or target_year is None: return False
    a, b = rng
    return a <= target_year <= b

def distance_to_range(omdb_year: Optional[str], target_year: Optional[int]) -> Optional[int]:
    rng = parse_omdb_year_field(omdb_year)
    if rng is None or target_year is None: return None
    a, b = rng
    if target_year < a:  return a - target_year
    if target_year > b:  return target_year - b
    return 0

# ---- OMDb fetch ----
def fetch_title_year(client: httpx.Client, title: str, year: Optional[int]) -> Optional[dict]:
    params = {"t": title, "plot": "short", "type": "movie", "apikey": API_KEY}
    if year is not None:
        try: params["y"] = int(year)
        except: pass
    r = client.get(OMDB_URL, params=params, timeout=20)
    if r.status_code != 200: return None
    d = r.json()
    return d if d.get("Response") == "True" else None

def fetch_title_only(client: httpx.Client, title: str) -> Optional[dict]:
    r = client.get(OMDB_URL, params={"t": title, "plot": "short", "type": "movie", "apikey": API_KEY}, timeout=20)
    if r.status_code != 200: return None
    d = r.json()
    return d if d.get("Response") == "True" else None

def main():
    if not API_KEY:
        sys.exit("OMDB_API_KEY not set. Add it to env.sh and `source env.sh`.")

    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()

    # Ensure columns exist (quote "cast")
    cols = {c[1] for c in cur.execute("PRAGMA table_info(movies);")}
    for col in ("imdb_id", "overview", "director", "cast", "omdb_year"):
        if col not in cols:
            cur.execute(f'ALTER TABLE movies ADD COLUMN "{col}" TEXT;')
    conn.commit()

    # Fresh DB after insert: enrich anything missing
    rows = cur.execute("""
        SELECT id, title, year, imdb_id, overview, director, "cast", omdb_year
        FROM movies
        WHERE imdb_id IS NULL OR overview IS NULL OR director IS NULL OR "cast" IS NULL OR omdb_year IS NULL
        ORDER BY id;
    """).fetchall()

    client = httpx.Client()
    updated = 0

    with tqdm(total=len(rows), desc="OMDb enrich", unit="movie") as bar:
        for mid, title, year, imdb_id, overview, director, cast, omdb_year_existing in rows:
            target_year = int(year) if year is not None else None
            data = None
            chosen_year_str: Optional[str] = None

            # Phase 1: title+year (must match/contain the target year)
            for cand in build_candidates(title):
                d = fetch_title_year(client, cand, target_year)
                if d and year_matches_or_inside_range(d.get("Year"), target_year):
                    data = d; chosen_year_str = d.get("Year"); break

            # Phase 2: title-only; choose closest year/range
            if data is None:
                best = None  # (distance, dict)
                for cand in build_candidates(title):
                    d = fetch_title_only(client, cand)
                    if not d: continue
                    dist = distance_to_range(d.get("Year"), target_year)
                    if dist is None: continue
                    if (best is None) or (dist < best[0]):
                        best = (dist, d)
                        if dist == 0: break
                if best:
                    data = best[1]; chosen_year_str = data.get("Year")

            if not data:
                bar.update(1); continue

            # only store omdb_year if DB year is NOT contained by OMDb year range
            omdb_year_to_store = None
            if chosen_year_str and not year_matches_or_inside_range(chosen_year_str, target_year):
                omdb_year_to_store = chosen_year_str

            cur.execute("""
                UPDATE movies
                SET imdb_id = COALESCE(?, imdb_id),
                    overview = COALESCE(?, overview),
                    director = COALESCE(?, director),
                    "cast"   = COALESCE(?, "cast"),
                    omdb_year = COALESCE(?, omdb_year)
                WHERE id = ?
            """, (
                data.get("imdbID") or imdb_id,
                data.get("Plot") or overview,
                data.get("Director") or director,
                data.get("Actors") or cast,
                omdb_year_to_store,
                mid
            ))
            updated += 1
            if updated % 100 == 0: conn.commit()
            bar.set_postfix(updated=updated, refresh=True)
            bar.update(1)

    conn.commit(); conn.close()
    print(f"[done] updated {updated} rows")

if __name__ == "__main__":
    main()
