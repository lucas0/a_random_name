#!/usr/bin/env python3
# Env: DB_PATH_V2, OMDB_API_KEY, [optional] OMDB_CONC, OMDB_TIMEOUT, OMDB_RETRIES
import os, re, sqlite3, asyncio, random, json
from typing import Optional, List, Tuple
import httpx
from tqdm.asyncio import tqdm

DB   = os.getenv("DB_PATH_V2", "movies.db")
KEY  = os.getenv("OMDB_API_KEY") or ""
URL  = "https://www.omdbapi.com/"
CONC = int(os.getenv("OMDB_CONC", "4"))           # parallel requests
TOUT = float(os.getenv("OMDB_TIMEOUT", "30"))     # per request
RETR = int(os.getenv("OMDB_RETRIES", "3"))        # retries on 5xx/timeout

YR_RE  = re.compile(r"(\d{4})")
POS_RE = re.compile(r"^[A-Za-z .'-]+?'s\s+(.*)$")

def candidates(title: str) -> List[str]:
    t = title.strip()
    out = [t]
    a = t.find("("); b = t.find(")", a+1) if a!=-1 else -1
    if a!=-1 and b!=-1:
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
    m = POS_RE.match(t)
    if m and m.group(1).strip(): out.append(m.group(1).strip())
    more = []
    for s in list(out):
        if "&" in s: more.append(s.replace("&", "and"))
        if " and " in s: more.append(s.replace(" and ", " & "))
    out += more
    # dedupe keep order
    seen, uniq = set(), []
    for s in out:
        if s and s not in seen:
            seen.add(s); uniq.append(s)
    return uniq

def year_range(omdb_year: Optional[str]) -> Optional[Tuple[int,int]]:
    if not omdb_year: return None
    toks = YR_RE.findall(str(omdb_year))
    if not toks: return None
    a = int(toks[0]); b = int(toks[1]) if len(toks)>1 else a
    return (a, b if b>=a else a)

def yr_dist(omdb_year: Optional[str], y: Optional[int]) -> Optional[int]:
    if y is None: return None
    r = year_range(omdb_year)
    if not r: return None
    a,b = r
    return 0 if a<=y<=b else (a-y if y<a else y-b)

async def get_omdb(cli: httpx.AsyncClient, params: dict) -> Optional[dict]:
    # retry + small jitter backoff on transient errors
    for i in range(RETR):
        try:
            r = await cli.get(URL, params=params, timeout=TOUT)
            if r.status_code in (429, 502, 503, 504):
                await asyncio.sleep((2**i)*0.5 + random.random()*0.25); continue
            r.raise_for_status()
            d = r.json()
            return d if d.get("Response") == "True" else None
        except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError):
            await asyncio.sleep((2**i)*0.5 + random.random()*0.25)
        except Exception:
            return None
    return None

async def fetch_best(cli: httpx.AsyncClient, title: str, year: Optional[int]) -> Optional[dict]:
    # phase 1: title+year, require range match
    for cand in candidates(title):
        d = await get_omdb(cli, {"t": cand, "type":"movie", "plot":"short", "apikey":KEY, "y": year or ""})
        if d and yr_dist(d.get("Year"), year) == 0:
            return d
    # phase 2: title only, min distance to year (0 best)
    best = (10**9, None)
    for cand in candidates(title):
        d = await get_omdb(cli, {"t": cand, "type":"movie", "plot":"short", "apikey":KEY})
        if not d: continue
        dist = yr_dist(d.get("Year"), year)
        if dist is None: continue
        if dist < best[0]: best = (dist, d)
        if dist == 0: break
    return best[1]

def ensure_columns(cur: sqlite3.Cursor):
    cols = {c[1] for c in cur.execute("PRAGMA table_info(movies);")}
    for col in ("omdb_overview","director","cast"):
        if col not in cols:
            cur.execute(f'ALTER TABLE movies ADD COLUMN "{col}" TEXT;')

def need_rows(cur: sqlite3.Cursor):
    return cur.execute("""
        SELECT id, title, year, director, "cast", omdb_overview
          FROM movies
         WHERE omdb_overview IS NULL OR director IS NULL OR "cast" IS NULL
         ORDER BY id
    """).fetchall()

async def worker(sem, cli, row, updates):
    mid, title, year, director, cast, ov = row
    if director and cast and ov:  # already complete
        return
    async with sem:
        d = await fetch_best(cli, title, int(year) if year is not None else None)
    if not d: return
    new_dir = d.get("Director") if not director else None
    new_cast= d.get("Actors")   if not cast     else None
    new_ov  = d.get("Plot")     if not ov       else None
    if any((new_dir, new_cast, new_ov)):
        updates.append((new_dir, new_cast, new_ov, mid))

async def main_async():
    if not KEY: raise SystemExit("OMDB_API_KEY not set.")
    conn = sqlite3.connect(DB); cur = conn.cursor()
    ensure_columns(cur); conn.commit()
    rows = need_rows(cur)
    if not rows:
        print("[omdb] nothing to do"); conn.close(); return
    sem = asyncio.Semaphore(CONC)
    async with httpx.AsyncClient(headers={"Accept":"application/json"}) as cli:
        updates: List[tuple] = []
        for row in await tqdm.gather(*[
            worker(sem, cli, r, updates) for r in rows
        ], desc="OMDb enrich (append-only)", total=len(rows)):
            pass
    if updates:
        cur.executemany("""
            UPDATE movies
               SET director      = COALESCE(?, director),
                   "cast"        = COALESCE(?, "cast"),
                   omdb_overview = COALESCE(?, omdb_overview)
             WHERE id = ?;
        """, updates)
        conn.commit()
    conn.close()
    print(f"[done] OMDb-enriched (appended) {len(updates)} rows")

if __name__ == "__main__":
    asyncio.run(main_async())
