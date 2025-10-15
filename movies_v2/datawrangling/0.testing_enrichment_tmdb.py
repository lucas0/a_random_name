#!/usr/bin/env python3
# Env: DB_PATH_V2, TMDB_API_KEY
import os, sys, sqlite3, httpx, re
from typing import Optional, Tuple, List
from tqdm import tqdm

DB_PATH = os.getenv("DB_PATH_V2", "movies.db")
TMDB_KEY = os.getenv("TMDB_API_KEY")
S_URL = "https://api.themoviedb.org/3/search/movie"
M_URL = "https://api.themoviedb.org/3/movie/{id}"
YEAR_RE = re.compile(r"(\d{4})")
DEBUG = False  # set True to see a few errors

if not TMDB_KEY:
    sys.exit("TMDB_API_KEY not set. `source env.sh` first.")
IS_V4 = TMDB_KEY.startswith("eyJ")  # JWT-ish v4 token

def _req(cli: httpx.Client, url: str, params: dict):
    """Support v3 api_key or v4 Bearer auth."""
    if IS_V4:
        return cli.get(url, params=params, headers={"Authorization": f"Bearer {TMDB_KEY}"}, timeout=20)
    else:
        p = dict(params or {}); p["api_key"] = TMDB_KEY
        return cli.get(url, params=p, timeout=20)

def candidates(t: str) -> List[str]:
    t=t.strip(); out=[t]
    a=t.find("("); b=t.find(")",a+1) if a!=-1 else -1
    if a!=-1 and b!=-1:
        inner=t[a+1:b].strip(); outer=t[:a].rstrip()
        if inner: out.append(inner)
        if outer: out.append(outer)
    if ", " in t:
        base,suf=t.rsplit(", ",1)
        if suf: out.append(f"{suf} {base}")
    if ":" in t:
        before,after=t.split(":",1)
        if before.strip(): out.append(before.strip())
        if after.strip():  out.append(after.strip())
    more=[]
    for s in list(out):
        if "&" in s: more.append(s.replace("&","and"))
        if " and " in s: more.append(s.replace(" and "," & "))
    out+=more
    seen=set(); uniq=[]
    for s in out:
        key=s.lower()
        if s and key not in seen:
            seen.add(key); uniq.append(s)
    return uniq

def parse_year_range(s: Optional[str]) -> Optional[Tuple[int,int]]:
    if not s: return None
    ys=[int(x) for x in YEAR_RE.findall(str(s))]
    if not ys: return None
    if len(ys)==1: return (ys[0],ys[0])
    a,b=ys[0],ys[1]
    return (min(a,b), max(a,b))

def year_dist(release_date: Optional[str], target: Optional[int]) -> int:
    if target is None: return 0
    rng=parse_year_range(release_date)
    if rng is None: return 10**9
    a,b=rng
    if a<=target<=b: return 0
    return min(abs(target-a),abs(target-b))

def search(cli, title, year):
    p={"query":title, "include_adult":True}
    if year is not None:
        try: p["year"]=int(year)
        except: pass
    r=_req(cli, S_URL, p)
    if r.status_code!=200:
        if DEBUG: print("search err", r.status_code, r.text[:200], file=sys.stderr)
        return []
    return (r.json() or {}).get("results",[]) or []

def details(cli, tid:int, with_alts=False):
    app="credits,alternative_titles" if with_alts else "credits"
    r=_req(cli, M_URL.format(id=tid), {"append_to_response": app})
    if r.status_code!=200:
        if DEBUG: print("details err", tid, r.status_code, r.text[:200], file=sys.stderr)
        return None
    return r.json()

def any_alt_match(alts: dict, candset: set) -> bool:
    for it in (alts or {}).get("titles", []):
        name=(it.get("title") or "").strip().lower()
        if name and name in candset: return True
    return False

# ---- Methods ----
def method_baseline(cli, title, yref) -> bool:
    res=search(cli, title.strip(), yref)
    if not res: return False
    pick=min(res, key=lambda r: year_dist(r.get("release_date"), yref))
    return bool(details(cli, pick["id"], with_alts=False))

def method_candidates(cli, title, yref) -> bool:
    for cand in candidates(title):
        res=search(cli, cand, yref)
        if not res: continue
        pick=min(res, key=lambda r: year_dist(r.get("release_date"), yref))
        if details(cli, pick["id"], with_alts=False): return True
    return False

def method_alttitles(cli, title, yref) -> bool:
    res=search(cli, title.strip(), yref)
    if not res: return False
    candset=set(s.lower() for s in candidates(title))
    best=False; best_d=10**9
    for r in res[:5]:
        det=details(cli, r["id"], with_alts=True)
        if not det: continue
        if any_alt_match(det.get("alternative_titles", {}), candset):
            return True
        d=year_dist(det.get("release_date"), yref)
        if d<best_d: best=True; best_d=d
    return best

def method_candidates_alttitles(cli, title, yref) -> bool:
    cand_list=candidates(title)
    candset=set(s.lower() for s in cand_list)
    best=False; best_d=10**9
    for q in cand_list:
        res=search(cli, q, yref)
        if not res: continue
        for r in res[:5]:
            det=details(cli, r["id"], with_alts=True)
            if not det: continue
            if any_alt_match(det.get("alternative_titles", {}), candset):
                return True
            d=year_dist(det.get("release_date"), yref)
            if d<best_d: best=True; best_d=d
        if best_d==0: break
    return best

def main():
    conn=sqlite3.connect(DB_PATH); cur=conn.cursor()
    rows=cur.execute("SELECT id, title, year FROM movies ORDER BY id;").fetchall()
    cli=httpx.Client()
    total=len(rows)
    c_base=c_cand=c_alt=c_cand_alt=0
    errs=0

    for _, title, year in tqdm(rows, desc="TMDb probe (4 methods)", unit="movie"):
        yref=int(year) if year is not None else None
        try:
            if method_baseline(cli, title, yref):               c_base+=1
            if method_candidates(cli, title, yref):             c_cand+=1
            if method_alttitles(cli, title, yref):              c_alt+=1
            if method_candidates_alttitles(cli, title, yref):   c_cand_alt+=1
        except Exception as e:
            errs+=1
            if DEBUG and errs<5:
                print("EXC", title, repr(e), file=sys.stderr)
            continue

    print("\n=== TMDb Probe Summary ===")
    print(f"Total movies:                        {total}")
    print(f"Baseline (title only):               {c_base}")
    print(f"Candidates (title fixes):            {c_cand}")
    print(f"Alternative titles (orig title):     {c_alt}")
    print(f"Candidates + alternative titles:     {c_cand_alt}")

if __name__=="__main__":
    main()
