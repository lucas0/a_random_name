#!/usr/bin/env python3
# Streamlit UI for MovieBot v2
# Env:
#   API_BASE (default: http://127.0.0.1:8000)
#
import os, json, time, requests, logging
import streamlit as st
from logging.handlers import TimedRotatingFileHandler

UI_API_BASE = os.getenv("UI_API_BASE", "http://127.0.0.1:8000")

# ---------- UI Logger (optional local logs) ----------
def setup_ui_logger():
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("ui")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = TimedRotatingFileHandler("logs/ui.jsonl", when="D", backupCount=7, encoding="utf-8")
        fmt = logging.Formatter('{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.propagate = False
    return logger

ui_log = setup_ui_logger()

st.set_page_config(page_title="MovieBot", page_icon="🎬", layout="centered")
st.title("🎬 MovieBot")

with st.sidebar:
    st.markdown("**API**: " + UI_API_BASE)
    topk = st.slider("Top-K retrieved", min_value=3, max_value=20, value=8, step=1)
    st.caption("Logs are written to `logs/` (UI) and API writes to `api/logs/` if configured.")

query = st.text_input("Ask me about movies (title, director, cast, year, or recommendations):", "")

if st.button("Ask") and query.strip():
    ui_log.info(f'{{"event":"submit","len":{len(query)}}}')
    with st.spinner("Thinking..."):
        t0 = time.perf_counter()
        try:
            # 1) /qa — do retrieval + LLM
            r = requests.post(f"{UI_API_BASE}/qa",
                              json={"query": query, "k": topk, "session_id": st.session_state.get("run_id")},
                              timeout=300)  # client timeout 5 min
            r.raise_for_status()
            data = r.json()
            latency = int((time.perf_counter()-t0)*1000)
            ui_log.info(json.dumps({
                "event": "qa",
                "latency_ms": latency,
                "ids": data.get("ids", [])[:10],
                "answer_preview": (data.get("answer","")[:200])
            }))
        except Exception as e:
            st.error(f"Request failed: {e}")
            ui_log.info(json.dumps({"event":"qa_error","error":str(e)}))
        else:
            st.subheader("Answer")
            st.write(data.get("answer","(no answer)"))

            ids = data.get("ids", [])
            if ids:
                # 2) /metadata — show sources
                try:
                    r2 = requests.post(f"{UI_API_BASE}/metadata", json={"ids": ids}, timeout=60)
                    r2.raise_for_status()
                    rows = r2.json().get("movies", [])
                except Exception as e:
                    st.warning(f"Could not load metadata: {e}")
                    rows = []

                if rows:
                    st.subheader("Sources")
                    for r in rows:
                        id = r.get("id")
                        title = r.get("title")
                        year  = r.get("year")
                        director = r.get("director") or ""
                        cast = r.get("cast") or ""
                        genres = r.get("genres") or ""
                        overview = r.get("tmdb_overview") or r.get("omdb_overview") or ""
                        st.markdown(f"**{id} - {title} ({year})**  \n🎦*Director:* {director}  \n🎨*Cast:* {cast}  \n🎭*Genres*: {genres}  \n📜*Overview:* {overview[:400]}{'…' if len(overview)>400 else ''}")
                else:
                    st.info("No source rows returned.")
