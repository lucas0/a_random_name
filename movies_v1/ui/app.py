# app.py
import os, requests, streamlit as st

API_BASE = os.getenv("API_BASE")  # e.g., http://127.0.0.1:8000
if not API_BASE:
    st.error("Set API_BASE in env (e.g., http://127.0.0.1:8000) and restart.")
    st.stop()

st.set_page_config(page_title="Movie QA", layout="wide")
st.title("🎬 Movie QA")

user_input = st.text_input("Ask about movies, directors, genres, years…", "")
k = st.slider("Top-K", 5, 50, 12)

if st.button("Search"):
    if not user_input.strip():
        st.warning("Type something first.")
    else:
        try:
            # 1) ids
            r = requests.get(f"{API_BASE}/search_ids", params={"q": user_input, "k": k}, timeout=30)
            r.raise_for_status()
            ids = r.json()

            # 2) metadata
            r = requests.post(f"{API_BASE}/metadata", json={"ids": ids}, timeout=30)
            r.raise_for_status()
            items = r.json()

            st.subheader("Results")
            for it in items:
                with st.expander(f"{it.get('title','')} ({it.get('year','—')}) — ⭐ {it.get('avg_rating','—')} ({it.get('n_ratings',0)})"):
                    st.write(f"**Genres:** {it.get('genres','—')}")
                    st.write(f"**Director:** {it.get('director','—')}")
                    st.write(f"**Cast:** {it.get('cast','—')}")
                    st.write(it.get("overview",""))

            # 3) answer
            r = requests.post(f"{API_BASE}/answer",
                              json={"user_input": user_input, "items": items[:10]},
                              timeout=120)
            r.raise_for_status()
            ans = r.json().get("answer","")
            st.markdown("### Answer")
            st.write(ans or "_No answer returned._")

        except requests.RequestException as e:
            st.error(f"API error: {e}")
