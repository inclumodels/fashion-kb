"""
Streamlit frontend for testing the Fashion KB pipeline.
Run: streamlit run frontend/app.py
Requires API server running: uvicorn api.server:app --reload
"""

import time
import httpx
import streamlit as st
import pandas as pd

API = "http://localhost:8000"

st.set_page_config(page_title="Fashion KB", page_icon="👗", layout="wide")
st.title("👗 Fashion Knowledge Base — Phase 1")

# ---------- Sidebar: DB Stats ----------
with st.sidebar:
    st.header("📊 Database Stats")
    if st.button("Refresh Stats"):
        st.session_state["refresh_stats"] = True

    try:
        stats = httpx.get(f"{API}/db/stats", timeout=5).json()
        st.metric("Total Documents", stats.get("total", 0))
        by_type = stats.get("by_type", {})
        if by_type:
            df_stats = pd.DataFrame(
                list(by_type.items()), columns=["Source Type", "Count"]
            )
            st.bar_chart(df_stats.set_index("Source Type"))
        else:
            st.info("No documents yet.")
    except Exception:
        st.warning("API not reachable. Start the server first.")

    st.markdown("---")
    st.caption("Auto-refresh in 30s")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🔍 Search", "📥 Ingest", "🕐 Recent Documents"])


# ===== TAB 1: SEARCH =====
with tab1:
    st.subheader("Search the Knowledge Base")
    query = st.text_input("Enter your fashion query", placeholder="e.g. what shoes go with a black dress")
    col1, col2 = st.columns([1, 3])
    top_k = col1.slider("Top K results", 1, 20, 5)
    store_q = col2.checkbox("Store this query in DB (self-learning)", value=True)

    if st.button("🔍 Search", type="primary") and query.strip():
        with st.spinner("Searching..."):
            try:
                resp = httpx.post(
                    f"{API}/search",
                    json={"query": query, "top_k": top_k, "store_query": store_q},
                    timeout=30,
                )
                data = resp.json()
                results = data.get("results", [])

                if not results:
                    st.info("No results found. Try ingesting some content first.")
                else:
                    st.success(f"Found {len(results)} results")
                    for i, r in enumerate(results):
                        score = r.get("_distance", r.get("score", "—"))
                        stype = r.get("source_type", "unknown")
                        badge_color = {
                            "image": "🖼️", "webpage": "🌐",
                            "scraped": "📰", "user_query": "💬"
                        }.get(stype, "📄")

                        with st.expander(f"{badge_color} [{stype}]  {r.get('content_text', '')[:80]}...  (score: {round(float(score), 4) if score != '—' else '—'})"):
                            st.write("**Content:**", r.get("content_text", ""))
                            st.write("**Source:**", r.get("source_url", "—") or "—")
                            st.write("**Metadata:**", r.get("metadata", {}))
                            st.write("**Created:**", r.get("created_at", "—"))
            except Exception as e:
                st.error(f"Search failed: {e}")


# ===== TAB 2: INGEST =====
with tab2:
    st.subheader("Add Content to the Knowledge Base")

    ingest_tab1, ingest_tab2, ingest_tab3, ingest_tab4 = st.tabs([
        "🖼️ Upload Image", "🔗 Image URL", "🌐 Web URL", "🤖 Run Scraper"
    ])

    # Upload image file
    with ingest_tab1:
        uploaded = st.file_uploader("Upload a fashion image", type=["jpg", "jpeg", "png", "webp"])
        if st.button("Ingest Image", key="ingest_img") and uploaded:
            with st.spinner("Embedding and storing..."):
                try:
                    resp = httpx.post(
                        f"{API}/ingest/image",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        st.success(f"Stored! ID: {resp.json().get('id')}")
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(str(e))

    # Image URL
    with ingest_tab2:
        img_url = st.text_input("Image URL", placeholder="https://example.com/dress.jpg")
        if st.button("Ingest Image URL") and img_url.strip():
            with st.spinner("Embedding and storing..."):
                try:
                    resp = httpx.post(f"{API}/ingest/image-url", json={"url": img_url}, timeout=60)
                    if resp.status_code == 200:
                        st.success(f"Stored! ID: {resp.json().get('id')}")
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(str(e))

    # Web URL
    with ingest_tab3:
        web_url = st.text_input("Web Page URL", placeholder="https://www.vogue.com/article/...")
        if st.button("Ingest Web Page") and web_url.strip():
            with st.spinner("Scraping, chunking, and storing..."):
                try:
                    resp = httpx.post(f"{API}/ingest/url", json={"url": web_url}, timeout=120)
                    if resp.status_code == 200:
                        d = resp.json()
                        st.success(f"Stored {d.get('chunks_stored')} chunks from {web_url}")
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(str(e))

    # Manual scraper trigger
    with ingest_tab4:
        st.write("Manually trigger the RSS scraper to pull the latest fashion news.")
        if st.button("🚀 Run Scraper Now"):
            with st.spinner("Scraping RSS feeds... this may take a minute"):
                try:
                    resp = httpx.post(f"{API}/ingest/scrape-now", timeout=300)
                    summary = resp.json().get("summary", {})
                    st.success("Scrape complete!")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Added", summary.get("added", 0))
                    col_b.metric("Skipped", summary.get("skipped", 0))
                    col_c.metric("Errors", summary.get("errors", 0))
                except Exception as e:
                    st.error(str(e))


# ===== TAB 3: RECENT =====
with tab3:
    st.subheader("Last 20 Inserted Documents")
    if st.button("Load Recent"):
        try:
            resp = httpx.get(f"{API}/db/recent", timeout=10)
            records = resp.json().get("records", [])
            if records:
                df = pd.DataFrame(records)[["id", "source_type", "source_url", "content_text", "created_at"]]
                df["content_text"] = df["content_text"].str[:100] + "..."
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No documents yet.")
        except Exception as e:
            st.error(str(e))
