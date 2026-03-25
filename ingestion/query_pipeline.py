"""
Query pipeline.
User text → embed → search LanceDB → return top-K results.
Also stores every query back into LanceDB (self-improving DB).
"""

import hashlib
from datetime import datetime, timezone

from models.embedder import Embedder
from vectordb.lancedb_store import VectorStore, make_record
from config import DEFAULT_TOP_K


def search(query: str, top_k: int = DEFAULT_TOP_K, store_query: bool = True) -> list[dict]:
    """
    Embed query, retrieve top_k results, optionally store the query in DB.
    Returns ranked list of result dicts with similarity scores.
    """
    embedder = Embedder()
    store = VectorStore()

    query_vec = embedder.embed_text(query)

    # Retrieve
    results = store.search(query_vec, top_k=top_k)

    # Store query for self-improvement (skip duplicates within same session)
    if store_query and query.strip():
        _store_user_query(query, query_vec, store, len(results))

    return results


def _store_user_query(query: str, query_vec, store: VectorStore, result_count: int):
    doc_id = hashlib.md5(f"query::{query.strip().lower()}".encode()).hexdigest()
    rec = make_record(
        source_type="user_query",
        source_url="",
        content_text=query,
        embedding=query_vec,
        metadata={
            "query": query,
            "result_count": result_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        doc_id=doc_id,
    )
    store.upsert(rec)  # upsert deduplicates same query
