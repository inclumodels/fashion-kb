"""
Conversation session memory.
Keeps last N turns in memory per session_id.
Also persists full conversation to LanceDB.
"""

from collections import defaultdict
from config import MAX_HISTORY_TURNS

# In-memory store: session_id → list of turns
_sessions: dict[str, list[dict]] = defaultdict(list)


def get_history(session_id: str) -> list[dict]:
    """Return last MAX_HISTORY_TURNS turns for this session."""
    return _sessions[session_id][-MAX_HISTORY_TURNS:]


def add_turn(session_id: str, query: str, answer: str, chunks: list[dict]):
    """Add a completed turn to session memory + persist to DB."""
    turn = {
        "query":    query,
        "answer":   answer,
        "sources":  [c.get("source_url", "") for c in chunks if c.get("source_url")],
    }
    _sessions[session_id].append(turn)

    # Persist to vector DB for long-term learning
    _persist(session_id, query, answer, chunks)


def _persist(session_id: str, query: str, answer: str, chunks: list[dict]):
    """Store the Q&A pair in LanceDB so future queries can use it as context."""
    try:
        from models.embedder import Embedder
        from vectordb.lancedb_store import VectorStore, make_record
        import hashlib

        embedder = Embedder()
        store    = VectorStore()

        # Embed the answer (not the query) — answers are the valuable context
        combined = f"Q: {query}\nA: {answer}"
        vec = embedder.embed_text(combined)

        doc_id = hashlib.md5(f"conv::{session_id}::{query}".encode()).hexdigest()
        rec = make_record(
            source_type  = "conversation",
            source_url   = "",
            content_text = combined,
            embedding    = vec,
            metadata     = {
                "session_id": session_id,
                "query":      query,
                "sources":    [c.get("source_url","") for c in chunks][:3],
            },
            doc_id = doc_id,
        )
        store.upsert(rec)
    except Exception as e:
        print(f"[conversation] persist failed: {e}")


def clear_session(session_id: str):
    _sessions.pop(session_id, None)
