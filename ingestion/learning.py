"""
Real-time DB learning.
Called after every user interaction to keep the DB improving.
"""

from models.embedder import Embedder
from vectordb.lancedb_store import VectorStore, make_record
import hashlib
from datetime import datetime, timezone


def store_user_image(image_pil, session_id: str = "") -> str:
    """
    Embed and store a user-pasted image in the DB.
    Returns the doc_id.
    """
    embedder = Embedder()
    store    = VectorStore()

    vec    = embedder._model.encode(image_pil, convert_to_numpy=True,
                                    normalize_embeddings=True)
    import numpy as np
    vec = vec.astype(np.float32)

    doc_id = hashlib.md5(
        f"user_image::{session_id}::{datetime.now(timezone.utc).isoformat()}".encode()
    ).hexdigest()

    rec = make_record(
        source_type  = "user_image",
        source_url   = "",
        content_text = "[user pasted image]",
        embedding    = vec,
        metadata     = {"session_id": session_id},
        doc_id       = doc_id,
    )
    store.upsert(rec)
    return doc_id
