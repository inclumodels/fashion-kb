"""
Image ingestion pipeline.
Accepts local file path or image URL → embeds → stores in LanceDB.
"""

import os
import hashlib
from models.embedder import Embedder
from vectordb.lancedb_store import VectorStore, make_record


def ingest_image(image_source: str, metadata: dict = None) -> dict:
    """
    Ingest a single image.
    Returns the stored record dict.
    """
    embedder = Embedder()
    store = VectorStore()

    # Stable ID based on source so same image isn't duplicated
    doc_id = hashlib.md5(image_source.encode()).hexdigest()

    embedding = embedder.embed_image(image_source)

    record = make_record(
        source_type="image",
        source_url=image_source,
        content_text=f"[image] {os.path.basename(image_source)}",
        embedding=embedding,
        metadata=metadata or {},
        doc_id=doc_id,
    )
    store.upsert(record)
    return record


def ingest_images_batch(sources: list[str], metadata: dict = None) -> list[dict]:
    records = []
    for src in sources:
        try:
            rec = ingest_image(src, metadata)
            records.append(rec)
        except Exception as e:
            print(f"[image_pipeline] skipped {src}: {e}")
    return records
