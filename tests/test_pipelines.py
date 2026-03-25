"""
Integration tests for image, webpage, and query pipelines.
Uses the test DB to avoid polluting the real one.
"""

import os
import sys
import shutil
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch config BEFORE any app imports
import config
config.DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pipeline_test_lancedb")
config.TABLE_NAME = "pipeline_test_kb"

from ingestion.image_pipeline import ingest_image
from ingestion.webpage_pipeline import ingest_url
from ingestion.query_pipeline import search
from vectordb.lancedb_store import VectorStore

# Picsum gives a real JPEG without auth
SAMPLE_IMAGE_URL = "https://picsum.photos/seed/fashion/224/224.jpg"
SAMPLE_PAGE_URL = "https://en.wikipedia.org/wiki/Fashion"


@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    shutil.rmtree(config.DB_PATH, ignore_errors=True)


def test_image_pipeline_url():
    rec = ingest_image(SAMPLE_IMAGE_URL)
    assert rec["source_type"] == "image"
    assert rec["id"] is not None
    assert len(rec["embedding"]) == 512


def test_image_pipeline_no_duplicate():
    store = VectorStore()
    before = store.count()
    ingest_image(SAMPLE_IMAGE_URL)
    ingest_image(SAMPLE_IMAGE_URL)
    assert store.count() == before  # upsert, not duplicated


def test_webpage_pipeline():
    records = ingest_url(SAMPLE_PAGE_URL)
    assert len(records) > 0
    for r in records:
        assert r["source_type"] == "webpage"
        assert r["source_url"] == SAMPLE_PAGE_URL
        assert len(r["content_text"]) > 0


def test_query_pipeline_returns_results():
    results = search("fashion trends", top_k=3, store_query=False)
    assert isinstance(results, list)
    # Results may be empty if DB is empty, but no crash
    for r in results:
        assert "content_text" in r
        assert "source_type" in r


def test_query_stored_in_db():
    import hashlib
    store = VectorStore()
    query = "what shoes go with a black dress"
    search(query, top_k=3, store_query=True)
    # Verify by doc_id — more reliable than count after delete+add cycle
    doc_id = hashlib.md5(f"query::{query.strip().lower()}".encode()).hexdigest()
    record = store.get_by_id(doc_id)
    assert record is not None, "Query was not stored in DB"
    assert record["source_type"] == "user_query"


def test_query_deduplication():
    store = VectorStore()
    search("blue handbag outfit", top_k=3, store_query=True)
    count_after_first = store.count()
    search("blue handbag outfit", top_k=3, store_query=True)
    count_after_second = store.count()
    assert count_after_first == count_after_second  # same query not duplicated
