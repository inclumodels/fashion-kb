import os
import sys
import numpy as np
import pytest
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch config BEFORE any app imports
import config
config.DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vectordb_test_lancedb")
config.TABLE_NAME = "vectordb_test_kb"

from vectordb.lancedb_store import VectorStore, make_record


@pytest.fixture(scope="module")
def store():
    s = VectorStore()
    yield s
    # cleanup
    shutil.rmtree(config.DB_PATH, ignore_errors=True)


def _random_vec():
    v = np.random.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_insert_and_count(store):
    before = store.count()
    rec = make_record("webpage", "http://test.com", "fashion content", _random_vec())
    store.insert(rec)
    assert store.count() == before + 1


def test_upsert_no_duplicate(store):
    vec = _random_vec()
    rec = make_record("image", "http://img.com/a.jpg", "[image] a.jpg", vec, doc_id="fixed-id-1")
    store.upsert(rec)
    store.upsert(rec)
    rows = store.search(vec, top_k=10)
    ids = [r["id"] for r in rows]
    assert ids.count("fixed-id-1") == 1


def test_search_returns_correct_top(store):
    # Insert a distinct vector and query for it
    target_vec = np.ones(512, dtype=np.float32)
    target_vec /= np.linalg.norm(target_vec)
    rec = make_record("scraped", "http://scraped.com", "target content", target_vec, doc_id="target-1")
    store.upsert(rec)

    results = store.search(target_vec, top_k=1)
    assert results[0]["id"] == "target-1"


def test_url_exists(store):
    vec = _random_vec()
    rec = make_record("webpage", "http://unique-url.com/page", "text", vec)
    store.insert(rec)
    assert store.url_exists("http://unique-url.com/page")
    assert not store.url_exists("http://does-not-exist.com")


def test_count_by_type(store):
    counts = store.count_by_type()
    assert isinstance(counts, dict)
    assert all(isinstance(v, int) for v in counts.values())


def test_batch_insert(store):
    before = store.count()
    records = [
        make_record("scraped", f"http://batch{i}.com", f"chunk {i}", _random_vec())
        for i in range(5)
    ]
    store.batch_insert(records)
    assert store.count() == before + 5


def test_recent(store):
    items = store.recent(5)
    assert isinstance(items, list)
    assert len(items) <= 5
