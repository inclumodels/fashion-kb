import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.embedder import Embedder

@pytest.fixture(scope="module")
def embedder():
    return Embedder()

def test_text_embedding_shape(embedder):
    vec = embedder.embed_text("red floral dress")
    assert vec.shape == (512,), f"Expected (512,), got {vec.shape}"
    assert vec.dtype == np.float32

def test_text_embedding_normalized(embedder):
    vec = embedder.embed_text("summer fashion trends")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

def test_batch_text_embedding(embedder):
    texts = ["blue jeans", "red dress", "white sneakers"]
    vecs = embedder.embed_texts_batch(texts)
    assert len(vecs) == 3
    for v in vecs:
        assert v.shape == (512,)

def test_semantic_similarity(embedder):
    """'red dress' should be closer to 'crimson gown' than to 'leather boots'"""
    v_query = embedder.embed_text("red dress")
    v_close = embedder.embed_text("crimson gown")
    v_far   = embedder.embed_text("leather boots")
    sim_close = float(np.dot(v_query, v_close))
    sim_far   = float(np.dot(v_query, v_far))
    assert sim_close > sim_far, f"Expected sim_close({sim_close:.3f}) > sim_far({sim_far:.3f})"

def test_singleton(embedder):
    e2 = Embedder()
    assert embedder is e2
