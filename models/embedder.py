"""
Unified CLIP embedder — text and images share the same 512-dim vector space.
This means a text query can find images and vice versa.
"""

import numpy as np
from PIL import Image
import requests
from io import BytesIO
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, EMBEDDING_DIM


class Embedder:
    _instance = None

    def __new__(cls):
        # Singleton — load model once
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = SentenceTransformer(EMBEDDING_MODEL)
        return cls._instance

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text string → 512-dim float32 vector."""
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_image(self, image_source: str) -> np.ndarray:
        """
        Embed an image → 512-dim float32 vector.
        image_source: local file path OR http/https URL
        """
        img = self._load_image(image_source)
        vec = self._model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_texts_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Batch embed a list of texts — faster than one-by-one."""
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
        return [v.astype(np.float32) for v in vecs]

    def _load_image(self, source: str) -> Image.Image:
        if source.startswith("http://") or source.startswith("https://"):
            resp = requests.get(source, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            img = Image.open(source).convert("RGB")
        return img

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM
