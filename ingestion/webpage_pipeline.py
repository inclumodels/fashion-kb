"""
Webpage ingestion pipeline.
URL → scrape text → chunk → embed each chunk → store in LanceDB.
"""

import hashlib
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.embedder import Embedder
from vectordb.lancedb_store import VectorStore, make_record
from config import CHUNK_SIZE, CHUNK_OVERLAP


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FashionBot/1.0)"}


def _scrape_text(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def ingest_url(url: str, metadata: dict = None) -> list[dict]:
    """
    Scrape a URL, chunk the text, embed and store each chunk.
    Returns list of stored records.
    """
    embedder = Embedder()
    store = VectorStore()

    text = _scrape_text(url)
    if not text.strip():
        return []

    chunks = _splitter.split_text(text)
    records = []

    texts = chunks  # batch embed
    embeddings = embedder.embed_texts_batch(texts)

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        doc_id = hashlib.md5(f"{url}::chunk::{i}".encode()).hexdigest()
        meta = {**(metadata or {}), "chunk_index": i, "total_chunks": len(chunks)}
        rec = make_record(
            source_type="webpage",
            source_url=url,
            content_text=chunk,
            embedding=emb,
            metadata=meta,
            doc_id=doc_id,
        )
        records.append(rec)

    store.batch_insert(records)
    return records
