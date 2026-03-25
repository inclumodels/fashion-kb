"""
Real-time fashion web scraper.
Polls RSS feeds → finds new articles → scrapes full text → embeds → stores in LanceDB.
Skips URLs already in the DB (no duplicates).
"""

import hashlib
import logging
from datetime import datetime

import feedparser
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.embedder import Embedder
from vectordb.lancedb_store import VectorStore, make_record
from config import RSS_FEEDS, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FashionBot/1.0)"}


def _scrape_article(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        return " ".join(soup.get_text(separator=" ").split())
    except Exception as e:
        logger.warning(f"scrape failed for {url}: {e}")
        return ""


def run_scraper() -> dict:
    """
    Run one full scrape cycle across all RSS feeds.
    Returns summary: { added: int, skipped: int, errors: int }
    """
    store = VectorStore()
    embedder = Embedder()

    added = skipped = errors = 0

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            entries = feed.entries[:20]  # cap per feed per run
        except Exception as e:
            logger.error(f"feed parse failed {feed_url}: {e}")
            errors += 1
            continue

        for entry in entries:
            url = getattr(entry, "link", None)
            if not url:
                continue

            # Skip if already in DB
            if store.url_exists(url):
                skipped += 1
                continue

            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            full_text = _scrape_article(url) or f"{title}. {summary}"

            if not full_text.strip():
                skipped += 1
                continue

            chunks = _splitter.split_text(full_text)
            if not chunks:
                skipped += 1
                continue

            try:
                embeddings = embedder.embed_texts_batch(chunks)
                records = []
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    doc_id = hashlib.md5(f"{url}::chunk::{i}".encode()).hexdigest()
                    records.append(make_record(
                        source_type="scraped",
                        source_url=url,
                        content_text=chunk,
                        embedding=emb,
                        metadata={
                            "title": title,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "scraped_at": datetime.utcnow().isoformat(),
                            "feed": feed_url,
                        },
                        doc_id=doc_id,
                    ))
                store.batch_insert(records)
                added += len(records)
            except Exception as e:
                logger.error(f"embed/store failed for {url}: {e}")
                errors += 1

    summary = {"added": added, "skipped": skipped, "errors": errors,
                "timestamp": datetime.utcnow().isoformat()}
    logger.info(f"Scraper run complete: {summary}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run_scraper())
