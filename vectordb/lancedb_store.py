"""
LanceDB vector store — all DB operations live here.
Schema supports text, image, scraped web content, and user queries.
"""

import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional

import lancedb
import pyarrow as pa
import numpy as np

import config


# ---------- Schema ----------
def _schema() -> pa.Schema:
    return pa.schema([
        pa.field("id",           pa.string()),
        pa.field("source_type",  pa.string()),   # image | webpage | scraped | user_query
        pa.field("source_url",   pa.string()),
        pa.field("content_text", pa.string()),
        pa.field("embedding",    pa.list_(pa.float32(), config.EMBEDDING_DIM)),
        pa.field("metadata",     pa.string()),   # JSON string
        pa.field("created_at",   pa.string()),
    ])


# ---------- Store ----------
class VectorStore:
    def __init__(self):
        os.makedirs(config.DB_PATH, exist_ok=True)
        self._db = lancedb.connect(config.DB_PATH)
        self._table = self._get_or_create_table()

    def _get_or_create_table(self):
        dummy = _make_record(
            source_type="init",
            source_url="",
            content_text="",
            embedding=np.zeros(config.EMBEDDING_DIM, dtype=np.float32),
        )
        tbl = self._db.create_table(
            config.TABLE_NAME, data=[dummy], schema=_schema(), exist_ok=True
        )
        # Remove the seed row only if we just created the table (count == 1 with init type)
        try:
            tbl.delete(f"source_type = 'init'")
        except Exception:
            pass
        return tbl

    # ---------- Write ----------
    def insert(self, record: dict):
        self._table.add([record])

    def batch_insert(self, records: list[dict]):
        if records:
            self._table.add(records)

    def upsert(self, record: dict):
        """Insert or replace by id."""
        try:
            self._table.delete(f"id = '{record['id']}'")
        except Exception:
            pass
        self._table.add([record])

    # ---------- Read ----------
    def search(self, query_vector: np.ndarray, top_k: int = None,
               source_type: Optional[str] = None) -> list[dict]:
        top_k = top_k or config.DEFAULT_TOP_K
        q = self._table.search(query_vector.tolist()).limit(top_k)
        if source_type:
            q = q.where(f"source_type = '{source_type}'")
        results = q.to_list()
        # Add score, drop raw embedding from response
        out = []
        for r in results:
            r.pop("embedding", None)
            r["metadata"] = json.loads(r.get("metadata", "{}"))
            out.append(r)
        return out

    def get_by_id(self, doc_id: str) -> Optional[dict]:
        df = self._table.to_pandas()
        rows = df[df["id"] == doc_id]
        if rows.empty:
            return None
        row = rows.iloc[0].to_dict()
        row.pop("embedding", None)
        return row

    def url_exists(self, url: str) -> bool:
        df = self._table.to_pandas()
        return not df[df["source_url"] == url].empty

    # ---------- Stats ----------
    def count(self) -> int:
        return self._table.count_rows()

    def count_by_type(self) -> dict:
        df = self._table.to_pandas()[["source_type"]]
        return df["source_type"].value_counts().to_dict()

    def recent(self, n: int = 20) -> list[dict]:
        df = self._table.to_pandas().sort_values("created_at", ascending=False).head(n)
        df = df.drop(columns=["embedding"], errors="ignore")
        return df.to_dict(orient="records")


# ---------- Helper ----------
def make_record(source_type: str, source_url: str, content_text: str,
                embedding: np.ndarray, metadata: dict = None, doc_id: str = None) -> dict:
    return {
        "id":           doc_id or str(uuid.uuid4()),
        "source_type":  source_type,
        "source_url":   source_url,
        "content_text": content_text,
        "embedding":    embedding.tolist(),
        "metadata":     json.dumps(metadata or {}),
        "created_at":   datetime.now(timezone.utc).isoformat(),
    }

# internal alias used during init
_make_record = make_record
