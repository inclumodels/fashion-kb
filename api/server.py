"""
FastAPI backend — exposes all pipelines via HTTP endpoints.
"""

import os
import json
import uuid
import shutil
import tempfile
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from ingestion.query_pipeline import search
from ingestion.image_pipeline import ingest_image
from ingestion.webpage_pipeline import ingest_url
from vectordb.lancedb_store import VectorStore
from scheduler import realtime_sync
from llm.gemini import ask, ask_stream
from llm.image_handler import from_base64, from_url, to_pil
from llm.conversation import get_history, add_turn
from ingestion.learning import store_user_image

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    realtime_sync.start()
    yield
    realtime_sync.stop()


app = FastAPI(title="Fashion KB API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request models ----------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    store_query: bool = True


class URLRequest(BaseModel):
    url: str
    metadata: dict = {}


class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    image_base64: Optional[str] = None   # data URL from clipboard paste
    image_url: Optional[str] = None      # image from URL
    top_k: int = 5


# ---------- Endpoints ----------

@app.post("/search")
def search_endpoint(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(400, "query cannot be empty")
    results = search(req.query, top_k=req.top_k, store_query=req.store_query)
    return {"query": req.query, "results": results, "count": len(results)}


@app.post("/ingest/image")
async def ingest_image_endpoint(file: UploadFile = File(...)):
    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        record = ingest_image(tmp.name, metadata={"original_filename": file.filename})
    finally:
        os.unlink(tmp.name)

    return {"status": "ok", "id": record["id"], "filename": file.filename}


@app.post("/ingest/image-url")
def ingest_image_url_endpoint(req: URLRequest):
    record = ingest_image(req.url, metadata=req.metadata)
    return {"status": "ok", "id": record["id"], "url": req.url}


@app.post("/ingest/url")
def ingest_url_endpoint(req: URLRequest):
    records = ingest_url(req.url, metadata=req.metadata)
    if not records:
        raise HTTPException(422, "No text could be extracted from that URL")
    return {"status": "ok", "chunks_stored": len(records), "url": req.url}


@app.post("/ingest/scrape-now")
def scrape_now_endpoint():
    summary = realtime_sync.trigger_now()
    return {"status": "ok", "summary": summary}


@app.get("/db/stats")
def db_stats():
    store = VectorStore()
    return {
        "total": store.count(),
        "by_type": store.count_by_type(),
    }


@app.get("/db/recent")
def db_recent():
    store = VectorStore()
    return {"records": store.recent(20)}


@app.post("/ask")
async def ask_endpoint(req: AskRequest):
    """
    Main conversational endpoint.
    Retrieves context from DB, sends to Gemini, streams the answer back.
    Also stores image + conversation in DB for self-learning.
    """
    if not req.query.strip():
        raise HTTPException(400, "query cannot be empty")

    session_id = req.session_id or str(uuid.uuid4())

    # Handle image input
    image_data = None
    if req.image_base64:
        try:
            image_data = from_base64(req.image_base64)
            pil_img    = to_pil(image_data)
            store_user_image(pil_img, session_id=session_id)
        except Exception as e:
            logger.warning(f"image processing failed: {e}")

    elif req.image_url:
        try:
            image_data = from_url(req.image_url)
            pil_img    = to_pil(image_data)
            store_user_image(pil_img, session_id=session_id)
        except Exception as e:
            logger.warning(f"image url failed: {e}")

    # Retrieve relevant chunks from DB
    chunks = search(req.query, top_k=req.top_k, store_query=True)

    # Get conversation history for this session
    history = get_history(session_id)

    # Stream Gemini response
    def generate():
        full_answer = []
        # Stream answer tokens
        for token in ask_stream(req.query, chunks, history, image_data):
            full_answer.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"

        # After streaming, send sources + session_id
        answer_text = "".join(full_answer)
        sources = list({c["source_url"] for c in chunks if c.get("source_url")})
        yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'sources': sources[:5]})}\n\n"

        # Store conversation in DB (self-learning)
        add_turn(session_id, req.query, answer_text, chunks)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/health")
def health():
    return {"status": "ok"}
