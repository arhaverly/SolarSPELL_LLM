"""Ask SolarSPELL: retrieval-augmented chat over the SolarSPELL library.

Serves the web UI and a streaming /api/query endpoint on one port, so any
device on the SolarSPELL hotspot can open http://<pi-address>:8000.
"""

import json
import logging
import time
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config, llm
from app.retrieval import Retriever

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.time()
    log.info("Loading FAISS index and embedding model...")
    app.state.retriever = Retriever(config.INDEX_PATH, config.CHUNKS_DB_PATH, config.EMBEDDING_MODEL_PATH)
    log.info("Loaded %d chunks in %.1fs", app.state.retriever.index.ntotal, time.time() - start)
    yield


app = FastAPI(title="Ask SolarSPELL", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=config.TOP_K, ge=1, le=10)


@app.get("/api/health")
def health():
    return {"retriever": True, "llm": llm.is_ready(), "prompt_format": config.PROMPT_FORMAT}


@app.post("/api/query")
def query(request: QueryRequest):
    """Streams newline-delimited JSON: one {"sources": [...]} line, then
    {"token": "..."} lines, or an {"error": "..."} line if generation fails."""
    chunks = app.state.retriever.search(request.query, request.top_k)

    def events():
        sources = [{"source": c.source, "page": c.page} for c in chunks]
        yield json.dumps({"sources": sources}) + "\n"
        try:
            for token in llm.stream_answer(request.query, chunks):
                yield json.dumps({"token": token}) + "\n"
        except requests.RequestException as e:
            log.error("LLM request failed: %s", e)
            yield json.dumps({"error": "The language model is not responding. Is llama-server running?"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="ui")
