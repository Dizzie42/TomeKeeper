"""FastAPI service.

Endpoints:
  GET  /health           - liveness + config echo
  POST /chat             - non-streaming: returns {"answer": str, "sources": [...]}
  POST /chat/stream      - SSE: emits one "sources" event, then "token"
                           chunks, then "done"
  POST /search           - raw retrieval (no LLM) for debugging

Run:
    uvicorn app.api:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.chain import build_chain_with_sources, format_docs, prompt
from app.config import settings
from app.llm import get_llm
from app.retrieval import get_retriever

log = logging.getLogger(__name__)


# ---------- request / response models ----------


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)


class SourceChunk(BaseModel):
    source: str
    page: int | str | None = None
    path: str | None = None
    kind: str | None = None
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    k: int | None = None


# ---------- helpers ----------


def _doc_to_source_chunk(d) -> dict:
    """Render a retrieved Document as a JSON-serializable source chunk."""
    return {
        "source": d.metadata.get("source", "unknown"),
        "page": d.metadata.get("page"),
        "path": d.metadata.get("path"),
        "kind": d.metadata.get("kind"),
        "snippet": d.page_content[:400],
    }


# ---------- app lifecycle ----------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up everything at startup so the first user request is fast.

    This loads:
      - Qdrant client + ensures collection exists
      - BM25 index from sidecar
      - Cross-encoder reranker model (~280 MB download on first launch)
      - Ollama LLM client (lazy until first call)
    """
    log.info("Building retrieval pipeline (this may take a moment on first run)...")
    app.state.retriever = get_retriever()
    app.state.chain_full = build_chain_with_sources()
    app.state.llm = get_llm()
    log.info("Ready.")
    yield


app = FastAPI(title="D&D RAG", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- routes ----------


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "qdrant_collection": settings.qdrant_collection,
        "top_k": settings.top_k,
        "version": app.version,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Non-streaming chat. Returns the full answer + sources in one shot."""
    result = await app.state.chain_full.ainvoke({"question": req.question})
    sources = [SourceChunk(**_doc_to_source_chunk(d)) for d in result["sources"]]
    return ChatResponse(answer=result["answer"], sources=sources)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Server-Sent Events stream.

    Event sequence:
      1. event: sources   data: JSON array of source chunks
      2. event: token     data: <single token / chunk of text>     (many)
      3. event: done      data: ""

    Why this shape: the UI can render the "Sources" panel as soon as the
    sources event arrives, then paint tokens into the answer area as they
    arrive. This is how ChatGPT-style UIs feel snappy.
    """
    retriever = app.state.retriever
    llm = app.state.llm

    # Retrieve once, use the docs for both the source preview and the LLM context.
    docs = await retriever.ainvoke(req.question)
    sources_payload = [_doc_to_source_chunk(d) for d in docs]
    context = format_docs(docs)

    # Build the prompt message list ahead of time so the streaming loop
    # is just "LLM consumes messages, yield tokens".
    messages = prompt.invoke({"context": context, "question": req.question}).to_messages()
    parser = StrOutputParser()

    async def event_gen() -> AsyncGenerator[dict, None]:
        yield {"event": "sources", "data": json.dumps(sources_payload)}
        async for chunk in llm.astream(messages):
            # AIMessageChunk -> str
            text = parser.invoke(chunk) if hasattr(chunk, "content") else str(chunk)
            if text:
                yield {"event": "token", "data": text}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_gen())


@app.post("/search")
async def search(req: SearchRequest):
    """Raw retrieval — bypasses the LLM. Useful for tuning the pipeline."""
    if req.k:
        from app.vectorstore import get_vector_store

        retriever = get_vector_store().as_retriever(search_kwargs={"k": req.k})
    else:
        retriever = app.state.retriever
    docs = await retriever.ainvoke(req.query)
    return [
        {
            **_doc_to_source_chunk(d),
            "content": d.page_content,
        }
        for d in docs
    ]
