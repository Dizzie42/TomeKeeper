"""BM25 sidecar storage.

Qdrant is the source of truth for chunks. BM25 needs all chunk text in
memory to build its index, so we keep a derived JSONL cache at
`data/chunks.jsonl`. The cache is rebuilt from Qdrant at the end of every
ingestion run.

Why a sidecar instead of Qdrant native sparse vectors?
  - Simpler: just a file you can `cat` and inspect.
  - Fewer dependencies: no FastEmbed / SPLADE models.
  - Easy to debug: BM25 is deterministic and inspectable.

For larger-scale production you'd want Qdrant native sparse vectors so
everything lives in one DB. For personal-scale (<100k chunks), sidecar
is faster to load and easier to reason about.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.config import settings
from app.vectorstore import get_qdrant_client

log = logging.getLogger(__name__)

SIDECAR_PATH = Path("data/chunks.jsonl")
SCROLL_BATCH = 500


# --- Build the sidecar from Qdrant ---


def rebuild_sidecar() -> int:
    """Scroll the entire Qdrant collection and write each chunk to JSONL.

    Returns the number of chunks written. Safe to run after every ingest.
    """
    client = get_qdrant_client()
    collection = settings.qdrant_collection

    if not client.collection_exists(collection):
        log.warning("Collection %s does not exist; nothing to rebuild.", collection)
        return 0

    SIDECAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SIDECAR_PATH.with_suffix(".jsonl.tmp")
    count = 0
    offset = None

    with tmp.open("w", encoding="utf-8") as f:
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                limit=SCROLL_BATCH,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for p in points:
                payload = p.payload or {}
                # LangChain QdrantVectorStore stores content under
                # "page_content" and metadata under "metadata" by default.
                content = payload.get("page_content", "")
                metadata = payload.get("metadata", {})
                f.write(
                    json.dumps({"page_content": content, "metadata": metadata})
                    + "\n"
                )
                count += 1
            if offset is None:
                break

    # Atomic replace so a partial rebuild can't poison the live sidecar
    tmp.replace(SIDECAR_PATH)
    log.info("BM25 sidecar rebuilt: %d chunks at %s", count, SIDECAR_PATH)
    return count


# --- Load the sidecar into a BM25Retriever ---


def load_documents() -> list[Document]:
    if not SIDECAR_PATH.exists():
        return []
    docs: list[Document] = []
    with SIDECAR_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(
                Document(
                    page_content=obj.get("page_content", ""),
                    metadata=obj.get("metadata", {}),
                )
            )
    return docs


def build_bm25_retriever(k: int) -> BM25Retriever | None:
    """Build an in-memory BM25 retriever from the sidecar.

    Returns None if the sidecar is empty or missing — callers should
    fall back to dense-only retrieval in that case.
    """
    docs = load_documents()
    if not docs:
        log.warning(
            "BM25 sidecar at %s is empty. Run `python -m ingest.run --path ...` "
            "(or call rebuild_sidecar()) to populate it.",
            SIDECAR_PATH,
        )
        return None
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever
