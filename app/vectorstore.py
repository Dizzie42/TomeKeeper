"""Qdrant client + LangChain vector store wrapper.

Bootstrapping the collection lazily (in `ensure_collection`) means:
  - first run after `docker compose up` will create the collection;
  - subsequent runs are no-ops;
  - destroying the volume (`make clean`) gives you a clean slate.

The embedding dimension is hard-coded to 768 because that's what
nomic-embed-text emits. If you switch embedding models, update EMBED_DIM
or compute it dynamically by embedding a dummy string at startup.
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

from app.config import settings
from app.embeddings import get_embeddings


# nomic-embed-text → 768-d dense vectors
EMBED_DIM = 768


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(client: QdrantClient, name: str | None = None) -> None:
    """Create the collection if it doesn't exist. Idempotent."""
    name = name or settings.qdrant_collection
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def get_vector_store() -> QdrantVectorStore:
    """Return a LangChain QdrantVectorStore bound to our collection.

    Used by both:
      - ingest pipeline (calls `.add_documents(...)`)
      - retrieval at query time (calls `.as_retriever(...)`)
    """
    client = get_qdrant_client()
    ensure_collection(client)
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=get_embeddings(),
    )
