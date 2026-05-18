"""Embedding model client.

We use Ollama's `nomic-embed-text` (768-dim) by default. Keeping embeddings
and chat under the same Ollama server means one less moving part.
"""
from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from app.config import settings


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
    )
