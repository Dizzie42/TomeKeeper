"""Chunking strategy.

For Phase 1 we use a RecursiveCharacterTextSplitter — the default LangChain
splitter. It tries paragraph -> sentence -> word boundaries, in that order,
which preserves semantic units better than a naive fixed-window split.

Phase 2 ideas:
  - semantic chunking (embedding-based boundary detection),
  - tabletop-aware splitting (keep stat blocks together).
"""
from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # ordered most- to least-preferred boundary
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk(docs: Iterable[Document]) -> list[Document]:
    splitter = get_splitter()
    return splitter.split_documents(list(docs))
