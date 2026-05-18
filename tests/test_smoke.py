"""Import-time smoke tests.

These do not require Ollama / Qdrant / Hugging Face models to be running.
They catch typos, circular imports, and missing deps before the heavy
stack is spun up.

Run:
    pytest -v
"""
from __future__ import annotations

import os


def test_app_imports():
    # Don't actually load the reranker model in tests
    os.environ["DISABLE_RERANKER"] = "1"

    from app import api, chain, config, embeddings, llm, retrieval, vectorstore  # noqa: F401
    from app import bm25_store  # noqa: F401

    assert config.settings.llm_model
    assert config.settings.embedding_model


def test_ingest_imports():
    from ingest import chunking, loaders, run  # noqa: F401
    from ingest import notion, notion_run  # noqa: F401


def test_chunking_basic():
    """No external services needed — just exercise the splitter."""
    from langchain_core.documents import Document
    from ingest.chunking import chunk

    long_text = "Lorem ipsum dolor sit amet. " * 200
    docs = [Document(page_content=long_text, metadata={"source": "test", "page": 1})]
    chunks = chunk(docs)

    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["source"] == "test"
        assert c.metadata["page"] == 1


def test_source_header_enrichment():
    """The filename-in-chunk trick that makes 'find me the X guide' queries work."""
    from langchain_core.documents import Document
    from ingest.run import enrich_with_source_header

    chunks = [
        Document(page_content="A chunk body.", metadata={"source": "MyGuide.pdf", "page": 12}),
    ]
    out = enrich_with_source_header(chunks)
    assert out[0].page_content.startswith("[Source: MyGuide.pdf p.12]")
    assert "A chunk body." in out[0].page_content


def test_loader_dispatcher():
    """The LOADERS dispatch should cover at least PDF + TXT + MD."""
    from ingest.loaders import LOADERS

    assert ".pdf" in LOADERS
    assert ".txt" in LOADERS
    assert ".md" in LOADERS
