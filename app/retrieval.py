"""Production-style retrieval pipeline.

    question
       │
       ▼
   MultiQueryRetriever        (LLM rewrites the query into 3 variants)
       │
       ▼
   EnsembleRetriever          (runs BOTH legs for each variant, dedups)
       ├── BM25Retriever      (keyword / proper-noun friendly)
       └── Qdrant dense       (semantic / paraphrase friendly)
       │
       ▼
   ContextualCompressionRetriever
       └── CrossEncoderReranker(BAAI/bge-reranker-base)
       │       scores every candidate against the ORIGINAL question
       │       returns top settings.top_k
       ▼
   chain/api

Why each piece:

- MultiQueryRetriever: "find me the Sean McGovern short outline" is
  vague — the LLM can rewrite it into more retrievable forms like
  "Sean McGovern Curse of Strahd outline" and "short outline summary".
- Ensemble (BM25 + dense): dense embeddings miss proper nouns and exact
  phrases; BM25 misses paraphrases. Together they cover both.
- CrossEncoderReranker: ensemble candidates are noisy. The cross-encoder
  reads each (question, chunk) pair fully (not just compares vectors)
  and assigns a precision score. Standard production pattern.

Graceful degradation: if BM25 sidecar is missing we fall back to dense
only. If sentence-transformers is unavailable, the reranker can be
disabled via DISABLE_RERANKER=1.
"""
from __future__ import annotations

import logging
import os

# langchain 1.x moved these classes to langchain_classic. We try the
# 0.3 layout first for back-compat, then fall back to the 1.x path.
try:
    from langchain.retrievers import (
        ContextualCompressionRetriever,
        EnsembleRetriever,
    )
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain.retrievers.multi_query import MultiQueryRetriever
except ImportError:  # langchain >= 1.0
    from langchain_classic.retrievers import (
        ContextualCompressionRetriever,
        EnsembleRetriever,
    )
    from langchain_classic.retrievers.document_compressors import (
        CrossEncoderReranker,
    )
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.retrievers import BaseRetriever

from app.bm25_store import build_bm25_retriever
from app.config import settings
from app.llm import get_llm
from app.vectorstore import get_vector_store

log = logging.getLogger(__name__)

# Each leg returns this many candidates BEFORE reranking. The reranker
# then trims to settings.top_k. Larger = better recall, slower.
CANDIDATES_PER_LEG = 20

RERANKER_MODEL = "BAAI/bge-reranker-base"


def get_dense_retriever(k: int = CANDIDATES_PER_LEG) -> BaseRetriever:
    return get_vector_store().as_retriever(search_kwargs={"k": k})


def get_hybrid_retriever(k: int = CANDIDATES_PER_LEG) -> BaseRetriever:
    """BM25 + dense ensemble. Falls back to dense-only if BM25 unavailable."""
    dense = get_dense_retriever(k=k)
    bm25 = build_bm25_retriever(k=k)
    if bm25 is None:
        log.warning("BM25 unavailable — using dense-only retrieval.")
        return dense
    # 50/50 weighting is a sensible default; tune with eval later
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])


def get_multiquery_retriever(base: BaseRetriever) -> BaseRetriever:
    """Wrap base retriever with LLM-driven query rewriting."""
    return MultiQueryRetriever.from_llm(
        retriever=base,
        # Temp 0.0 so the rewrites are deterministic per query
        llm=get_llm(temperature=0.0),
        include_original=True,  # always retrieve for the user's literal query too
    )


def get_reranker() -> CrossEncoderReranker | None:
    """Load the cross-encoder reranker. Downloads ~280 MB on first call."""
    if os.environ.get("DISABLE_RERANKER") == "1":
        log.info("Reranker disabled via DISABLE_RERANKER=1")
        return None
    log.info("Loading cross-encoder: %s", RERANKER_MODEL)
    model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    return CrossEncoderReranker(model=model, top_n=settings.top_k)


def get_retriever() -> BaseRetriever:
    """Assemble the full pipeline."""
    hybrid = get_hybrid_retriever()
    multi = get_multiquery_retriever(hybrid)
    reranker = get_reranker()
    if reranker is None:
        return multi
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=multi,
    )
