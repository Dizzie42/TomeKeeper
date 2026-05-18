"""CLI: ingest a directory of documents into Qdrant.

Supported formats: .pdf, .txt, .md (see ingest.loaders.LOADERS).

Usage:
    python -m ingest.run --path "Z:\\_Tabletop"
    python -m ingest.run --path "Z:\\_Tabletop" --limit 3   # smoke test

Idempotent at the chunk level — re-running on the same files will upsert
(LangChain generates deterministic IDs from content). Adding new files
and re-running picks up the new ones; existing chunks are no-ops.

We prepend `[Source: <filename>]` to each chunk before embedding so that
dense-vector retrieval can match queries that mention the filename
("the Sean McGovern guide"). Without this, the only signal the embedder
sees is the chunk body, which often doesn't contain the filename.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from langchain_core.documents import Document

from app.bm25_store import rebuild_sidecar
from app.vectorstore import get_vector_store
from ingest.chunking import chunk
from ingest.loaders import LOADERS, iter_supported_files, load_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    p.add_argument(
        "--path",
        required=True,
        help='Root folder to scan, e.g. "Z:\\_Tabletop"',
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: stop after N files (handy for smoke tests).",
    )
    return p.parse_args(argv)


ANCHOR_CHARS = 400


def extract_document_anchor(docs: list[Document], max_chars: int = ANCHOR_CHARS) -> str:
    """Pull the first ~400 chars of front matter (title page / opening).

    Typically contains the document title, author, version — exactly the
    metadata users mention in queries like "the Sean McGovern guide" or
    "the 5e Player's Handbook". By prepending this anchor to every chunk,
    BM25 and dense retrieval can both surface ANY chunk of the document
    when the user names the author/title.

    This is a lightweight version of Anthropic's "Contextual Retrieval"
    (Sept 2024): we use front-matter as the per-doc context instead of
    LLM-generated summaries — same idea, ~free at ingest time.
    """
    if not docs:
        return ""
    text = docs[0].page_content[:max_chars * 2]  # take some slack, then clean
    # Collapse whitespace so the anchor stays compact in chunk content
    cleaned = " ".join(text.split())
    return cleaned[:max_chars].strip()


def enrich_with_source_header(
    chunks: list[Document],
    anchor: str = "",
) -> list[Document]:
    """Prepend [Source: ...] (and optional [Doc context: ...]) to each chunk.

    The source line lets the filename be matchable by retrieval.
    The doc-context line lets author/title text propagate to every chunk.
    """
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        page = c.metadata.get("page")
        header = f"[Source: {src}" + (f" p.{page}" if page else "") + "]"
        if anchor:
            header = f"{header}\n[Doc context: {anchor}]"
        if not c.page_content.startswith("[Source:"):
            c.page_content = f"{header}\n\n{c.page_content}"
    return chunks


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    root = Path(args.path)
    if not root.exists():
        log.error("Path does not exist: %s", root)
        return 2

    log.info("Scanning %s for %s...", root, ", ".join(LOADERS.keys()))
    vs = get_vector_store()

    total_files = 0
    total_chunks = 0
    by_kind: dict[str, int] = {}
    t0 = time.perf_counter()

    for path in iter_supported_files(root):
        if args.limit is not None and total_files >= args.limit:
            break

        rel = path.relative_to(root)
        log.info("Loading %s", rel)
        try:
            docs = list(load_file(path))
        except Exception as e:  # noqa: BLE001
            log.warning("  failed to load (%s); skipping", e)
            continue

        if not docs:
            log.warning("  no extractable text; skipping")
            continue

        chunks = chunk(docs)
        if not chunks:
            log.warning("  produced 0 chunks; skipping")
            continue

        # Per-document anchor: front matter (title page / opening) text
        # gets prepended to every chunk so author/title queries hit any
        # chunk in the document.
        anchor = extract_document_anchor(docs)
        chunks = enrich_with_source_header(chunks, anchor=anchor)

        try:
            vs.add_documents(chunks)
        except Exception as e:  # noqa: BLE001
            log.exception("  embed/upsert failed: %s", e)
            continue

        kind = path.suffix.lstrip(".").lower()
        by_kind[kind] = by_kind.get(kind, 0) + 1
        total_files += 1
        total_chunks += len(chunks)
        log.info("  %d docs -> %d chunks", len(docs), len(chunks))

    elapsed = time.perf_counter() - t0
    kind_summary = ", ".join(f"{n} {k}" for k, n in sorted(by_kind.items()))
    log.info(
        "Done. %d files (%s), %d chunks indexed in %.1fs.",
        total_files,
        kind_summary or "none",
        total_chunks,
        elapsed,
    )

    # Rebuild the BM25 sidecar so hybrid retrieval has fresh keyword index.
    log.info("Rebuilding BM25 sidecar from Qdrant...")
    n = rebuild_sidecar()
    log.info("BM25 sidecar: %d chunks.", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
