"""CLI: ingest Notion pages into Qdrant.

Usage:
    python -m ingest.notion_run
    python -m ingest.notion_run --limit 5     # smoke test

Requires NOTION_TOKEN in environment / .env.
See ingest/notion.py for one-time integration setup.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

# Load .env so NOTION_TOKEN is available
from dotenv import load_dotenv

load_dotenv()

from app.bm25_store import rebuild_sidecar  # noqa: E402
from app.vectorstore import get_vector_store  # noqa: E402
from ingest.chunking import chunk  # noqa: E402
from ingest.notion import iter_pages  # noqa: E402
from ingest.run import enrich_with_source_header, extract_document_anchor  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest.notion")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest Notion pages into Qdrant")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: stop after N pages (handy for smoke tests).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    vs = get_vector_store()
    t0 = time.perf_counter()
    total_pages = 0
    total_chunks = 0

    for doc in iter_pages():
        if args.limit is not None and total_pages >= args.limit:
            break

        title = doc.metadata.get("source", "Notion: ?")
        log.info("Ingesting %s", title)

        chunks = chunk([doc])
        if not chunks:
            log.warning("  produced 0 chunks; skipping")
            continue
        anchor = extract_document_anchor([doc])
        chunks = enrich_with_source_header(chunks, anchor=anchor)

        try:
            vs.add_documents(chunks)
        except Exception as e:  # noqa: BLE001
            log.exception("  embed/upsert failed: %s", e)
            continue

        total_pages += 1
        total_chunks += len(chunks)
        log.info("  -> %d chunks", len(chunks))

    elapsed = time.perf_counter() - t0
    log.info(
        "Done. %d Notion pages, %d chunks in %.1fs.",
        total_pages,
        total_chunks,
        elapsed,
    )

    log.info("Rebuilding BM25 sidecar from Qdrant...")
    n = rebuild_sidecar()
    log.info("BM25 sidecar: %d chunks.", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
