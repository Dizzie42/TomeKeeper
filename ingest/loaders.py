"""Document loaders.

PyMuPDF (`fitz`) is used for PDFs because it's:
  - fast (much faster than pypdf on large rulebooks),
  - preserves page numbers (we use them in citations),
  - handles complex layouts (multi-column stat blocks) reasonably well.

Each PDF yields one Document per page. Plain-text formats (.txt, .md)
yield a single Document for the whole file. Chunking happens downstream.

To support a new format:
  1. Write a loader returning Iterable[Document] with at least
     `source`, `path`, `page`, `kind` in metadata.
  2. Register it in the LOADERS dict at the bottom of this file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import fitz  # PyMuPDF
from langchain_core.documents import Document


# ---------- PDF ----------


def load_pdf(path: Path) -> Iterable[Document]:
    """Yield one Document per page of the PDF, with rich metadata."""
    doc = fitz.open(path)
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if not text or not text.strip():
                continue
            yield Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "path": str(path),
                    "page": i + 1,
                    "kind": "pdf",
                },
            )
    finally:
        doc.close()


# ---------- Plain text (.txt, .md) ----------


def load_text(path: Path) -> Iterable[Document]:
    """Load a plain-text file as a single Document.

    .md files are treated as text — we don't parse markdown structure here
    because the RecursiveCharacterTextSplitter handles paragraph-level
    splits well enough for Phase 1. Phase 2 may swap in
    MarkdownHeaderTextSplitter for heading-aware chunking.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Some Windows-authored text files are CP1252-encoded.
        text = path.read_text(encoding="cp1252", errors="replace")

    if not text.strip():
        return

    yield Document(
        page_content=text,
        metadata={
            "source": path.name,
            "path": str(path),
            "page": 1,  # keeps citation format consistent across formats
            "kind": path.suffix.lstrip(".").lower(),
        },
    )


# ---------- Dispatcher ----------


# Map extension -> loader function. Add new formats here.
LOADERS: dict[str, Callable[[Path], Iterable[Document]]] = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_text,
}


def iter_supported_files(root: Path) -> Iterable[Path]:
    """Recursively yield files we know how to ingest, sorted for determinism."""
    files: list[Path] = []
    for ext in LOADERS:
        files.extend(root.rglob(f"*{ext}"))
    yield from sorted(files)


def load_file(path: Path) -> Iterable[Document]:
    """Dispatch to the right loader based on file extension."""
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        return iter(())
    return loader(path)
