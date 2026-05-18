"""Notion ingestion via the official Notion API.

Setup (one-time):
  1. Visit https://www.notion.so/my-integrations
  2. Click "+ New integration", give it a name like "D&D RAG",
     pick your workspace, "Internal" type.
  3. Copy the "Internal Integration Token" (starts with `secret_` or `ntn_`).
  4. Put it in your .env file:    NOTION_TOKEN=ntn_xxx...
  5. In Notion, open every page (or top-level folder) you want indexed,
     click the "..." menu (top-right) -> "Connections" -> add your
     integration. Subpages inherit.

The integration only sees pages you've explicitly shared with it, which
is the security model. Nothing leaks unless you choose to share it.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

from langchain_core.documents import Document

log = logging.getLogger(__name__)


def get_client():
    """Return a Notion client, lazily importing so the dep is optional."""
    token = os.environ.get("NOTION_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "NOTION_TOKEN is not set. Add it to .env or your environment. "
            "See ingest/notion.py for setup instructions."
        )
    try:
        from notion_client import Client
    except ImportError as e:
        raise SystemExit(
            "notion-client not installed. Run: pip install -r requirements.txt"
        ) from e
    return Client(auth=token)


# ---------- Text extraction ----------


def _rich_text_to_str(rich_text: list[dict]) -> str:
    return "".join(rt.get("plain_text", "") for rt in rich_text or [])


def _block_text(block: dict) -> str:
    """Extract plain text from one Notion block. Returns "" for non-text blocks."""
    btype = block.get("type", "")
    data = block.get(btype, {})

    # Most text-bearing blocks expose `rich_text`
    if "rich_text" in data:
        text = _rich_text_to_str(data["rich_text"])
        # Render headings with markdown-style prefix so chunking respects them
        if btype == "heading_1":
            return f"# {text}"
        if btype == "heading_2":
            return f"## {text}"
        if btype == "heading_3":
            return f"### {text}"
        if btype == "bulleted_list_item":
            return f"- {text}"
        if btype == "numbered_list_item":
            return f"1. {text}"
        if btype == "to_do":
            checked = data.get("checked", False)
            return f"- [{'x' if checked else ' '}] {text}"
        if btype == "quote":
            return f"> {text}"
        if btype == "code":
            lang = data.get("language", "")
            return f"```{lang}\n{text}\n```"
        return text

    # Table cells live deeper; covered by recursion through has_children
    if btype == "child_page":
        return f"## {data.get('title', '')}"

    return ""


def _walk_page(client, page_id: str, depth: int = 0) -> str:
    """Recursively collect text from all blocks of a page."""
    if depth > 6:
        log.debug("Hit recursion cap at page %s", page_id)
        return ""

    parts: list[str] = []
    cursor: str | None = None
    while True:
        kwargs = {"block_id": page_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        result = client.blocks.children.list(**kwargs)

        for block in result["results"]:
            t = _block_text(block)
            if t:
                parts.append(t)
            if block.get("has_children"):
                child = _walk_page(client, block["id"], depth=depth + 1)
                if child:
                    parts.append(child)

        if not result.get("has_more"):
            break
        cursor = result.get("next_cursor")

    return "\n".join(parts)


def _page_title(page: dict) -> str:
    """Pull the title out of a Notion page object."""
    props = page.get("properties", {})
    for v in props.values():
        if v.get("type") == "title":
            t = _rich_text_to_str(v.get("title", []))
            if t:
                return t
    return "Untitled"


# ---------- Public API ----------


def iter_pages() -> Iterable[Document]:
    """Yield one Document per shared Notion page."""
    client = get_client()
    cursor: str | None = None

    while True:
        kwargs = {
            "filter": {"property": "object", "value": "page"},
            "page_size": 100,
        }
        if cursor:
            kwargs["start_cursor"] = cursor

        result = client.search(**kwargs)

        for page in result["results"]:
            title = _page_title(page)
            page_id = page["id"]
            url = page.get("url", "")

            try:
                text = _walk_page(client, page_id)
            except Exception as e:  # noqa: BLE001
                log.warning("Failed to read page %r (%s): %s", title, page_id, e)
                continue

            if not text or not text.strip():
                log.info("Skipping empty page: %s", title)
                continue

            # Prefix with title so it's embedded with the content
            content = f"# {title}\n\n{text}"

            yield Document(
                page_content=content,
                metadata={
                    "source": f"Notion: {title}",
                    "path": url,
                    "page": 1,
                    "kind": "notion",
                    "notion_id": page_id,
                    "notion_url": url,
                },
            )

        if not result.get("has_more"):
            break
        cursor = result.get("next_cursor")
