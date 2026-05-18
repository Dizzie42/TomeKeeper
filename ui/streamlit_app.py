"""Streamlit chat UI with SSE streaming and click-to-open citations.

Run:
    streamlit run ui/streamlit_app.py

Talks to FastAPI at API_URL. The /chat/stream endpoint emits:
  - one "sources" event (JSON array)
  - many "token" events
  - one "done" event
"""
from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Iterator

import requests
import streamlit as st


# ---------- sidebar config ----------

st.set_page_config(page_title="TomeKeeper", page_icon="🎲", layout="wide")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", "http://localhost:8000")
    use_streaming = st.checkbox("Stream responses (recommended)", value=True)
    show_sources = st.checkbox("Show sources", value=True)
    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    with st.expander("Server status"):
        try:
            r = requests.get(f"{api_url}/health", timeout=2)
            r.raise_for_status()
            st.json(r.json())
        except Exception as e:
            st.error(f"API unreachable: {e}")


# ---------- state ----------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- header ----------

st.title("🎲 TomeKeeper")
st.caption("Ask questions across your tabletop library. Local LLM, your data, your machine.")


# ---------- source rendering ----------


def _source_label(s: dict) -> str:
    name = s.get("source", "unknown")
    page = s.get("page")
    kind = s.get("kind", "")
    suffix = f" — p. {page}" if page and page != 1 else ""
    return f"{name}{suffix}" + (f"  ·  *{kind}*" if kind else "")


def _open_link(s: dict) -> str | None:
    """Return a clickable link to the source file, when possible."""
    path = s.get("path")
    if not path:
        return None
    # Notion URLs already work as-is
    if path.startswith("http"):
        return path
    # Local file: build a file:// URL Windows can open
    try:
        p = Path(path)
        if not p.exists():
            return None
        # PDFs with #page=N: many viewers support this; harmless if not
        page = s.get("page")
        anchor = f"#page={page}" if (s.get("kind") == "pdf" and page) else ""
        return p.as_uri() + anchor
    except Exception:
        return None


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 {len(sources)} sources"):
        for i, s in enumerate(sources, 1):
            label = _source_label(s)
            link = _open_link(s)
            if link:
                st.markdown(f"**{i}.** [{label}]({link})")
            else:
                st.markdown(f"**{i}.** {label}")
            st.code(s.get("snippet", ""), language=None)


# ---------- render conversation ----------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            render_sources(msg["sources"])


# ---------- SSE client ----------


def stream_chat(api_url: str, question: str) -> Iterator[tuple[str, str]]:
    """Yield ('event', 'data') tuples from /chat/stream."""
    with requests.post(
        f"{api_url}/chat/stream",
        json={"question": question},
        stream=True,
        timeout=300,
        headers={"Accept": "text/event-stream"},
    ) as r:
        r.raise_for_status()
        event = "message"
        data_buf: list[str] = []
        for raw in r.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            if raw == "":
                # Empty line ends an SSE event
                if data_buf:
                    yield event, "\n".join(data_buf)
                event, data_buf = "message", []
                continue
            if raw.startswith(":"):
                # SSE comment / keep-alive
                continue
            if raw.startswith("event:"):
                event = raw[len("event:") :].strip()
            elif raw.startswith("data:"):
                # Per SSE spec, strip exactly ONE leading space after
                # "data:" — not all whitespace. Using lstrip() here ate
                # legitimate leading spaces in token chunks and made the
                # output run together with no word boundaries.
                data = raw[len("data:") :]
                if data.startswith(" "):
                    data = data[1:]
                data_buf.append(data)


# ---------- input ----------

if q := st.chat_input("Ask about your tabletop library..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        sources_slot = st.empty()
        answer_slot = st.empty()

        try:
            if use_streaming:
                sources: list[dict] = []
                answer = ""
                for event, data in stream_chat(api_url, q):
                    if event == "sources":
                        try:
                            sources = json.loads(data)
                        except json.JSONDecodeError:
                            sources = []
                        if show_sources:
                            with sources_slot.container():
                                render_sources(sources)
                    elif event == "token":
                        answer += data
                        answer_slot.markdown(answer + "▌")
                    elif event == "done":
                        break
                answer_slot.markdown(answer)
            else:
                r = requests.post(
                    f"{api_url}/chat",
                    json={"question": q},
                    timeout=300,
                )
                r.raise_for_status()
                payload = r.json()
                answer = payload["answer"]
                sources = payload.get("sources", [])
                answer_slot.markdown(answer)
                if show_sources:
                    with sources_slot.container():
                        render_sources(sources)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except requests.HTTPError as e:
            st.error(f"API error: {e.response.status_code} {e.response.text}")
        except Exception as e:
            st.error(f"Error: {e}")
