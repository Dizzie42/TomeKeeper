"""LLM client.

Thin factory around ChatOllama. We isolate this so the rest of the codebase
never imports langchain_ollama directly — that makes it trivial to swap
Ollama for vLLM / TGI / OpenAI later without touching the chain code.
"""
from __future__ import annotations

from langchain_ollama import ChatOllama

from app.config import settings


def get_llm(temperature: float = 0.2) -> ChatOllama:
    """Return a configured chat LLM.

    Streaming is enabled implicitly when the caller uses `.astream(...)`
    on the chain — no separate flag needed.
    """
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
        temperature=temperature,
    )
