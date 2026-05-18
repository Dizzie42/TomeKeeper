"""End-to-end smoke test.

Verifies the whole stack is wired up:
  1. Ollama is reachable and has the models loaded.
  2. Qdrant is reachable and the collection exists.
  3. We can embed a sentence and write a single Document.
  4. We can run a one-shot retrieval + LLM call.

Run after `docker compose up -d` and `ollama serve` (Ollama on Windows
runs automatically once installed):

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly without `python -m`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from langchain_core.documents import Document

from app.chain import build_chain_with_sources
from app.config import settings
from app.vectorstore import get_vector_store


def check(label: str, fn):
    print(f"... {label}", end=" ", flush=True)
    try:
        fn()
        print("OK")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"FAIL\n    {type(e).__name__}: {e}")
        return False


def main() -> int:
    ok = True

    def ping_ollama():
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        r.raise_for_status()
        names = {m["name"] for m in r.json().get("models", [])}
        if not names:
            raise RuntimeError(
                "Ollama has no models installed — run scripts/pull_models.ps1"
            )
        # Each tag looks like "llama3.1:8b"; match by family or full tag.
        needed = {settings.llm_model, settings.embedding_model}
        installed_families = {n.split(":")[0] for n in names}
        missing = {
            m for m in needed
            if m not in names and m.split(":")[0] not in installed_families
        }
        if missing:
            raise RuntimeError(
                f"Missing Ollama models: {missing}. Run scripts/pull_models.ps1"
            )

    ok &= check("Ollama reachable", ping_ollama)

    def ping_qdrant():
        r = httpx.get(f"{settings.qdrant_url}/", timeout=5)
        r.raise_for_status()

    ok &= check("Qdrant reachable", ping_qdrant)

    def write_one():
        vs = get_vector_store()
        vs.add_documents(
            [
                Document(
                    page_content="The wizard cast a fireball at the goblin.",
                    metadata={"source": "_smoke_test.txt", "page": 1},
                )
            ]
        )

    ok &= check("Embed + upsert one doc", write_one)

    def query_one():
        chain = build_chain_with_sources()
        result = chain.invoke({"question": "What did the wizard cast?"})
        assert "fireball" in result["answer"].lower(), f"Unexpected: {result['answer']}"

    ok &= check("Retrieve + LLM round trip", query_one)

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
