"""The RAG chain.

Pure LangChain LCEL (LangChain Expression Language). The chain is:

    {"question": str}
        ├──► retriever ─► format_docs ─► "context"
        └──────────────────────────────► "question"
                            │
                            ▼
                         prompt
                            │
                            ▼
                          LLM
                            │
                            ▼
                      StrOutputParser
                            │
                            ▼
                          str

Two builders are exposed:
  - build_chain():            returns answer string only
  - build_chain_with_sources(): returns {"answer": str, "sources": [Doc, ...]}
"""
from __future__ import annotations

from operator import itemgetter
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from app.llm import get_llm
from app.retrieval import get_retriever


SYSTEM_PROMPT = """You are a helpful tabletop RPG assistant.

Answer the user's question using ONLY the CONTEXT below. If the answer is
not in the context, say so explicitly — do not invent rules.

GUIDELINES:
- If the user names a specific document (e.g. "the Sean McGovern guide"),
  prefer chunks whose [Source: ...] tag matches that name.
- If the user asks for a specific section (e.g. "the short outline",
  "the table of contents", "the summary"), scan the CONTEXT for matching
  headings and reproduce that section's content faithfully.
- Cite every fact like [filename p.N]. Use the exact filename from the
  [Source: ...] tag.
- Be concise but complete. Don't omit information the user explicitly
  asked for.

CONTEXT:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def format_docs(docs: list[Document]) -> str:
    """Render retrieved docs into a single context string with citations."""
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        parts.append(f"[{src} p.{page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain() -> Runnable:
    """Chain returning just the answer string."""
    retriever = get_retriever()
    llm = get_llm()

    return (
        RunnableParallel(
            context=itemgetter("question") | retriever | RunnableLambda(format_docs),
            question=itemgetter("question"),
        )
        | prompt
        | llm
        | StrOutputParser()
    )


def build_chain_with_sources() -> Runnable:
    """Chain returning {'answer': str, 'sources': list[Document]}.

    Useful for the UI: shows users which chunks informed the answer.
    """
    retriever = get_retriever()
    llm = get_llm()

    answer_only: Runnable = prompt | llm | StrOutputParser()

    def _run_answer(payload: dict[str, Any]) -> str:
        return answer_only.invoke(
            {
                "context": format_docs(payload["sources"]),
                "question": payload["question"],
            }
        )

    return RunnableParallel(
        sources=itemgetter("question") | retriever,
        question=itemgetter("question"),
    ) | RunnablePassthrough.assign(answer=RunnableLambda(_run_answer))
