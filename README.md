# D&D RAG — Local LLM over Tabletop PDFs & Notion

A production-style Retrieval-Augmented Generation (RAG) pipeline that runs
**entirely on your own machine**. Ask questions across your D&D PDFs and
Notion notes through a streaming chat UI.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit (ui/streamlit_app.py) — http://localhost:8501│
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────────┐
│  FastAPI (app/api.py)            — http://localhost:8000│
│    /health     /chat     /chat/stream   /search         │
└──────────┬───────────────────────────────────┬──────────┘
           │                                   │
┌──────────▼─────────┐                ┌────────▼─────────┐
│  LangChain chain   │                │  Qdrant          │
│  • Retriever       │◄──── vectors ──┤  (Docker :6333)  │
│  • Prompt          │                └──────────────────┘
│  • ChatOllama      │
└────────┬───────────┘                ┌──────────────────┐
         │                            │  Ollama          │
         └──────── chat / embed ─────►│  (host :11434)   │
                                      │   llama3.1:8b    │
                                      │   nomic-embed    │
                                      └──────────────────┘

Ingestion (one-shot CLI):
  PDFs ──► PyMuPDF page text ──► RecursiveCharacterTextSplitter
                                          │
                                          ▼
                                  Ollama embed (768-d)
                                          │
                                          ▼
                                    Qdrant collection
```

## Stack — and why each piece

| Layer            | Choice                | Reason                                              |
|------------------|-----------------------|-----------------------------------------------------|
| LLM runtime      | **Ollama**            | Industry-standard local server; OpenAI-compatible API |
| Chat model       | **llama3.1:8b**       | Strong general reasoning at ~5 GB VRAM (Q4_K_M)     |
| Embedding model  | **nomic-embed-text**  | 768-d, runs inside Ollama, no extra service         |
| Vector DB        | **Qdrant** (Docker)   | Real service, REST + gRPC, used in real jobs        |
| Orchestration    | **LangChain**         | Most prevalent orchestrator          |
| PDF loader       | **PyMuPDF (fitz)**    | Fast, reliable text + page metadata                 |
| API              | **FastAPI** + SSE     | Production-grade async server with streaming        |
| Frontend         | **Streamlit**         | One-file Python chat UI                             |
| Config           | **pydantic-settings** | Typed `.env`-driven config                          |

## Prerequisites

- **Windows 10/11** (paths use Windows conventions in this README)
- **Python 3.11+**
- **Docker Desktop**
- **Ollama** — download from https://ollama.com/download
- An **NVIDIA GPU with 8+ GB VRAM**

## First-time setup

```powershell
# 1. From repo root, create a venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install Python deps
pip install -r requirements.txt

# 3. Copy the env template
copy .env.example .env

# 4. Pull the Ollama models (once, ~5.5 GB total)
.\scripts\pull_models.ps1

# 5. Start Qdrant
docker compose up -d
```

## Running it

You need three things running. Use three terminals (each is independently
startable and stoppable, so you can release GPU/RAM whenever you want).

```powershell
# Terminal A — Qdrant (already running from `docker compose up -d`)
# Verify:  docker compose ps

# Terminal B — FastAPI
.\.venv\Scripts\Activate.ps1
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

# Terminal C — Streamlit UI
.\.venv\Scripts\Activate.ps1
streamlit run ui/streamlit_app.py
```

Open http://localhost:8501 — that's your chat interface.

## Ingesting your D&D library

```powershell
.\.venv\Scripts\Activate.ps1
python -m ingest.run --path "Z:\_Tabletop"
```

This walks the folder, extracts text from every PDF, chunks it, embeds it
via Ollama, and writes vectors into Qdrant. Re-run any time you add PDFs
(re-runs are idempotent at the chunk level — duplicate chunks get the same
ID and are upserted).

Optional: limit for a quick smoke test
```powershell
python -m ingest.run --path "Z:\_Tabletop" --limit 1
```

## Turning everything off

```powershell
# Stop Qdrant (releases its container)
docker compose down

# Ctrl+C in the FastAPI and Streamlit terminals

# Ollama keeps a small daemon running; to fully stop it:
#   Right-click the Ollama tray icon → Quit
# This releases all GPU/RAM held by loaded models.
```

## Project layout

```
AI-LLM/
├── app/                     # FastAPI service + LangChain pipeline
│   ├── api.py               # HTTP endpoints: /health, /chat, /chat/stream
│   ├── chain.py             # RAG chain: retrieve → prompt → LLM
│   ├── config.py            # pydantic-settings (loads .env)
│   ├── embeddings.py        # Ollama embedding wrapper
│   ├── llm.py               # Ollama chat wrapper
│   ├── retrieval.py         # Retriever builder
│   └── vectorstore.py       # Qdrant client + collection bootstrap
├── ingest/                  # One-shot ingestion CLI
│   ├── chunking.py          # RecursiveCharacterTextSplitter
│   ├── loaders.py           # PyMuPDF page-level loader
│   └── run.py               # `python -m ingest.run --path ...`
├── ui/
│   └── streamlit_app.py     # Chat frontend
├── scripts/
│   ├── pull_models.ps1      # Pulls llama3.1:8b + nomic-embed-text
│   └── smoke_test.py        # End-to-end sanity check
├── tests/
│   └── test_smoke.py        # Import smoke + structural tests
├── docker-compose.yml       # Qdrant service
├── requirements.txt
├── .env.example
└── README.md
```

## Roadmap

### Phase 1 — MVP (this scaffold)
- [x] Ollama + LangChain + Qdrant end-to-end
- [x] PDF ingestion with page-level metadata
- [x] FastAPI `/chat` and `/chat/stream`
- [x] Streamlit chat UI

### Phase 2 — Real RAG quality
- [ ] Hybrid retrieval (BM25 + dense) via Qdrant fastembed sparse
- [ ] Cross-encoder reranker (BAAI/bge-reranker-base)
- [ ] Notion ingestion via official API
- [ ] Source citations rendered in UI with page links
- [ ] Token-streaming UI (consume SSE in Streamlit)
- [ ] Query rewriting / HyDE for vague questions

### Phase 3 — Production patterns
- [ ] Langfuse observability (self-hosted in docker-compose)
- [ ] RAGAS eval suite + golden Q&A set
- [ ] Prompt versioning
- [ ] pytest integration tests against the API
- [ ] GitHub Actions CI
- [ ] Optional: replace Streamlit with Next.js frontend

## Troubleshooting

**`httpx.ConnectError` to Ollama** — Ollama isn't running. Open it from
the Start menu (it lives in the tray).

**`qdrant_client...ConnectionError`** — Qdrant container isn't up. Run
`docker compose ps`; if empty, `docker compose up -d`.

**Slow first query** — Ollama lazy-loads models into VRAM on first use.
Subsequent queries are fast.

**Out-of-memory loading the LLM** — Drop to a smaller model:
edit `.env` → `LLM_MODEL=llama3.2:3b`, then `ollama pull llama3.2:3b`.
