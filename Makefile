.PHONY: help up down logs ollama-pull ingest api ui smoke test clean

help:
	@echo "Common tasks:"
	@echo "  make up           Start Qdrant (docker compose up -d)"
	@echo "  make down         Stop Qdrant"
	@echo "  make ollama-pull  Pull the LLM and embedding model into Ollama"
	@echo "  make ingest PATH=\"Z:\\_Tabletop\"   Ingest PDFs from a folder"
	@echo "  make api          Run the FastAPI server (port 8000)"
	@echo "  make ui           Run the Streamlit UI (port 8501)"
	@echo "  make smoke        Quick end-to-end smoke test"
	@echo "  make test         Run pytest"

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f qdrant

ollama-pull:
	ollama pull llama3.1:8b
	ollama pull nomic-embed-text

PATH ?= "Z:\\_Tabletop"
ingest:
	python -m ingest.run --path $(PATH)

api:
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run ui/streamlit_app.py

smoke:
	python scripts/smoke_test.py

test:
	pytest -v

clean:
	docker compose down -v
	rm -rf data/qdrant
