"""Typed application configuration loaded from environment / .env file.

We use pydantic-settings so every config value has a type, a default,
and is centrally documented. Anywhere in the codebase we do:

    from app.config import settings
    settings.llm_model  # always typed and validated
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Ollama ----
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"

    # ---- Qdrant ----
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "tabletop"

    # ---- App ----
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # ---- Retrieval / chunking ----
    top_k: int = 10
    chunk_size: int = 800
    chunk_overlap: int = 150


settings = Settings()
