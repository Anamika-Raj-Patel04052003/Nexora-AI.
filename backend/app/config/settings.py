
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# --------------------------------
# BACKEND DIRECTORY LOCATION
# --------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):

    # --------------------------------
    # GROQ CONFIGURATION
    # --------------------------------

    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # --------------------------------
    # LANGSMITH CONFIGURATION
    # --------------------------------

    langsmith_api_key: str | None = None
    langsmith_tracing: bool = False
    langsmith_project: str = "NEXORA-AI"

    # --------------------------------
    # SQLITE DATABASE CONFIGURATION
    # --------------------------------

    # Used for storing chat history
    database_url: str = "sqlite:///./nexora.db"

    # --------------------------------
    # POSTGRESQL + PGVECTOR CONFIGURATION
    # --------------------------------

    # Used for storing document embeddings
    vector_database_url: str | None = None

    # --------------------------------
    # EMBEDDING CONFIGURATION
    # --------------------------------

    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # --------------------------------
    # APPLICATION CONFIGURATION
    # --------------------------------

    app_name: str = "NEXORA AI"
    debug: bool = True

    # --------------------------------
    # ENVIRONMENT CONFIGURATION
    # --------------------------------

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# --------------------------------
# SETTINGS INSTANCE
# --------------------------------

settings = Settings()
