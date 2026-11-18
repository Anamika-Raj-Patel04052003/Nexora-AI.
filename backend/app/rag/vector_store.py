
"""
NEXORA AI - PostgreSQL Vector Store

Manages document embeddings and vector storage
using LangChain PGVector.
"""

import logging

from functools import lru_cache

from sqlalchemy import delete

from langchain_postgres import PGVector

from app.config.settings import settings
from app.rag.embeddings import get_embedding_model


logger = logging.getLogger(__name__)

COLLECTION_NAME = "nexora_documents"


# ==========================================
# 1. DATABASE CONNECTION
# ==========================================

def get_database_connection() -> str:
    """Validate and return PostgreSQL connection URL."""

    database_url = settings.vector_database_url

    if not database_url:
        raise ValueError(
            "PostgreSQL is not configured. "
            "Set VECTOR_DATABASE_URL in your .env file."
        )

    if not database_url.startswith("postgresql+psycopg://"):
        raise ValueError(
            "Invalid PostgreSQL URL. "
            "Use postgresql+psycopg:// format."
        )

    return database_url


# ==========================================
# 2. INITIALIZE VECTOR STORE
# ==========================================

@lru_cache(maxsize=1)
def get_vector_store() -> PGVector:
    """Initialize and cache the PGVector store."""

    connection = get_database_connection()
    embeddings = get_embedding_model()

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=connection,
        use_jsonb=True,
        create_extension=True
    )

    logger.info(
        "PGVector initialized. Collection: %s",
        COLLECTION_NAME
    )

    return vector_store


# ==========================================
# 3. ADD DOCUMENTS
# ==========================================

def add_documents_to_vector_store(documents: list) -> list:
    """Generate embeddings and store document chunks."""

    if not documents:
        raise ValueError("No document chunks provided.")

    vector_store = get_vector_store()

    ids = vector_store.add_documents(
        documents=documents
    )

    if not ids:
        raise RuntimeError(
            "Document vectors could not be stored."
        )

    logger.info(
        "Stored %s document vectors.",
        len(ids)
    )

    return ids


# ==========================================
# 4. DELETE DOCUMENT
# ==========================================

def delete_document_from_vector_store(
    source: str,
    conversation_id: int | None = None
) -> int:
    """
    Delete vectors matching a document source.

    Optionally restrict deletion to one conversation.
    """

    if not isinstance(source, str) or not source.strip():
        raise ValueError("Valid document source is required.")

    if conversation_id is not None:
        if (
            not isinstance(conversation_id, int)
            or isinstance(conversation_id, bool)
            or conversation_id < 1
        ):
            raise ValueError("Invalid conversation ID.")

    vector_store = get_vector_store()

    try:

        with vector_store._make_sync_session() as session:

            collection = vector_store.get_collection(session)

            if not collection:
                return 0

            statement = (
                delete(vector_store.EmbeddingStore)
                .where(
                    vector_store.EmbeddingStore.collection_id
                    == collection.uuid
                )
                .where(
                    vector_store.EmbeddingStore.cmetadata["source"].astext
                    == source.strip()
                )
            )

            if conversation_id is not None:

                statement = statement.where(
                    vector_store.EmbeddingStore.cmetadata[
                        "conversation_id"
                    ].astext == str(conversation_id)
                )

            result = session.execute(statement)

            deleted_count = result.rowcount or 0

            session.commit()

            logger.info(
                "Deleted %s vectors for source: %s",
                deleted_count,
                source
            )

            return deleted_count

    except Exception:

        logger.exception(
            "Failed to delete document vectors."
        )

        raise
