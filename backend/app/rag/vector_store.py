
from functools import lru_cache

from langchain_postgres import PGVector

from app.config.settings import settings
from app.rag.embeddings import get_embedding_model
from sqlalchemy import delete

COLLECTION_NAME = "nexora_documents"


def get_database_connection():

    database_url = settings.vector_database_url

    if not database_url:
        raise ValueError(
            "PostgreSQL is not configured. "
            "Set VECTOR_DATABASE_URL in your .env file."
        )

    if not database_url.startswith(
        "postgresql+psycopg://"
    ):
        raise ValueError(
            "Invalid PostgreSQL URL. "
            "Use postgresql+psycopg:// format."
        )

    return database_url


@lru_cache(maxsize=1)
def get_vector_store():

    connection = get_database_connection()

    embeddings = get_embedding_model()

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=connection,
        use_jsonb=True,
        create_extension=True
    )

    return vector_store


def add_documents_to_vector_store(documents):

    if not documents:
        raise ValueError(
            "No document chunks provided."
        )

    vector_store = get_vector_store()

    ids = vector_store.add_documents(
        documents=documents
    )

    return ids


def delete_document_from_vector_store(source: str):
    vector_store = get_vector_store()

    with vector_store._make_sync_session() as session:

        # Get current collection
        collection = vector_store.get_collection(session)

        if not collection:
            return 0

        # Delete only records matching the source
        stmt = (
            delete(vector_store.EmbeddingStore)
            .where(
                vector_store.EmbeddingStore.collection_id == collection.uuid
            )
            .where(
                vector_store.EmbeddingStore.cmetadata["source"].astext == source
            )
        )

        result = session.execute(stmt)
        session.commit()

        return result.rowcount or 0