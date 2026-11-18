"""
NEXORA AI - Embedding Model Service

Loads the Hugging Face embedding model and generates
document/query embeddings.
"""

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config.settings import settings


logger = logging.getLogger(__name__)


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model():

    model_name = settings.embedding_model

    if not model_name or not model_name.strip():
        raise ValueError(
            "Embedding model name is not configured."
        )

    try:
        logger.info(
            "Loading embedding model: %s",
            model_name
        )

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,

            model_kwargs={
                "device": "cpu"
            },

            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32
            }
        )

        logger.info(
            "Embedding model loaded successfully."
        )

        return embeddings

    except Exception as error:

        logger.exception(
            "Failed to load embedding model."
        )

        raise RuntimeError(
            "Unable to load the embedding model."
        ) from error


# --------------------------------------------------
# DOCUMENT EMBEDDINGS
# --------------------------------------------------

def generate_embeddings(texts: list[str]) -> list[list[float]]:

    if not texts:
        return []

    if not isinstance(texts, list):
        raise ValueError(
            "Texts must be provided as a list."
        )

    if any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError(
            "All document texts must be non-empty strings."
        )

    try:

        embeddings = get_embedding_model()

        vectors = embeddings.embed_documents(texts)

        logger.info(
            "Generated embeddings for %d documents.",
            len(texts)
        )

        return vectors

    except Exception as error:

        logger.exception(
            "Failed to generate document embeddings."
        )

        raise RuntimeError(
            "Unable to generate document embeddings."
        ) from error


# --------------------------------------------------
# QUERY EMBEDDING
# --------------------------------------------------

def generate_query_embedding(text: str) -> list[float]:

    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    try:

        embeddings = get_embedding_model()

        vector = embeddings.embed_query(text)

        logger.info(
            "Query embedding generated successfully."
        )

        return vector

    except Exception as error:

        logger.exception(
            "Failed to generate query embedding."
        )

        raise RuntimeError(
            "Unable to generate query embedding."
        ) from error