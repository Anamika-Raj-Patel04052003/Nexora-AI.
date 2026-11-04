
from functools import lru_cache

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from app.config.settings import settings


@lru_cache(maxsize=1)
def get_embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32
        }
    )

    return embeddings


def generate_embeddings(texts: list[str]):

    if not texts:
        return []

    embeddings = get_embedding_model()

    vectors = embeddings.embed_documents(
        texts
    )

    return vectors


def generate_query_embedding(text: str):

    if not text.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    embeddings = get_embedding_model()

    return embeddings.embed_query(text)
