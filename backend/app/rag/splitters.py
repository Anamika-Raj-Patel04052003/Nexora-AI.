
"""
NEXORA AI - Document Text Splitter

Splits loaded documents into smaller overlapping chunks
for embedding generation and vector database storage.
"""

import logging

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


logger = logging.getLogger(__name__)


# --------------------------------------------------
# CREATE TEXT SPLITTER
# --------------------------------------------------

def create_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):

    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError(
            "chunk_size must be a positive integer."
        )

    if (
        not isinstance(chunk_overlap, int)
        or isinstance(chunk_overlap, bool)
        or chunk_overlap < 0
        or chunk_overlap >= chunk_size
    ):
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,

        separators=[
            "\n\n",
            "\n",
            "। ",
            ". ",
            " ",
            ""
        ],

        length_function=len,
        add_start_index=True
    )


# --------------------------------------------------
# SPLIT DOCUMENTS
# --------------------------------------------------

def split_documents(
    documents,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):

    if not documents:
        raise ValueError(
            "No documents provided for splitting."
        )

    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    try:

        chunks = splitter.split_documents(documents)

        # Remove empty chunks
        chunks = [
            chunk
            for chunk in chunks
            if chunk.page_content
            and chunk.page_content.strip()
        ]

        if not chunks:
            raise ValueError(
                "No meaningful text chunks could be generated."
            )

        logger.info(
            "Split %d document sections into %d chunks.",
            len(documents),
            len(chunks)
        )

        return chunks

    except ValueError:
        raise

    except Exception as error:

        logger.exception(
            "Document splitting failed."
        )

        raise RuntimeError(
            "Unable to split document content."
        ) from error
