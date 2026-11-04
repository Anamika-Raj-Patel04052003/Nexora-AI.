
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


def create_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
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

    return splitter


def split_documents(
    documents,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):

    splitter = create_text_splitter(
        chunk_size,
        chunk_overlap
    )

    chunks = splitter.split_documents(
        documents
    )

    # Remove empty chunks
    chunks = [
        chunk
        for chunk in chunks
        if chunk.page_content.strip()
    ]

    return chunks
