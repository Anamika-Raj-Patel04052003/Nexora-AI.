
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".docx",
    ".csv",
}


def load_document(file_path: str):

    path = Path(file_path).resolve()

    # Check file existence
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    # Select loader based on extension
    if extension == ".pdf":
        loader = PyPDFLoader(str(path))

    elif extension in {".txt", ".md"}:
        loader = TextLoader(
            str(path),
            encoding="utf-8",
            autodetect_encoding=True
        )

    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))

    elif extension == ".csv":
        loader = CSVLoader(
            str(path),
            encoding="utf-8"
        )

    documents = loader.load()

    if not documents:
        raise ValueError(
            "No readable content found in document."
        )

    # Add file information to metadata
    for document in documents:
        document.metadata["source"] = path.name
        document.metadata["file_type"] = extension

    return documents


def load_multiple_documents(file_paths: list[str]):

    all_documents = []

    for file_path in file_paths:

        documents = load_document(file_path)

        all_documents.extend(documents)

    return all_documents
