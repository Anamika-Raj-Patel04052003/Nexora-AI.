
"""
NEXORA AI - Document Loader Service

Supports PDF, TXT, Markdown, DOCX and CSV files.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)


logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".docx",
    ".csv",
}


# --------------------------------------------------
# LOAD SINGLE DOCUMENT
# --------------------------------------------------

def load_document(file_path: str):

    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("A valid file path is required.")

    path = Path(file_path).resolve()

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    try:

        logger.info(
            "Loading document: %s",
            path.name
        )

        # Select loader based on file extension
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

        else:  # CSV

            loader = CSVLoader(
                str(path),
                encoding="utf-8"
            )

        documents = loader.load()

        if not documents:
            raise ValueError(
                "No readable content found in document."
            )

        # Attach consistent metadata to every document
        for document in documents:

            document.metadata["source"] = str(path)

            document.metadata["filename"] = path.name

            document.metadata["file_type"] = extension

        logger.info(
            "Successfully loaded %d document sections from %s.",
            len(documents),
            path.name
        )

        return documents

    except (ValueError, FileNotFoundError):
        raise

    except Exception as error:

        logger.exception(
            "Failed to load document: %s",
            path.name
        )

        raise RuntimeError(
            f"Unable to read document: {path.name}"
        ) from error


# --------------------------------------------------
# LOAD MULTIPLE DOCUMENTS
# --------------------------------------------------

def load_multiple_documents(
    file_paths: list[str]
):

    if not isinstance(file_paths, list) or not file_paths:
        raise ValueError(
            "At least one document path is required."
        )

    all_documents = []

    for file_path in file_paths:

        documents = load_document(file_path)

        all_documents.extend(documents)

    logger.info(
        "Loaded a total of %d document sections from %d files.",
        len(all_documents),
        len(file_paths)
    )

    return all_documents
