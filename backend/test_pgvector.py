
from langchain_core.documents import Document

from app.rag.vector_store import get_vector_store


def test_pgvector():

    print("\nInitializing PGVector...")

    vector_store = get_vector_store()

    # Unique source for test document
    test_source = "nexora_pgvector_test"

    # Sample document
    document = Document(
        page_content=(
            "NEXORA AI is a multilingual AI chatbot "
            "that uses Retrieval Augmented Generation "
            "to answer questions from documents."
        ),
        metadata={
            "source": test_source,
            "file_type": "txt"
        }
    )

    try:

        # Insert document and generate embeddings
        print("\nInserting document...")

        ids = vector_store.add_documents(
            documents=[document]
        )

        print("Document inserted successfully!")
        print("Document IDs:", ids)

        # Similarity search
        print("\nPerforming similarity search...")

        results = vector_store.similarity_search(
            query="What is NEXORA AI?",
            k=1
        )

        print("\nSearch Results:")

        for result in results:
            print("Content:", result.page_content)
            print("Metadata:", result.metadata)

        if results:
            print("\nSimilarity Search Successful!")
        else:
            print("\nNo matching documents found.")

    finally:

        # Remove only our test document
        vector_store.delete(
            filter={"source": test_source}
        )

        print("\nTest document cleanup completed.")


if __name__ == "__main__":
    test_pgvector()
