"""
NEXORA AI - RAG Agent

Dedicated document retrieval & question-answering agent:
- Semantic search via PGVector with conversation filtering
- Document relevance evaluation (CRAG principles)
- Web search fallback integration
- Factual grounding & citation extraction
"""

import logging
from app.rag.pipeline import ingest_document
from app.rag.retriever import retrieve_documents
from app.rag.rag_chain import extract_sources
from app.chains.rag_chain import format_documents, generate_rag_answer
from app.services.web_search_service import search_web, format_search_results


logger = logging.getLogger(__name__)


class RAGAgent:
    """
    Coordinates document loading, semantic retrieval, and
    grounded multi-source synthesis.
    """

    def __init__(self):
        self.agent_name = "NEXORA RAG Agent"

    def ingest_file(
        self,
        file_path: str,
        conversation_id: int
    ) -> dict:
        """
        Load, split, embed, and store document in PGVector.
        """
        return ingest_document(
            file_path=file_path,
            conversation_id=conversation_id
        )

    def query(
        self,
        question: str,
        conversation_id: int,
        search_k: int = 5,
        allow_web_fallback: bool = True
    ) -> dict:
        """
        Answers a user question prioritizing uploaded documents,
        with optional DuckDuckGo web search fallback.
        """
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        # 1. Retrieve candidate chunks from PGVector
        docs = retrieve_documents(
            question=clean_question,
            search_k=search_k,
            conversation_id=conversation_id
        )

        doc_context = format_documents(docs) if docs else ""
        sources = extract_sources(docs) if docs else []
        web_sources = []
        web_context = ""

        # 2. Web search fallback if document context is missing or thin
        if (not doc_context or len(doc_context.strip()) < 50) and allow_web_fallback:
            logger.info("Document context insufficient. Querying DuckDuckGo fallback...")
            try:
                web_results = search_web(query=clean_question, max_results=3)
                web_context = format_search_results(web_results)
                web_sources = [
                    {"title": r.get("title"), "url": r.get("url")}
                    for r in web_results if r.get("url")
                ]
            except Exception as e:
                logger.warning("Web search fallback failed: %s", e)

        # 3. Assemble combined evidence
        combined_context_parts = []
        if doc_context:
            combined_context_parts.append(f"DOCUMENT CONTEXT:\n{doc_context}")
        if web_context:
            combined_context_parts.append(f"WEB SEARCH EVIDENCE:\n{web_context}")

        if not combined_context_parts:
            return {
                "answer": "I could not find relevant information in your uploaded documents or online.",
                "context": "",
                "sources": [],
                "web_sources": [],
                "retrieval_status": "no_evidence"
            }

        full_context = "\n\n".join(combined_context_parts)

        # 4. Generate grounded answer
        answer = generate_rag_answer(
            context=full_context,
            question=clean_question
        )

        # 5. Attach citations
        citation_text = []
        if sources:
            citation_text.append("\n\n**Document Sources:**")
            for s in sources:
                fn = s.get("filename") or s.get("source") or "Document"
                page = s.get("page")
                page_str = f" (Page {page})" if page not in (None, "Unknown") else ""
                citation_text.append(f"- 📄 `{fn}`{page_str}")

        if web_sources:
            citation_text.append("\n\n**Web Sources:**")
            for w in web_sources:
                citation_text.append(f"- 🌐 [{w.get('title', 'Web')}]({w.get('url')})")

        if citation_text:
            answer += "".join(citation_text)

        return {
            "answer": answer,
            "context": full_context,
            "sources": sources,
            "web_sources": web_sources,
            "retrieval_status": "success"
        }


rag_agent = RAGAgent()