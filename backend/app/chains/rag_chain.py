
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.services.llm_service import llm


# ==========================================
# 1. FORMAT RETRIEVED DOCUMENTS
# ==========================================

def format_documents(documents):

    formatted = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        formatted.append(
            f"""
SOURCE {index}
File: {source}
Page: {page}

Content:
{document.page_content}
"""
        )

    return "\n\n".join(formatted)


# ==========================================
# 2. RAG SYSTEM PROMPT
# ==========================================

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are NEXORA AI, an intelligent multilingual
document assistant.

Answer questions using the retrieved document context.

Rules:

1. Carefully understand the user's question.
2. Use the provided context as your primary evidence.
3. Never invent information from the documents.
4. If the answer is unavailable in the context,
   clearly say that the documents do not contain
   sufficient information.
5. Respond in the same language as the user.
6. Explain answers clearly.
7. Mention relevant source files and page numbers
   when available.

Retrieved Context:

{context}
"""
        ),

        (
            "human",
            "{question}"
        )
    ]
)


# ==========================================
# 3. RAG LCEL CHAIN
# ==========================================

rag_chain = (
    RAG_PROMPT
    | llm
    | StrOutputParser()
)


# ==========================================
# 4. GENERATE ANSWER
# ==========================================

def generate_rag_answer(
    context: str,
    question: str
):

    return rag_chain.invoke(
        {
            "context": context,
            "question": question
        }
    )
