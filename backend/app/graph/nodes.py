"""
NEXORA AI - LangGraph Nodes

Complete implementation of:
1. Input validation & state initialization
2. Supervisor-backed Multimodal Intent Routing:
   - Document RAG (CRAG + Self-RAG + Web search)
   - In-chat Image Generation (FLUX / Local SD)
   - In-chat Audio Generation (Edge-TTS)
   - In-chat Video Slideshow Creation (MoviePy)
   - In-chat Blog Generation (Formulas, Tables, Figures)
   - Normal Conversational Chat (Groq)
3. Document retrieval from PGVector
4. CRAG: Relevance grading (relevant vs irrelevant chunks)
5. Query rewriting for ambiguous/failed retrievals
6. Web search fallback via DuckDuckGo (ddgs)
7. Answer generation with document-first prioritization
8. Self-RAG: Grounding & relevance evaluation + bounded revision
9. Response packaging & history management
"""

import asyncio
import json
import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.chains.chat_chain import generate_chat_response
from app.chains.rag_chain import format_documents
from app.rag.retriever import retrieve_documents
from app.rag.rag_chain import extract_sources
from app.services.llm_service import llm
from app.services.web_search_service import search_web, format_search_results
from app.services.image_service import generate_image
from app.services.audio_generation_service import generate_audio
from app.services.creative_service import generate_blog
from app.services.blog_media_service import add_images_to_blog
from app.services.video_creation_service import create_local_video
from app.services.blog_audio_service import generate_blog_audio
from app.monitoring.evaluation import evaluate_answer
from app.agents.supervisor import classify_user_intent


logger = logging.getLogger(__name__)


# ==========================================
# NODE 1: INPUT NODE
# ==========================================

def input_node(state: dict) -> dict:
    user_input = state.get("user_input", "")

    if not isinstance(user_input, str) or not user_input.strip():
        return {"error": "User input cannot be empty."}

    clean_input = user_input.strip()

    return {
        "user_input": clean_input,
        "query": clean_input,
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", 2),
        "error": ""
    }


# ==========================================
# NODE 2: SUPERVISOR INTENT NODE
# ==========================================

def intent_node(state: dict) -> dict:
    """
    Supervisor Agent analyzes the prompt and decides whether to route to:
    - rag
    - generate_image
    - generate_audio
    - generate_video
    - generate_blog
    - chat
    """
    explicit_intent = state.get("intent", "auto")
    user_input = state.get("user_input", "")
    conversation_id = state.get("conversation_id")

    if explicit_intent in {"rag", "chat", "generate_image", "generate_audio", "generate_video", "generate_blog"}:
        return {
            "intent": explicit_intent,
            "cleaned_prompt": user_input
        }

    # Use Supervisor Agent for automatic multimodal routing
    routing_decision = classify_user_intent(
        user_input=user_input,
        conversation_id=conversation_id
    )

    logger.info("Supervisor routed request to: %s", routing_decision["intent"])

    return {
        "intent": routing_decision["intent"],
        "cleaned_prompt": routing_decision.get("cleaned_prompt") or user_input
    }


# ==========================================
# NODE: GENERATE IMAGE NODE (IN-CHAT)
# ==========================================

def generate_image_node(state: dict) -> dict:
    prompt = state.get("cleaned_prompt") or state.get("user_input", "")
    logger.info("In-chat image generation triggered for: '%s'", prompt[:60])

    try:
        result = generate_image(prompt=prompt, mode="cloud")
        filename = result["filename"]
        image_url = f"/api/creative/images/{filename}"

        markdown_response = (
            f"Here is the image you requested:\n\n"
            f"![{prompt}]({image_url})\n\n"
            f"*Prompt: {prompt}*\n\n"
            f"[View Full Resolution Image]({image_url})"
        )

        return {
            "answer": markdown_response,
            "response": markdown_response,
            "final_answer": markdown_response,
            "response_type": "creative_image"
        }
    except Exception as e:
        logger.exception("In-chat image generation failed: %s", e)
        error_msg = f"I attempted to generate an image for '{prompt}', but encountered an issue: {str(e)}"
        return {
            "answer": error_msg,
            "response": error_msg,
            "final_answer": error_msg,
            "error": str(e)
        }


# ==========================================
# NODE: GENERATE AUDIO NODE (IN-CHAT)
# ==========================================

def generate_audio_node(state: dict) -> dict:
    prompt = state.get("cleaned_prompt") or state.get("user_input", "")
    logger.info("In-chat audio TTS triggered for: '%s'", prompt[:60])

    try:
        result = asyncio.run(generate_audio(text=prompt))
        filename = result["filename"]
        audio_url = f"/api/creative/audio/{filename}"

        markdown_response = (
            f"Here is your generated audio speech:\n\n"
            f"🎙 [Listen to Audio Narration]({audio_url})\n\n"
            f"> *Narrated Text:* {prompt[:250]}..."
        )

        return {
            "answer": markdown_response,
            "response": markdown_response,
            "final_answer": markdown_response,
            "response_type": "creative_audio"
        }
    except Exception as e:
        logger.exception("In-chat audio generation failed: %s", e)
        error_msg = f"Audio generation encountered an issue: {str(e)}"
        return {
            "answer": error_msg,
            "response": error_msg,
            "final_answer": error_msg,
            "error": str(e)
        }


# ==========================================
# NODE: GENERATE VIDEO NODE (IN-CHAT)
# ==========================================

def generate_video_node(state: dict) -> dict:
    prompt = state.get("cleaned_prompt") or state.get("user_input", "")
    logger.info("In-chat video slideshow compilation triggered for: '%s'", prompt[:60])

    try:
        # Step 1: Generate 2 key scenes for the video
        img1 = generate_image(prompt=f"{prompt}, cinematic wide shot, 4k", mode="cloud")
        img2 = generate_image(prompt=f"{prompt}, detailed close-up shot, 4k", mode="cloud")
        image_filenames = [img1["filename"], img2["filename"]]

        # Step 2: Generate narration audio
        narration_text = f"Exploring {prompt}. Experience the future of intelligent multimodal creation with NEXORA AI."
        audio_res = asyncio.run(generate_audio(text=narration_text))
        audio_filename = audio_res["filename"]

        # Step 3: Compile into MP4 slideshow
        video_res = create_local_video(
            image_filenames=image_filenames,
            audio_filename=audio_filename
        )
        video_filename = video_res["filename"]
        video_url = f"/api/creative/videos/{video_filename}"

        markdown_response = (
            f"Here is the video compilation you requested:\n\n"
            f"🎬 [Watch MP4 Video Slideshow]({video_url})\n\n"
            f"- **Topic:** {prompt}\n"
            f"- **Duration:** {video_res['duration']}s | **Resolution:** {video_res['resolution']}"
        )

        return {
            "answer": markdown_response,
            "response": markdown_response,
            "final_answer": markdown_response,
            "response_type": "creative_video"
        }
    except Exception as e:
        logger.exception("In-chat video generation failed: %s", e)
        error_msg = f"Video compilation encountered an issue: {str(e)}"
        return {
            "answer": error_msg,
            "response": error_msg,
            "final_answer": error_msg,
            "error": str(e)
        }


# ==========================================
# NODE: GENERATE BLOG NODE (IN-CHAT)
# ==========================================

def generate_blog_node(state: dict) -> dict:
    prompt = state.get("cleaned_prompt") or state.get("user_input", "")
    logger.info("In-chat blog publication triggered for: '%s'", prompt[:60])

    try:
        raw_blog = generate_blog(topic=prompt, length="Medium")
        media_res = add_images_to_blog(raw_blog, mode="cloud")
        final_blog = media_res["content"]

        return {
            "answer": final_blog,
            "response": final_blog,
            "final_answer": final_blog,
            "response_type": "creative_blog"
        }
    except Exception as e:
        logger.exception("In-chat blog generation failed: %s", e)
        error_msg = f"Blog publication encountered an issue: {str(e)}"
        return {
            "answer": error_msg,
            "response": error_msg,
            "final_answer": error_msg,
            "error": str(e)
        }


# ==========================================
# NODE 3: RETRIEVE DOCUMENTS NODE (RAG)
# ==========================================

def retrieve_documents_node(state: dict) -> dict:
    query = state.get("rewritten_query") or state.get("query") or state.get("user_input", "")
    conversation_id = state.get("conversation_id")

    if not conversation_id:
        return {
            "retrieved_documents": [],
            "retrieval_status": "missing_conversation_id",
            "needs_web_search": True
        }

    try:
        conv_id_int = int(conversation_id)
        docs = retrieve_documents(
            question=query,
            search_k=5,
            conversation_id=conv_id_int
        )

        if not docs:
            logger.info("No document chunks found for query: '%s'", query[:60])
            return {
                "retrieved_documents": [],
                "retrieved_context": "",
                "sources": [],
                "retrieval_status": "no_documents",
                "needs_web_search": True
            }

        sources = extract_sources(docs)

        return {
            "retrieved_documents": docs,
            "sources": sources,
            "retrieval_status": "documents_found",
            "needs_web_search": False
        }

    except Exception as e:
        logger.exception("Document retrieval failed: %s", e)
        return {
            "retrieved_documents": [],
            "retrieval_status": "error",
            "needs_web_search": True
        }


# ==========================================
# NODE 4: GRADE DOCUMENTS NODE (CRAG)
# ==========================================

def grade_documents_node(state: dict) -> dict:
    query = state.get("query") or state.get("user_input", "")
    documents = state.get("retrieved_documents", [])

    if not documents:
        return {
            "relevant_documents": [],
            "irrelevant_documents": [],
            "needs_web_search": True,
            "document_relevance": "none"
        }

    grader_prompt = """
You are a strict evaluator assessing whether a retrieved document chunk is relevant to a user's question.

Question: {question}

Document Chunk:
{document}

Assess if the chunk contains information that directly answers or helps answer the question.
Respond with ONLY valid JSON:
{{"relevant": true, "reason": "short explanation"}}
"""

    relevant_docs = []
    irrelevant_docs = []

    for doc in documents:
        try:
            formatted_prompt = grader_prompt.format(
                question=query,
                document=doc.page_content[:1500]
            )
            response = llm.invoke([HumanMessage(content=formatted_prompt)])
            cleaned = response.content.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)

            if data.get("relevant") is True:
                relevant_docs.append(doc)
            else:
                irrelevant_docs.append(doc)
        except Exception:
            relevant_docs.append(doc)

    logger.info(
        "CRAG grading complete: %d relevant, %d irrelevant chunks.",
        len(relevant_docs),
        len(irrelevant_docs)
    )

    if relevant_docs:
        context = format_documents(relevant_docs)
        return {
            "relevant_documents": relevant_docs,
            "irrelevant_documents": irrelevant_docs,
            "retrieved_context": context,
            "needs_web_search": False,
            "document_relevance": "relevant"
        }

    return {
        "relevant_documents": [],
        "irrelevant_documents": irrelevant_docs,
        "retrieved_context": "",
        "needs_web_search": True,
        "document_relevance": "irrelevant"
    }


# ==========================================
# NODE 5: REWRITE QUERY NODE
# ==========================================

def rewrite_query_node(state: dict) -> dict:
    current_query = state.get("query") or state.get("user_input", "")
    retry_count = state.get("retry_count", 0)

    prompt = f"""
You are an expert search query optimizer.
The following user query did not yield relevant document matches:
"{current_query}"

Rewrite this query into a concise, highly specific search query focusing on keywords and core concepts.
Return ONLY the rewritten query text, nothing else.
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()
        logger.info("Rewrote query from '%s' to '%s'", current_query, rewritten)
    except Exception:
        rewritten = current_query

    return {
        "rewritten_query": rewritten,
        "query": rewritten,
        "retry_count": retry_count + 1
    }


# ==========================================
# NODE 6: WEB SEARCH NODE
# ==========================================

def web_search_node(state: dict) -> dict:
    search_query = state.get("rewritten_query") or state.get("query") or state.get("user_input", "")
    logger.info("Executing web search fallback for: '%s'", search_query[:60])

    try:
        results = search_web(query=search_query, max_results=5)
        web_context = format_search_results(results)

        web_sources = [
            {"title": r.get("title"), "url": r.get("url")}
            for r in results if r.get("url")
        ]

        return {
            "web_results": results,
            "web_context": web_context,
            "web_sources": web_sources,
            "needs_web_search": False
        }
    except Exception as e:
        logger.warning("Web search fallback failed: %s", e)
        return {
            "web_results": [],
            "web_context": "",
            "web_sources": [],
            "needs_web_search": False
        }


# ==========================================
# NODE 7: GENERATE ANSWER NODE
# ==========================================

def generate_answer_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    doc_context = state.get("retrieved_context", "").strip()
    web_context = state.get("web_context", "").strip()

    sources = state.get("sources", [])
    web_sources = state.get("web_sources", [])

    system_prompt = """
You are NEXORA AI, an intelligent, factual AI assistant.

Your task is to answer the user's question using the provided context.

RULES:
1. Prioritize uploaded document context over web search results.
2. If document context contains the full answer, use it and cite the document filename.
3. If document context contains partial info and web context supplies the rest, clearly distinguish:
   - "According to your uploaded document: ..."
   - "According to external web sources: ..."
4. If no document was relevant and web search was used, state that the answer is based on web search.
5. Never invent or hallucinate facts not present in the context.
6. Answer in the same language as the user's question.
7. Use clean Markdown formatting with bullet points and bold headers where appropriate.
"""

    context_parts = []
    if doc_context:
        context_parts.append(f"=== UPLOADED DOCUMENT CONTEXT ===\n{doc_context}")
    if web_context:
        context_parts.append(f"=== EXTERNAL WEB SEARCH CONTEXT ===\n{web_context}")

    combined_context = "\n\n".join(context_parts) if context_parts else "No supporting context available."

    human_content = f"QUESTION:\n{user_input}\n\nSUPPORTING CONTEXT:\n{combined_context}"

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ])
        answer = response.content.strip()

        citation_lines = []
        if sources:
            citation_lines.append("\n\n**Document Sources:**")
            for s in sources:
                filename = s.get("filename") or s.get("source") or "Unknown"
                page = s.get("page")
                page_str = f" (Page {page})" if page not in (None, "Unknown") else ""
                citation_lines.append(f"- 📄 `{filename}`{page_str}")

        if web_sources:
            citation_lines.append("\n\n**Web Sources:**")
            for w in web_sources[:3]:
                title = w.get("title") or "Web Page"
                url = w.get("url")
                citation_lines.append(f"- 🌐 [{title}]({url})")

        if citation_lines:
            answer += "".join(citation_lines)

        return {
            "answer": answer,
            "response": answer,
            "answer_status": "generated",
            "response_type": "rag"
        }

    except Exception as e:
        logger.exception("Answer generation failed: %s", e)
        return {"error": "Failed to generate answer from context."}


# ==========================================
# NODE 8: EVALUATE ANSWER NODE (SELF-RAG)
# ==========================================

def evaluate_answer_node(state: dict) -> dict:
    question = state.get("user_input", "")
    answer = state.get("answer", "")
    doc_context = state.get("retrieved_context", "")
    web_context = state.get("web_context", "")
    context = (doc_context + "\n" + web_context).strip()

    if not context:
        return {
            "answer_relevant": True,
            "answer_grounded": True,
            "final_answer": answer
        }

    try:
        eval_result = evaluate_answer(
            question=question,
            context=context,
            answer=answer
        )

        relevant = eval_result.get("answer_relevant", True)
        grounded = eval_result.get("answer_grounded", True)
        feedback = eval_result.get("feedback", "")

        return {
            "answer_relevant": relevant,
            "answer_grounded": grounded,
            "evaluation_feedback": feedback,
            "final_answer": answer
        }

    except Exception as e:
        logger.warning("Self-RAG evaluation fallback: %s", e)
        return {
            "answer_relevant": True,
            "answer_grounded": True,
            "final_answer": answer
        }


# ==========================================
# NODE 9: REVISE ANSWER NODE
# ==========================================

def revise_answer_node(state: dict) -> dict:
    question = state.get("user_input", "")
    previous_answer = state.get("answer", "")
    feedback = state.get("evaluation_feedback", "Some claims are unsupported by context.")
    doc_context = state.get("retrieved_context", "")
    web_context = state.get("web_context", "")
    context = (doc_context + "\n" + web_context).strip()
    retry_count = state.get("retry_count", 0)

    prompt = f"""
You are NEXORA AI. Your previous response was evaluated as having unsupported or ungrounded claims.

QUESTION: {question}

SUPPORTING EVIDENCE:
{context}

PREVIOUS ANSWER:
{previous_answer}

EVALUATION FEEDBACK:
{feedback}

TASK:
Rewrite the answer so that EVERY factual statement is strictly supported by the SUPPORTING EVIDENCE.
Remove any speculative or ungrounded claims.
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        revised = response.content.strip()
        return {
            "answer": revised,
            "final_answer": revised,
            "response": revised,
            "retry_count": retry_count + 1
        }
    except Exception as e:
        logger.warning("Answer revision failed: %s", e)
        return {"retry_count": retry_count + 1}


# ==========================================
# NODE 10: CHAT NODE
# ==========================================

def chat_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    history = state.get("messages", [])

    if not user_input:
        return {"error": "Chat input is missing."}

    try:
        response = generate_chat_response(
            message=user_input,
            history=history
        )

        return {
            "response": response,
            "answer": response,
            "final_answer": response,
            "answer_status": "generated",
            "response_type": "chat"
        }
    except Exception:
        logger.exception("Chat node failed.")
        return {"error": "Unable to generate chat response."}


# ==========================================
# NODE 11: RESPONSE NODE
# ==========================================

def response_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    response = (
        state.get("final_answer")
        or state.get("response")
        or state.get("answer")
        or ""
    )

    if not response.strip():
        return {"error": "No response was generated."}

    new_messages = [
        HumanMessage(content=user_input),
        AIMessage(content=response)
    ]

    return {
        "messages": new_messages,
        "response": response,
        "final_answer": response
    }