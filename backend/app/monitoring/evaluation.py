
"""
NEXORA AI - Self-RAG Evaluation

Evaluates answer relevance and factual grounding
against retrieved document context.
"""

import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from app.services.llm_service import llm


logger = logging.getLogger(__name__)


EVALUATION_PROMPT = """
You are an evaluator for a Retrieval-Augmented Generation system.

Evaluate the generated answer using the question and supplied context.

Return ONLY valid JSON with these fields:

{
    "answer_relevant": true,
    "answer_grounded": true,
    "feedback": "Short explanation",
    "score": 1
}

Rules:

1. answer_relevant:
   True if the answer directly addresses the user's question.

2. answer_grounded:
   True if factual claims in the answer are supported by the context.
   If the context is empty, set this to false.

3. feedback:
   Briefly explain the evaluation.

4. score:
   Integer between 0 and 1.

Do not assume facts that are not present in the context.
"""


def evaluate_answer(
    question: str,
    context: str,
    answer: str
) -> dict:
    """
    Evaluate answer relevance and grounding.

    Returns a structured evaluation result.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer cannot be empty.")

    if not isinstance(context, str):
        raise ValueError("Context must be a string.")

    if not context.strip():
        return {
            "answer_relevant": False,
            "answer_grounded": False,
            "feedback": "No supporting context was available.",
            "score": 0
        }

    messages = [
        SystemMessage(content=EVALUATION_PROMPT),
        HumanMessage(
            content=(
                f"QUESTION:\n{question}\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"GENERATED ANSWER:\n{answer}"
            )
        )
    ]

    try:

        result = llm.invoke(messages)

        raw_output = result.content

        if not isinstance(raw_output, str):
            raise ValueError("Evaluator returned an invalid response.")

        # Handle accidental Markdown code fences.
        cleaned_output = raw_output.strip()

        if cleaned_output.startswith("```"):
            cleaned_output = (
                cleaned_output
                .replace("```json", "", 1)
                .replace("```", "")
                .strip()
            )

        evaluation = json.loads(cleaned_output)

        relevant = evaluation.get("answer_relevant")
        grounded = evaluation.get("answer_grounded")
        feedback = evaluation.get("feedback")
        score = evaluation.get("score")

        if not isinstance(relevant, bool):
            raise ValueError("Invalid answer_relevant value.")

        if not isinstance(grounded, bool):
            raise ValueError("Invalid answer_grounded value.")

        if not isinstance(feedback, str):
            raise ValueError("Invalid evaluation feedback.")

        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= score <= 1
        ):
            raise ValueError("Invalid evaluation score.")

        return {
            "answer_relevant": relevant,
            "answer_grounded": grounded,
            "feedback": feedback,
            "score": float(score)
        }

    except Exception as error:

        logger.exception("Self-RAG evaluation failed.")

        raise RuntimeError(
            "Unable to evaluate the generated answer."
        ) from error
