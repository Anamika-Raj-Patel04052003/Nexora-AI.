"""
NEXORA AI - Creative Studio (Blog Generation Service)

Professional blog generation service using Groq LLM.
Generates structured Markdown content with:
- LaTeX mathematical and scientific formulas ($...$ inline, $$...$$ block)
- Rich Markdown comparison tables and data matrices
- Photorealistic inline image placeholders (<!-- NEXORA_IMAGE: descriptive prompt -->)
"""

import logging
from langchain_groq import ChatGroq
from app.config.settings import settings

logger = logging.getLogger(__name__)

MAX_TOPIC_LENGTH = 500
MAX_LANGUAGE_LENGTH = 50
MAX_TONE_LENGTH = 50

BLOG_LENGTH_GUIDELINES = {
    "Short": "Approximately 400-600 words with 1 comparison table and 1 relevant image placeholder.",
    "Medium": "Approximately 800-1200 words with 1-2 comparison tables, mathematical/technical formulas where applicable, and 2 relevant image placeholders.",
    "Long": "Approximately 1500-2000 words comprehensive guide with 2+ tables, detailed formulas/derivations, and 2-3 relevant image placeholders."
}


def generate_blog(
    topic: str,
    language: str = "English",
    tone: str = "Professional",
    length: str = "Medium"
) -> str:
    """
    Generate an in-depth, structured Markdown blog using Groq LLM.
    Includes rich tables, LaTeX formulas, and image placeholders.
    """

    # ----------------------------------------------
    # 1. INPUT VALIDATION
    # ----------------------------------------------

    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Blog topic cannot be empty.")

    topic = topic.strip()
    if len(topic) > MAX_TOPIC_LENGTH:
        raise ValueError(
            f"Blog topic cannot exceed {MAX_TOPIC_LENGTH} characters."
        )

    if not isinstance(language, str) or not language.strip():
        raise ValueError("Blog language cannot be empty.")

    language = language.strip()
    if len(language) > MAX_LANGUAGE_LENGTH:
        raise ValueError("Blog language is too long.")

    if not isinstance(tone, str) or not tone.strip():
        raise ValueError("Blog tone cannot be empty.")

    tone = tone.strip()
    if len(tone) > MAX_TONE_LENGTH:
        raise ValueError("Blog tone is too long.")

    if length not in BLOG_LENGTH_GUIDELINES:
        raise ValueError(
            "Invalid blog length. Choose 'Short', 'Medium', or 'Long'."
        )

    # ----------------------------------------------
    # 2. LLM INITIALIZATION
    # ----------------------------------------------

    try:
        llm = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0.65,
            max_retries=2,
            timeout=90
        )

        # ------------------------------------------
        # 3. ENHANCED PROMPT WITH FORMULAS & TABLES
        # ------------------------------------------

        prompt = f"""
You are the elite creative writing and technical publication engine of NEXORA AI.

Your mission is to produce an exceptional, comprehensive, publication-ready Markdown blog.

BLOG SPECIFICATIONS:
- Topic: {topic}
- Target Language: {language}
- Tone: {tone}
- Target Length: {length} ({BLOG_LENGTH_GUIDELINES[length]})

MANDATORY STRUCTURAL REQUIREMENTS:

1. TITLE & SUBTITLE:
   - Start with a compelling H1 title (# Title).
   - Follow immediately with a 2-line italicized overview / hook.

2. EXECUTIVE SUMMARY / KEY TAKEAWAYS:
   - Provide an Executive Summary box using Markdown blockquotes with 3-4 bullet points.

3. SECTION HEADINGS:
   - Organize logically into multiple sections using H2 (##) and subsections with H3 (###).
   - Use clean, engaging headings.

4. RICH DATA & COMPARISON TABLES:
   - Include at least ONE well-formatted Markdown comparison or specifications table.
   - Example table format:
     | Feature / Metric | Approach A | Approach B | Significance |
     |:---|:---:|:---:|:---|
     | Metric 1 | Value | Value | Explanation |
   - Tables must be visually balanced with valid alignment colons.

5. MATHEMATICAL & TECHNICAL FORMULAS (LaTeX):
   - Whenever discussing algorithms, economics, physics, data science, statistics, ROI, or benchmarks:
     - Use inline LaTeX: $formula$ (e.g. $E = mc^2$ or $\\mathcal{{O}}(n \\log n)$)
     - Use display block LaTeX for key equations:
       $$
       \\sigma = \\sqrt{{\\frac{{\\sum_{{i=1}}^{{N}} (x_i - \\mu)^2}}{{N}}}}
       $$
   - Ensure backslashes are valid LaTeX.

6. IN-BETWEEN PHOTOREALISTIC IMAGE PLACEHOLDERS:
   - Place between 1 and 3 image placeholders strategically between sections (never inside code blocks or tables).
   - Exact format required:
     <!-- NEXORA_IMAGE: A photorealistic, high-resolution visual representing [detailed visual concept] -->
   - The visual description inside must be rich, vibrant, descriptive, and relevant to the section.

7. FAQ SECTION:
   - Include 3 frequently asked questions with direct, insightful answers before the conclusion.

8. CONCLUSION & ACTIONABLE TAKEAWAY:
   - Summarize final thoughts with actionable next steps.

IMPORTANT CONSTRAINTS:
- Write strictly in {language}.
- Output ONLY valid GitHub Flavored Markdown (no conversational filler before or after the article).
- Maintain high factual accuracy and avoid hallucinating specific statistical claims.
"""

        # ------------------------------------------
        # 4. EXECUTION
        # ------------------------------------------

        response = llm.invoke(prompt)
        content = response.content

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Groq LLM returned an empty blog response.")

        content = content.strip()

        # Remove outer code block wrapping if model wrapped markdown in ```markdown
        if content.startswith("```markdown") and content.endswith("```"):
            content = content[11:-3].strip()
        elif content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()

        logger.info(
            "Enhanced blog generated successfully for topic: '%s'",
            topic[:80]
        )

        return content

    except ValueError:
        raise
    except Exception as error:
        logger.exception("Blog generation failed.")
        raise RuntimeError(f"Blog generation failed: {error}") from error