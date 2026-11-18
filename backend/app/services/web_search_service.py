
"""
NEXORA AI - Web Search Service

Provides web search fallback when uploaded documents
do not contain sufficient information.

Provider: DuckDuckGo
"""

import logging
from typing import Any

from ddgs import DDGS


logger = logging.getLogger(__name__)


# ----------------------------------------
# DEFAULT CONFIGURATION
# ----------------------------------------

DEFAULT_MAX_RESULTS = 5
MAX_QUERY_LENGTH = 500


# ----------------------------------------
# WEB SEARCH
# ----------------------------------------

def search_web(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS
) -> list[dict[str, Any]]:
    """
    Search the web using DuckDuckGo.

    Returns normalized search results containing:
    - title
    - url
    - snippet
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")

    query = query.strip()

    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"Search query cannot exceed {MAX_QUERY_LENGTH} characters."
        )

    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or not 1 <= max_results <= 10
    ):
        raise ValueError("max_results must be between 1 and 10.")

    try:

        with DDGS(timeout=15) as search_client:

            raw_results = search_client.text(
                query,
                max_results=max_results,
                safesearch="moderate"
            )

            results = []

            for item in raw_results:

                title = item.get("title", "")
                url = item.get("href", "")
                snippet = item.get("body", "")

                if not url:
                    continue

                results.append({
                    "title": title.strip(),
                    "url": url.strip(),
                    "snippet": snippet.strip()
                })

            logger.info(
                "Web search completed. Results found: %s",
                len(results)
            )

            return results

    except Exception as error:

        logger.exception("DuckDuckGo web search failed.")

        raise RuntimeError(
            "Web search is currently unavailable."
        ) from error


# ----------------------------------------
# FORMAT SEARCH RESULTS
# ----------------------------------------

def format_search_results(
    results: list[dict[str, Any]]
) -> str:
    """
    Convert search results into context for the LLM.
    """

    if not results:
        return ""

    formatted_results = []

    for index, result in enumerate(results, start=1):

        formatted_results.append(
            f"""
Source {index}

Title: {result.get("title", "Unknown")}

URL: {result.get("url", "")}

Content:
{result.get("snippet", "")}
""".strip()
        )

    return "\n\n".join(formatted_results)
