
"""
NEXORA AI - LangSmith Monitoring

Configures LangSmith tracing for debugging,
observability and workflow monitoring.
"""

import logging
import os

from app.config.settings import settings


logger = logging.getLogger(__name__)


def configure_langsmith() -> bool:
    """
    Configure LangSmith tracing using application settings.

    Returns:
        bool: True when tracing is enabled, otherwise False.
    """

    if not settings.langsmith_tracing:
        logger.info("LangSmith tracing is disabled.")
        return False

    if not settings.langsmith_api_key:
        logger.warning(
            "LangSmith tracing is enabled but API key is missing."
        )
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

    logger.info(
        "LangSmith tracing configured for project: %s",
        settings.langsmith_project
    )

    return True


def get_langsmith_status() -> dict:
    """
    Return LangSmith configuration status.
    """

    enabled = (
        settings.langsmith_tracing
        and bool(settings.langsmith_api_key)
    )

    return {
        "enabled": enabled,
        "project": settings.langsmith_project,
        "configured": bool(settings.langsmith_api_key)
    }
