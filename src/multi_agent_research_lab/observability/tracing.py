"""Tracing hooks for multi-agent execution observability."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


def setup_tracing_environment() -> None:
    """Initialize LangSmith/Langfuse environment variables if API keys are set."""
    settings = get_settings()

    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        logger.info("LangSmith tracing enabled for project: %s", settings.langsmith_project)

    if settings.langfuse_public_key and settings.langfuse_secret_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
        os.environ["LANGFUSE_HOST"] = settings.langfuse_host
        logger.info("Langfuse tracing configured for host: %s", settings.langfuse_host)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Span context for recording granular execution metrics and tracing."""
    started = perf_counter()
    timestamp = time.time()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "timestamp": timestamp,
        "duration_seconds": None,
        "status": "running",
    }
    try:
        yield span
        span["status"] = "success"
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = round(perf_counter() - started, 4)
        logger.debug(
            "Span [%s] completed in %.4fs (status: %s)",
            name,
            span["duration_seconds"],
            span["status"],
        )
