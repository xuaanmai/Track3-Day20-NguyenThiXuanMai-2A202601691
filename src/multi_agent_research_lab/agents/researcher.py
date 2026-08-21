"""Researcher agent implementation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, search_client: SearchClient | None = None) -> None:
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        max_sources = state.request.max_sources

        logger.info("Researcher searching for query: %s (limit: %d)", query, max_sources)
        try:
            docs = self.search_client.search(query=query, max_results=max_sources)
            state.sources = docs

            if not docs:
                state.errors.append("Researcher found no documents.")
                state.research_notes = "No relevant source documents found."
            else:
                formatted_notes = []
                for i, doc in enumerate(docs, 1):
                    url_str = f" ({doc.url})" if doc.url else ""
                    formatted_notes.append(f"[{i}] **{doc.title}**{url_str}\n{doc.snippet}")
                state.research_notes = "\n\n".join(formatted_notes)

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=state.research_notes or "",
                    metadata={"num_sources": len(docs)},
                )
            )
            state.add_trace_event("researcher.done", {"num_sources": len(docs)})
        except Exception as exc:
            logger.error("Researcher error: %s", exc)
            state.errors.append(f"Researcher failed: {exc}")

        return state
