"""Analyst agent implementation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes` from research notes and sources."""
        if not state.sources and not state.research_notes:
            state.errors.append("Analyst cannot proceed: missing research sources.")
            state.analysis_notes = "No sources available for analysis."
            return state

        system_prompt = (
            "You are an expert Research Analyst in an advanced multi-agent system. "
            "Your task is to critically analyze the provided research notes and source materials.\n"
            "1. Identify core themes, claims, and architectural patterns.\n"
            "2. Compare perspectives, trade-offs, and nuances.\n"
            "3. Assess the credibility, limitations, and evidence strength.\n"
            "4. Organize your analysis with concise bullet points."
        )

        user_prompt = (
            f"User Research Query: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Research Notes & Sources:\n{state.research_notes}\n\n"
            "Provide structured analysis and key insights."
        )

        logger.info("Analyst processing research notes for query: %s", state.request.query)
        try:
            response = self.llm_client.complete(
                system_prompt=system_prompt, user_prompt=user_prompt
            )
            state.analysis_notes = response.content

            metadata = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            }
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=state.analysis_notes,
                    metadata=metadata,
                )
            )
            state.add_trace_event("analyst.done", metadata)
        except Exception as exc:
            logger.error("Analyst error: %s", exc)
            state.errors.append(f"Analyst failed: {exc}")

        return state
