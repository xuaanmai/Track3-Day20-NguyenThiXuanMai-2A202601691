"""Critic agent implementation for fact-checking and quality audit."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Fact-checking, citation audit, and hallucination verification agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer, verify citation coverage, and append findings."""
        if not state.final_answer:
            state.errors.append("Critic cannot evaluate: missing final answer.")
            return state

        # Compute citation coverage
        cited_count = 0
        answer_text = state.final_answer.lower()
        for doc in state.sources:
            is_cited = (
                (doc.url and doc.url.lower() in answer_text)
                or doc.title.lower() in answer_text
                or any(word.lower() in answer_text for word in doc.title.split() if len(word) > 5)
            )
            if is_cited:
                cited_count += 1

        coverage = (cited_count / len(state.sources)) if state.sources else 1.0

        system_prompt = (
            "You are a Quality & Fact-Checking Critic in a multi-agent research team. "
            "Review the final answer against the retrieved sources and analysis. "
            "Confirm that claims are substantiated, citations are accurate, and no errors exist."
        )

        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Sources Cited: {len(state.sources)}\n"
            f"Final Draft:\n{state.final_answer}\n\n"
            "Provide an audit verdict (Score 0-10, Citation Fidelity, and Status)."
        )

        logger.info("Critic auditing final answer for query: %s", state.request.query)
        try:
            response = self.llm_client.complete(
                system_prompt=system_prompt, user_prompt=user_prompt
            )
            audit_report = (
                f"### Quality Audit Report\n"
                f"- **Citation Coverage**: {coverage:.0%}\n"
                f"- **Review Summary**: {response.content}\n"
            )

            metadata = {
                "citation_coverage": coverage,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            }
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=audit_report,
                    metadata=metadata,
                )
            )
            state.add_trace_event("critic.done", metadata)
        except Exception as exc:
            logger.error("Critic error: %s", exc)
            state.errors.append(f"Critic failed: {exc}")

        return state
