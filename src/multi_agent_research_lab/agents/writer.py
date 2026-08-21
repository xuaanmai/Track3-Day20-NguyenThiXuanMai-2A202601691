"""Writer agent implementation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes with verified citations."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer` with comprehensive content and citations."""
        system_prompt = (
            "You are a Principal Technical Writer in a research laboratory. "
            "Synthesize a polished, authoritative research report based on the provided analysis.\n"
            "Requirements:\n"
            "1. Tailor depth and tone to the target audience.\n"
            "2. Ground every major claim with numeric bracket citations like [1], [2].\n"
            "3. Conclude with an explicit '### References' section of sources.\n"
            "4. Maintain structural clarity with Executive Summary and Deep Dive."
        )

        sources_ref_text = "\n".join(
            f"[{i}] {doc.title} ({doc.url or 'N/A'}) - {doc.snippet}"
            for i, doc in enumerate(state.sources, 1)
        )

        user_prompt = (
            f"Topic: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Analyst Findings:\n{state.analysis_notes or 'N/A'}\n\n"
            f"Available Sources:\n{sources_ref_text or state.research_notes or 'None'}\n\n"
            "Write the comprehensive final report."
        )

        logger.info("Writer synthesizing final report for query: %s", state.request.query)
        try:
            response = self.llm_client.complete(
                system_prompt=system_prompt, user_prompt=user_prompt
            )
            content = response.content.strip()

            # Ensure References section exists if sources are present
            if state.sources and "### References" not in content and "## References" not in content:
                ref_list = [
                    f"[{i}] {doc.title}" + (f" ({doc.url})" if doc.url else "")
                    for i, doc in enumerate(state.sources, 1)
                ]
                content += "\n\n### References\n" + "\n".join(ref_list)

            state.final_answer = content
            metadata = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            }
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=state.final_answer,
                    metadata=metadata,
                )
            )
            state.add_trace_event("writer.done", metadata)
        except Exception as exc:
            logger.error("Writer error: %s", exc)
            state.errors.append(f"Writer failed: {exc}")

        return state
