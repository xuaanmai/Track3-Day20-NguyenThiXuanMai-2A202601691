"""Supervisor / router implementation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.max_iterations = self.settings.max_iterations

    def decide_route(self, state: ResearchState) -> str:
        """Determine the next destination route based on current state."""
        # 1. Guard against infinite loops
        if state.iteration >= self.max_iterations:
            return "done"

        # 2. Check if we have errors that prevent further processing
        if state.errors and not state.sources:
            return "done"

        # 3. Missing research sources
        if not state.sources:
            return "researcher"

        # 4. Missing analytical synthesis
        if not state.analysis_notes:
            return "analyst"

        # 5. Missing final answer
        if not state.final_answer:
            return "writer"

        # 6. Optional Critic / Quality review stage
        has_critic = any(res.agent == AgentName.CRITIC for res in state.agent_results)
        if not has_critic:
            return "critic"

        # 7. Workflow completed
        return "done"

    def run(self, state: ResearchState) -> ResearchState:
        """Evaluate state, determine next step, and update route history."""
        next_route = self.decide_route(state)
        state.record_route(next_route)
        state.add_trace_event(
            "supervisor.decision",
            {
                "next_route": next_route,
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_analysis": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
            },
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"Routing to: {next_route} (iteration {state.iteration})",
                metadata={"next_route": next_route, "iteration": state.iteration},
            )
        )
        logger.info(
            "Supervisor decided next route: %s (iteration %d)",
            next_route,
            state.iteration,
        )
        return state
