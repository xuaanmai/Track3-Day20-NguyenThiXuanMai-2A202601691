"""LangGraph workflow implementation for the Multi-Agent Research System."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and orchestrates the multi-agent graph with LangGraph."""

    def __init__(
        self,
        settings: Settings | None = None,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(self.settings)
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self.critic = critic or CriticAgent()
        self._compiled_graph: Any | None = None

    def _ensure_state(self, state: Any) -> ResearchState:
        if isinstance(state, ResearchState):
            return state
        return ResearchState.model_validate(state)

    def _route_condition(self, state: Any) -> str:
        st = self._ensure_state(state)
        if st.route_history:
            return st.route_history[-1]
        return "done"

    def build(self) -> Any:
        """Create and configure the LangGraph stateful orchestration graph."""
        builder = StateGraph(ResearchState)

        # 1. Add nodes
        builder.add_node("supervisor", lambda s: self.supervisor.run(self._ensure_state(s)))
        builder.add_node("researcher", lambda s: self.researcher.run(self._ensure_state(s)))
        builder.add_node("analyst", lambda s: self.analyst.run(self._ensure_state(s)))
        builder.add_node("writer", lambda s: self.writer.run(self._ensure_state(s)))
        builder.add_node("critic", lambda s: self.critic.run(self._ensure_state(s)))

        # 2. Set entry point
        builder.set_entry_point("supervisor")

        # 3. Add conditional routing edges from supervisor
        builder.add_conditional_edges(
            "supervisor",
            self._route_condition,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                "done": END,
            },
        )

        # 4. Add cyclical edges back to supervisor
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", "supervisor")
        builder.add_edge("critic", "supervisor")

        self._compiled_graph = builder.compile()
        return self._compiled_graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow graph and return the final ResearchState."""
        if self._compiled_graph is None:
            self.build()

        assert self._compiled_graph is not None
        logger.info("Starting MultiAgentWorkflow execution for query: %s", state.request.query)
        result = self._compiled_graph.invoke(state)
        final_state = self._ensure_state(result)
        logger.info(
            "Workflow finished with %d iterations. Route history: %s",
            final_state.iteration,
            final_state.route_history,
        )
        return final_state
