"""Unit tests for multi-agent roles and routing policy."""

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_supervisor_routing_sequence() -> None:
    supervisor = SupervisorAgent()
    state = ResearchState(request=ResearchQuery(query="GraphRAG architecture"))

    # Initial state -> needs researcher
    assert supervisor.decide_route(state) == "researcher"
    state = supervisor.run(state)
    assert state.route_history[-1] == "researcher"

    # Add sources -> needs analyst
    state.sources = [SourceDocument(title="Doc 1", snippet="Content 1")]
    assert supervisor.decide_route(state) == "analyst"

    # Add analysis -> needs writer
    state.analysis_notes = "Key analysis notes"
    assert supervisor.decide_route(state) == "writer"

    # Add final answer -> needs critic
    state.final_answer = "Draft final answer"
    assert supervisor.decide_route(state) == "critic"

    # Critic completed -> done
    state.agent_results.append(AgentResult(agent=AgentName.CRITIC, content="Approved"))
    assert supervisor.decide_route(state) == "done"


def test_supervisor_max_iterations_guard() -> None:
    supervisor = SupervisorAgent()
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.iteration = supervisor.max_iterations + 1
    assert supervisor.decide_route(state) == "done"


def test_researcher_agent() -> None:
    agent = ResearcherAgent()
    state = ResearchState(request=ResearchQuery(query="RAG survey", max_sources=2))
    result_state = agent.run(state)
    assert len(result_state.sources) > 0
    assert result_state.research_notes is not None
    assert any(res.agent == AgentName.RESEARCHER for res in result_state.agent_results)


def test_analyst_agent() -> None:
    agent = AnalystAgent()
    state = ResearchState(request=ResearchQuery(query="RAG vs Fine-tuning"))
    state.sources = [SourceDocument(title="RAG Paper", snippet="RAG uses retrieval.")]
    state.research_notes = "- RAG Paper: RAG uses retrieval."
    result_state = agent.run(state)
    assert result_state.analysis_notes is not None
    assert any(res.agent == AgentName.ANALYST for res in result_state.agent_results)


def test_writer_agent() -> None:
    agent = WriterAgent()
    state = ResearchState(request=ResearchQuery(query="RAG vs Fine-tuning"))
    state.sources = [
        SourceDocument(title="Doc 1", url="https://example.com/1", snippet="Snippet 1")
    ]
    state.analysis_notes = "Comparative analysis findings"
    result_state = agent.run(state)
    assert result_state.final_answer is not None
    assert any(res.agent == AgentName.WRITER for res in result_state.agent_results)


def test_critic_agent() -> None:
    agent = CriticAgent()
    state = ResearchState(request=ResearchQuery(query="RAG vs Fine-tuning"))
    state.sources = [
        SourceDocument(title="Doc 1", url="https://example.com/1", snippet="Snippet 1")
    ]
    state.final_answer = "Here is the report with Doc 1 cited."
    result_state = agent.run(state)
    assert any(res.agent == AgentName.CRITIC for res in result_state.agent_results)


def test_multi_agent_workflow_e2e() -> None:
    workflow = MultiAgentWorkflow()
    initial_state = ResearchState(request=ResearchQuery(query="Multi-agent collaboration"))
    final_state = workflow.run(initial_state)

    assert final_state.final_answer is not None
    assert len(final_state.sources) > 0
    assert final_state.research_notes is not None
    assert final_state.analysis_notes is not None
    assert "researcher" in final_state.route_history
    assert "analyst" in final_state.route_history
    assert "writer" in final_state.route_history
    assert final_state.iteration > 0
