"""Unit tests for benchmark metrics and evaluation."""

from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    compute_citation_coverage,
    compute_quality_score,
    run_benchmark,
)


def test_compute_citation_coverage() -> None:
    state = ResearchState(request=ResearchQuery(query="GraphRAG overview"))
    state.sources = [
        SourceDocument(
            title="GraphRAG Paper",
            url="https://arxiv.org/abs/2404.16130",
            snippet="GraphRAG summary",
        ),
        SourceDocument(
            title="Survey of LLMs",
            url="https://example.com/survey",
            snippet="LLM overview",
        ),
    ]
    state.final_answer = (
        "According to GraphRAG Paper (https://arxiv.org/abs/2404.16130), "
        "graph structures enhance LLMs."
    )
    coverage = compute_citation_coverage(state)
    assert coverage == 0.5


def test_quality_score() -> None:
    state = ResearchState(request=ResearchQuery(query="RAG overview"))
    state.sources = [SourceDocument(title="RAG Paper", snippet="RAG survey")]
    state.final_answer = "## Summary\n\n### References\n[1] RAG Paper"
    score = compute_quality_score(state)
    assert score >= 7.0


def test_run_benchmark() -> None:
    def dummy_runner(q: str) -> ResearchState:
        st = ResearchState(request=ResearchQuery(query=q))
        st.final_answer = "Final answer content"
        return st

    state, metrics = run_benchmark("dummy_run", "Test query", dummy_runner)
    assert metrics.run_name == "dummy_run"
    assert metrics.latency_seconds >= 0.0
    assert metrics.quality_score is not None
