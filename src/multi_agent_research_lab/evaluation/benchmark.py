"""Benchmark evaluation for single-agent vs multi-agent workflows."""

from __future__ import annotations

import logging
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)
Runner = Callable[[str], ResearchState]


def compute_citation_coverage(state: ResearchState) -> float:
    """Calculate the ratio of retrieved sources referenced in the final answer."""
    if not state.sources or not state.final_answer:
        return 0.0

    answer_lower = state.final_answer.lower()
    cited_count = 0

    for doc in state.sources:
        title_words = [w.lower() for w in doc.title.split() if len(w) > 4]
        is_cited = (
            (doc.url and doc.url.lower() in answer_lower)
            or doc.title.lower() in answer_lower
            or (title_words and any(word in answer_lower for word in title_words))
        )
        if is_cited:
            cited_count += 1

    return round(min(1.0, cited_count / len(state.sources)), 2)


def compute_quality_score(state: ResearchState) -> float:
    """Heuristic quality scoring (0.0 to 10.0) based on structure, depth, citations, and errors."""
    if not state.final_answer or state.errors:
        return 2.0 if state.final_answer else 0.0

    score = 5.0
    text = state.final_answer

    # 1. Structural headings and organization
    if "##" in text or "###" in text:
        score += 1.5

    # 2. Citations / References presence
    if "references" in text.lower() or "[" in text:
        score += 1.5

    # 3. Source grounding
    coverage = compute_citation_coverage(state)
    score += coverage * 1.5

    # 4. Length and technical depth
    if len(text.split()) > 100:
        score += 0.5

    return round(min(10.0, score), 1)


def compute_estimated_cost(state: ResearchState) -> float:
    """Aggregate total USD cost recorded across all agent results."""
    total_cost = 0.0
    for res in state.agent_results:
        cost = res.metadata.get("cost_usd")
        if cost and isinstance(cost, (int, float)):
            total_cost += float(cost)
    return round(total_cost, 6)


def compute_failure_rate(state: ResearchState) -> float:
    """Return 1.0 if fatal errors occurred or final answer missing, else 0.0."""
    if not state.final_answer or len(state.errors) > 2:
        return 1.0
    return 0.0


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Execute runner, measure latency, quality, cost, and citation metrics."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=compute_estimated_cost(state),
        quality_score=compute_quality_score(state),
        citation_coverage=compute_citation_coverage(state),
        failure_rate=compute_failure_rate(state),
        notes=f"Completed in {state.iteration} iterations with {len(state.sources)} sources",
    )
    return state, metrics
