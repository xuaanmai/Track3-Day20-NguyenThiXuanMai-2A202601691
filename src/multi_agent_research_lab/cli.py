"""Command-line entrypoint for the Multi-Agent Research Lab."""

from __future__ import annotations

from time import perf_counter
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import setup_tracing_environment
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    setup_tracing_environment()


def _parse_query(
    query: str, max_sources: int = 5, audience: str = "technical learners"
) -> ResearchQuery:
    try:
        return ResearchQuery(query=query, max_sources=max_sources, audience=audience)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    audience: Annotated[
        str, typer.Option("--audience", "-a", help="Target audience")
    ] = "technical learners",
) -> None:
    """Run a real single-agent baseline LLM call and measure metrics."""
    _init()
    request = _parse_query(query=query, audience=audience)
    state = ResearchState(request=request)

    llm_client = LLMClient()
    system_prompt = f"You are a helpful AI research assistant. Provide an overview for {audience}."

    started = perf_counter()
    response = llm_client.complete(system_prompt=system_prompt, user_prompt=query)
    latency = perf_counter() - started

    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "latency_seconds": latency,
            },
        )
    )

    console.print(
        Panel(
            state.final_answer,
            title="[bold green]Single-Agent Baseline Response[/bold green]",
        )
    )

    metrics_table = Table(title="Baseline Execution Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="magenta")
    metrics_table.add_row("Latency", f"{latency:.3f} s")
    metrics_table.add_row("Input Tokens", str(response.input_tokens or "N/A"))
    metrics_table.add_row("Output Tokens", str(response.output_tokens or "N/A"))
    metrics_table.add_row(
        "Estimated Cost",
        f"${response.cost_usd:.6f}" if response.cost_usd else "N/A",
    )
    console.print(metrics_table)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    max_sources: Annotated[int, typer.Option("--max-sources", "-s", help="Max search sources")] = 5,
    audience: Annotated[
        str, typer.Option("--audience", "-a", help="Target audience")
    ] = "technical learners",
) -> None:
    """Run the complete multi-agent LangGraph workflow."""
    _init()
    request = _parse_query(query=query, max_sources=max_sources, audience=audience)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()

    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc

    console.print(
        Panel(
            result.final_answer or "No answer produced.",
            title=f"[bold green]Multi-Agent Answer ({len(result.sources)} sources)[/bold green]",
        )
    )

    summary_table = Table(title="Multi-Agent Workflow Summary")
    summary_table.add_column("Property", style="cyan")
    summary_table.add_column("Value", style="magenta")
    summary_table.add_row("Total Iterations", str(result.iteration))
    summary_table.add_row("Route Sequence", " -> ".join(result.route_history))
    summary_table.add_row("Sources Retrieved", str(len(result.sources)))
    summary_table.add_row("Trace Events", str(len(result.trace)))
    summary_table.add_row("Errors Encountered", str(len(result.errors)))
    console.print(summary_table)


@app.command("benchmark")
def benchmark(
    query: Annotated[
        str, typer.Option("--query", "-q", help="Research query")
    ] = "Research GraphRAG state-of-the-art",
) -> None:
    """Run comparative benchmark between single-agent baseline and multi-agent workflow."""
    _init()

    def run_single(q: str) -> ResearchState:
        st = ResearchState(request=_parse_query(q))
        res = LLMClient().complete("You are an assistant.", q)
        st.final_answer = res.content
        st.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=res.content,
                metadata={"cost_usd": res.cost_usd},
            )
        )
        return st

    def run_multi(q: str) -> ResearchState:
        st = ResearchState(request=_parse_query(q))
        return MultiAgentWorkflow().run(st)

    console.print(f"[bold yellow]Benchmarking query: '{query}'...[/bold yellow]")
    _, m_single = run_benchmark("Single-Agent Baseline", query, run_single)
    _, m_multi = run_benchmark("Multi-Agent Workflow", query, run_multi)

    report_md = render_markdown_report([m_single, m_multi])
    console.print(Panel(report_md, title="Benchmark Summary Report"))


if __name__ == "__main__":
    app()
