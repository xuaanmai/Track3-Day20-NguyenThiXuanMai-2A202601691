"""Unit tests for services (LLMClient & SearchClient)."""

from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


def test_llm_client_fallback_complete() -> None:
    client = LLMClient()
    res = client.complete(system_prompt="You are an analyst.", user_prompt="Explain GraphRAG.")
    assert res.content
    assert res.input_tokens and res.input_tokens > 0
    assert res.output_tokens and res.output_tokens > 0
    assert res.cost_usd is not None


def test_llm_client_gemini_complete(monkeypatch) -> None:
    from unittest.mock import MagicMock

    client = LLMClient(api_key="mock-key", model="gemini-2.0-flash")
    mock_resp = MagicMock()
    mock_resp.text = "Generated content by Gemini"
    mock_resp.usage_metadata.prompt_token_count = 100
    mock_resp.usage_metadata.candidates_token_count = 50

    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.return_value = mock_resp
    client._gemini_client = mock_gemini

    res = client.complete(system_prompt="You are an assistant.", user_prompt="Hello")
    assert res.content == "Generated content by Gemini"
    assert res.input_tokens == 100
    assert res.output_tokens == 50
    assert res.cost_usd is not None


def test_llm_client_gemini_error_fallback() -> None:
    from unittest.mock import MagicMock

    client = LLMClient(api_key="mock-key", model="gemini-2.0-flash")
    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = RuntimeError("API down")
    client._gemini_client = mock_gemini

    res = client.complete(system_prompt="You are a critic.", user_prompt="Verify")
    assert "Verification" in res.content


def test_search_client_search() -> None:
    client = SearchClient()
    docs = client.search(query="GraphRAG knowledge graph", max_results=3)
    assert len(docs) <= 3
    assert len(docs) > 0
    assert docs[0].title
    assert docs[0].snippet
