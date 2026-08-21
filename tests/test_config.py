from multi_agent_research_lab.core.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.gemini_model
    assert settings.openai_model == settings.gemini_model
    assert settings.max_iterations >= 1
