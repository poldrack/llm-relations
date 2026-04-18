"""Opt-in smoke test that hits the real Anthropic API.

Run with: uv run pytest -m smoke -v
Skipped unless the smoke marker is selected AND ANTHROPIC_API_KEY is set.
"""
import os

import pytest

from llm_relations.runner.client import ClaudeClient
from llm_relations.parser import parse_answer, ParseError


@pytest.mark.smoke
def test_real_haiku_call_returns_parseable_response():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    client = ClaudeClient(api_key=api_key)
    result = client.call(
        model="claude-haiku-4-5-20251001",
        user_prompt=(
            "Memory scenario: a florp is underneath a greeble. "
            "The florp has a blue button on top. Pressing the blue button on the florp activates it.\n\n"
            "Perception scenario: a mek is underneath a quib. "
            "The mek has a blue button on top.\n\n"
            "Which object in the perception scenario is the florp-analog?"
        ),
    )

    assert result.output_tokens > 0
    # Response should include a fenced JSON answer per the system prompt.
    try:
        answer = parse_answer(result.response_text)
    except ParseError as e:
        pytest.fail(f"Response not parseable: {e}\n\nResponse:\n{result.response_text}")
    assert "analog" in answer and "button_color" in answer
