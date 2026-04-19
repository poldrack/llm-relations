"""Opt-in smoke test that hits a real LMStudio server.

Run with: uv run pytest -m smoke -v

Skipped unless the smoke marker is selected AND a model is loaded in
LMStudio at the configured URL. Configure with environment variables:
- LMSTUDIO_URL (default: http://127.0.0.1:1234)
- LMSTUDIO_MODEL (default: google/gemma-3n-e4b)
"""
import os

import pytest
from anthropic import APIConnectionError

from llm_relations.runner.client import ClaudeClient


@pytest.mark.smoke
def test_real_lmstudio_call_returns_response():
    base_url = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234")
    model = os.environ.get("LMSTUDIO_MODEL", "google/gemma-3n-e4b")

    client = ClaudeClient(
        api_key="lmstudio",
        base_url=base_url,
        cache_system_prompt=False,
    )
    try:
        result = client.call(
            model=model,
            user_prompt="Say the single word: hello",
            max_tokens=32,
        )
    except APIConnectionError as e:
        pytest.skip(f"LMStudio not reachable at {base_url} ({type(e).__name__}: {e})")

    assert result.output_tokens > 0
    assert isinstance(result.response_text, str)
    assert len(result.response_text) > 0
