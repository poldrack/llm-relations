from unittest.mock import MagicMock

import pytest

from llm_relations.runner.client import (
    ClaudeClient,
    CallResult,
    SYSTEM_PROMPT,
    build_system_prompt,
)


def _mock_message(text: str, input_tokens: int = 100, output_tokens: int = 200) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text, type="text")]
    msg.usage.input_tokens = input_tokens
    msg.usage.output_tokens = output_tokens
    msg.usage.cache_creation_input_tokens = 0
    msg.usage.cache_read_input_tokens = 0
    return msg


def test_client_calls_messages_create_with_expected_arguments(mocker):
    fake_anthropic = MagicMock()
    fake_anthropic.messages.create.return_value = _mock_message("hello")
    mocker.patch("llm_relations.runner.client.Anthropic", return_value=fake_anthropic)

    client = ClaudeClient(api_key="test-key")
    result = client.call(model="claude-haiku-4-5-20251001", user_prompt="Solve this.")

    assert isinstance(result, CallResult)
    assert result.response_text == "hello"
    assert result.input_tokens == 100
    assert result.output_tokens == 200

    call_kwargs = fake_anthropic.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
    assert call_kwargs["max_tokens"] == 4096
    assert call_kwargs["temperature"] == 1.0
    # System prompt uses cache_control
    assert call_kwargs["system"][0]["text"] == SYSTEM_PROMPT
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call_kwargs["messages"] == [{"role": "user", "content": "Solve this."}]


def test_build_system_prompt_with_cot_includes_think_step_by_step():
    prompt = build_system_prompt(use_cot=True)
    assert "Think step by step" in prompt
    # Still includes the JSON format instruction.
    assert "```json" in prompt


def test_build_system_prompt_without_cot_omits_think_step_by_step():
    prompt = build_system_prompt(use_cot=False)
    assert "Think step by step" not in prompt
    assert "step by step" not in prompt.lower()
    # Still includes the JSON format instruction.
    assert "```json" in prompt


def test_default_system_prompt_matches_cot_variant():
    assert SYSTEM_PROMPT == build_system_prompt(use_cot=True)


def test_client_call_uses_provided_system_prompt(mocker):
    fake_anthropic = MagicMock()
    fake_anthropic.messages.create.return_value = _mock_message("hi")
    mocker.patch("llm_relations.runner.client.Anthropic", return_value=fake_anthropic)

    custom_prompt = "CUSTOM SYSTEM PROMPT"
    client = ClaudeClient(api_key="test-key")
    client.call(
        model="claude-haiku-4-5-20251001",
        user_prompt="Solve.",
        system_prompt=custom_prompt,
    )

    call_kwargs = fake_anthropic.messages.create.call_args.kwargs
    assert call_kwargs["system"][0]["text"] == custom_prompt


def test_client_retries_on_rate_limit(mocker):
    from anthropic import RateLimitError

    fake_anthropic = MagicMock()
    # Fail twice with rate limit, then succeed
    err = RateLimitError(
        message="rate limited", response=MagicMock(status_code=429), body=None
    )
    fake_anthropic.messages.create.side_effect = [err, err, _mock_message("ok")]
    mocker.patch("llm_relations.runner.client.Anthropic", return_value=fake_anthropic)
    # Patch the sleep-between-retries to avoid slow tests
    mocker.patch("llm_relations.runner.client._sleep", return_value=None)

    client = ClaudeClient(api_key="test-key")
    result = client.call(model="claude-haiku-4-5-20251001", user_prompt="Solve.")

    assert result.response_text == "ok"
    assert fake_anthropic.messages.create.call_count == 3


def test_client_retries_then_fails_after_max_attempts(mocker):
    from anthropic import APIStatusError

    fake_anthropic = MagicMock()
    err = APIStatusError(
        message="overloaded", response=MagicMock(status_code=529), body=None
    )
    fake_anthropic.messages.create.side_effect = err
    mocker.patch("llm_relations.runner.client.Anthropic", return_value=fake_anthropic)
    mocker.patch("llm_relations.runner.client._sleep", return_value=None)

    client = ClaudeClient(api_key="test-key", max_retries=3)
    with pytest.raises(APIStatusError):
        client.call(model="claude-haiku-4-5-20251001", user_prompt="Solve.")
    assert fake_anthropic.messages.create.call_count == 3
