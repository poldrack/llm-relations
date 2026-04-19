import pytest

from llm_relations.runner.client import ClaudeClient
from llm_relations.runner.specs import (
    LMSTUDIO_PREFIX,
    ModelSpec,
    build_model_specs,
)


def test_lmstudio_prefix_constant():
    assert LMSTUDIO_PREFIX == "lmstudio:"


def test_bare_model_id_routes_to_anthropic_client():
    specs = build_model_specs(
        ["claude-opus-4-7"],
        anthropic_api_key="key",
        lmstudio_url="http://127.0.0.1:1234",
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.display_name == "claude-opus-4-7"
    assert spec.api_model_name == "claude-opus-4-7"
    assert isinstance(spec.client, ClaudeClient)


def test_lmstudio_prefixed_entry_strips_prefix_for_api_call():
    specs = build_model_specs(
        ["lmstudio:google/gemma-3n-e4b"],
        anthropic_api_key=None,
        lmstudio_url="http://127.0.0.1:1234",
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.display_name == "lmstudio:google/gemma-3n-e4b"
    assert spec.api_model_name == "google/gemma-3n-e4b"
    assert isinstance(spec.client, ClaudeClient)


def test_lmstudio_only_does_not_require_anthropic_key():
    # Should not raise.
    build_model_specs(
        ["lmstudio:google/gemma-3n-e4b"],
        anthropic_api_key=None,
        lmstudio_url="http://127.0.0.1:1234",
    )


def test_missing_anthropic_key_raises_when_anthropic_model_requested():
    with pytest.raises(SystemExit):
        build_model_specs(
            ["claude-opus-4-7"],
            anthropic_api_key=None,
            lmstudio_url="http://127.0.0.1:1234",
        )


def test_mixed_list_shares_one_client_per_provider():
    specs = build_model_specs(
        ["claude-opus-4-7", "claude-haiku-4-5-20251001",
         "lmstudio:google/gemma-3n-e4b", "lmstudio:other/model"],
        anthropic_api_key="key",
        lmstudio_url="http://127.0.0.1:1234",
    )
    anthropic_clients = {id(s.client) for s in specs if not s.display_name.startswith("lmstudio:")}
    lmstudio_clients = {id(s.client) for s in specs if s.display_name.startswith("lmstudio:")}
    assert len(anthropic_clients) == 1
    assert len(lmstudio_clients) == 1
    # And the two providers must use distinct client instances.
    assert anthropic_clients.isdisjoint(lmstudio_clients)


def test_lmstudio_client_constructed_with_base_url_and_no_cache(mocker):
    fake_anthropic_cls = mocker.patch("llm_relations.runner.client.Anthropic")
    build_model_specs(
        ["lmstudio:google/gemma-3n-e4b"],
        anthropic_api_key=None,
        lmstudio_url="http://example:9999",
    )
    fake_anthropic_cls.assert_called_once_with(
        api_key="lmstudio", base_url="http://example:9999"
    )


def test_lmstudio_client_disables_system_prompt_caching(mocker):
    fake_anthropic = mocker.MagicMock()
    fake_anthropic.messages.create.return_value = mocker.MagicMock(
        content=[mocker.MagicMock(text="hi", type="text")],
        usage=mocker.MagicMock(
            input_tokens=1, output_tokens=1, cache_read_input_tokens=0
        ),
    )
    mocker.patch("llm_relations.runner.client.Anthropic", return_value=fake_anthropic)
    specs = build_model_specs(
        ["lmstudio:google/gemma-3n-e4b"],
        anthropic_api_key=None,
        lmstudio_url="http://example:9999",
    )
    specs[0].client.call(
        model="google/gemma-3n-e4b", user_prompt="hello", system_prompt="SYS"
    )
    sent = fake_anthropic.messages.create.call_args.kwargs["system"]
    assert sent == "SYS"
