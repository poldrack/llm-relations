from __future__ import annotations

from dataclasses import dataclass

from llm_relations.runner.client import ClaudeClient


LMSTUDIO_PREFIX = "lmstudio:"


@dataclass(frozen=True)
class ModelSpec:
    """One row in the benchmark matrix.

    `display_name` is the label written into SampleRecord.model and the
    summary CSV; it also becomes the on-disk directory name for raw
    samples. `api_model_name` is the literal string passed as `model=` to
    the SDK. `client` is the ClaudeClient used for this entry.
    """

    display_name: str
    api_model_name: str
    client: ClaudeClient


def build_model_specs(
    model_args: list[str],
    anthropic_api_key: str | None,
    lmstudio_url: str,
) -> list[ModelSpec]:
    """Parse the CLI --models list into ModelSpecs.

    Bare entries route to the Anthropic API client. Entries prefixed with
    `lmstudio:` route to a single shared ClaudeClient pointed at the
    LMStudio Anthropic-compatible endpoint, with system-prompt caching
    disabled (LMStudio's compat layer is not guaranteed to honor it).
    """
    has_anthropic = any(not m.startswith(LMSTUDIO_PREFIX) for m in model_args)
    has_lmstudio = any(m.startswith(LMSTUDIO_PREFIX) for m in model_args)

    anthropic_client: ClaudeClient | None = None
    if has_anthropic:
        if not anthropic_api_key:
            raise SystemExit("ANTHROPIC_API_KEY is not set")
        anthropic_client = ClaudeClient(api_key=anthropic_api_key)

    lmstudio_client: ClaudeClient | None = None
    if has_lmstudio:
        lmstudio_client = ClaudeClient(
            api_key="lmstudio",
            base_url=lmstudio_url,
            cache_system_prompt=False,
        )

    specs: list[ModelSpec] = []
    for m in model_args:
        if m.startswith(LMSTUDIO_PREFIX):
            assert lmstudio_client is not None
            specs.append(ModelSpec(
                display_name=m,
                api_model_name=m[len(LMSTUDIO_PREFIX):],
                client=lmstudio_client,
            ))
        else:
            assert anthropic_client is not None
            specs.append(ModelSpec(
                display_name=m,
                api_model_name=m,
                client=anthropic_client,
            ))
    return specs
