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
