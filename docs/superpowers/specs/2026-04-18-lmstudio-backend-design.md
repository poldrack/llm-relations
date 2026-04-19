# LMStudio Backend for Benchmark

**Date:** 2026-04-18
**Status:** Approved (pending implementation)

## Goal

Add the ability to run the relational-reasoning benchmark against locally hosted models served by [LMStudio](https://lmstudio.ai/), using LMStudio's Anthropic-compatible `POST /v1/messages` endpoint. The first target model is `google/gemma-3n-e4b` (or whichever Gemma variant is loaded in LMStudio at the time of the run), which is expected to fail the task and serve as a low-capability baseline.

## Why

The benchmark currently only runs against Anthropic API models. Including a small, locally hosted model that is expected to fail provides a useful lower-bound comparison and demonstrates that the benchmark discriminates between capable and weak models.

## Non-goals

- Supporting providers other than Anthropic API and LMStudio.
- Streaming, tool use, or any feature beyond the single `messages.create`-equivalent call already used.
- Auto-discovering the LMStudio model list.
- Hosting/managing the LMStudio process itself.

## Approach

Reuse `ClaudeClient` by parameterizing it with `base_url` and `api_key`. LMStudio (≥ 0.4.1) exposes an Anthropic-compatible `POST /v1/messages` endpoint, so the existing `anthropic` SDK can talk to it by passing `base_url="http://127.0.0.1:1234"` and a dummy API key.

No new dependencies are required.

## Components

### `src/llm_relations/runner/client.py`

`ClaudeClient.__init__` gains two optional parameters:

- `base_url: str | None = None` — forwarded to `Anthropic(...)`. When `None`, the SDK uses its default (Anthropic API).
- `cache_system_prompt: bool = True` — when `True`, the system prompt is sent as a structured block with `cache_control={"type": "ephemeral"}` (current behavior). When `False`, the system prompt is sent as a plain string. LMStudio's compat layer is not guaranteed to honor `cache_control`; the safe default for the LMStudio client is `False`.

Retry logic, `CallResult`, `_TASK_DESCRIPTION`, `_COT_INSTRUCTION`, `_ANSWER_FORMAT`, and `build_system_prompt` are unchanged.

### `src/llm_relations/runner/benchmark.py`

Introduce a `ModelSpec` dataclass:

```python
@dataclass(frozen=True)
class ModelSpec:
    display_name: str   # what appears in SampleRecord.model and summary.csv
    api_model_name: str # what is sent to the SDK as `model=`
    client: ClaudeClient
```

`run_benchmark` signature changes from:

```python
def run_benchmark(..., models: list[str], client: ClaudeClient, ...)
```

to:

```python
def run_benchmark(..., model_specs: list[ModelSpec], ...)
```

The loop iterates `model_specs`, uses `spec.client.call(model=spec.api_model_name, ...)`, and writes `spec.display_name` into `SampleRecord.model`. Everything downstream (per-sample JSON paths, summary aggregation key) continues to use the display name, so existing layout under `results/raw/<prompt_variant>/<model>/<problem_id>/sample_N.json` is preserved with the prefixed name.

### `scripts/run_benchmark.py`

CLI changes:

- `--models` continues to accept a space-separated list. Entries may be:
  - A bare Anthropic model ID (e.g., `claude-opus-4-7`), routed to the Anthropic API client.
  - An entry prefixed with `lmstudio:` (e.g., `lmstudio:google/gemma-3n-e4b`), routed to the LMStudio client. Everything after the prefix is the LMStudio model ID sent in the API call.
- New `--lmstudio-url` flag, default `http://127.0.0.1:1234`.
- A pure helper `build_model_specs(model_args, anthropic_api_key, lmstudio_url) -> list[ModelSpec]` performs the parsing and client construction. It is unit-testable without subprocesses or network.
- The Anthropic client is constructed only if at least one Anthropic model is requested. If only LMStudio models are requested, `ANTHROPIC_API_KEY` is not required and its absence does not raise.
- If at least one Anthropic model is requested and `ANTHROPIC_API_KEY` is unset, `SystemExit` is raised as today.

The `display_name` for an LMStudio entry retains the `lmstudio:` prefix, so summary rows clearly distinguish providers.

## Data flow

```
CLI args
   └── build_model_specs() ──► [ModelSpec, ModelSpec, ...]
                                     │
                                     ▼
                            run_benchmark(model_specs)
                                     │
                              for spec in specs:
                                for problem, sample:
                                  spec.client.call(
                                    model=spec.api_model_name,
                                    user_prompt=problem.prompt_text,
                                    system_prompt=...,
                                  )
                                  → SampleRecord(model=spec.display_name, ...)
                                  → write JSON, accumulate
                              _write_summary(...)
```

## Error handling

- LMStudio is local and unauthenticated — no rate-limit retries are expected to fire, but the existing retry path stays in place. It only triggers on HTTP 429/529 from the SDK.
- Network errors (LMStudio not running, wrong port) propagate as ordinary `anthropic` SDK exceptions and abort the run. The user should start LMStudio before invoking the benchmark.
- If LMStudio rejects the request shape (e.g., refuses `cache_control`, refuses certain fields), we mitigate with `cache_system_prompt=False`. Any further incompatibilities surface during the smoke check; further fallbacks are out of scope for this design and will be handled as bugs if they arise.

## Testing

Test-driven. Tests are written and committed before the implementation.

**`tests/test_runner_client.py` (extend):**
- `Anthropic` is constructed with `base_url` when one is provided to `ClaudeClient`.
- `Anthropic` is constructed without `base_url` when `ClaudeClient` is given none (existing behavior preserved).
- When `cache_system_prompt=False`, the `system=` kwarg passed to `messages.create` is a plain string (not the structured list with `cache_control`).
- When `cache_system_prompt=True` (default), the existing structured `system=[{...cache_control...}]` shape is preserved.

**`tests/test_runner_benchmark.py` (extend / refactor):**
- `run_benchmark` accepts a list of `ModelSpec`s, calls each spec's `client.call(model=spec.api_model_name, ...)`, and writes `spec.display_name` into `SampleRecord.model`.
- `summary.csv` rows contain `display_name` in the `model` column.
- Per-sample JSON files are written under `results/raw/<prompt_variant>/<display_name>/<problem_id>/sample_N.json`.

**`tests/test_run_benchmark_cli.py` (new):**
- `build_model_specs` parses bare names as Anthropic specs (uses the Anthropic client, `display_name == api_model_name`).
- `build_model_specs` parses `lmstudio:` entries as LMStudio specs (`api_model_name` strips the prefix, `display_name` keeps it, client is constructed with the supplied `base_url` and `cache_system_prompt=False`).
- When only LMStudio entries are present, `build_model_specs` succeeds without an `ANTHROPIC_API_KEY`.
- When at least one Anthropic entry is present and the API key is `None`, `build_model_specs` raises `SystemExit`.
- A mixed list yields one shared client per provider (specs share the same `client` instance, not one per spec).

**Optional smoke test (`tests/test_smoke_lmstudio.py`):**
- Marked `@pytest.mark.smoke`, opt-in. Hits a real LMStudio at `127.0.0.1:1234` with one tiny prompt and asserts that the call returns text + non-zero token counts. Skipped unless explicitly requested.

## Out of scope

- Adding non-Anthropic, non-LMStudio providers.
- Configuration files for model lists (kept on the CLI for now).
- Changes to problem generation, scoring, or analysis.
- Performance optimization; LMStudio throughput is whatever the local hardware delivers.
