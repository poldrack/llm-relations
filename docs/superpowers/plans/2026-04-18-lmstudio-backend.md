# LMStudio Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow the relational-reasoning benchmark to call locally hosted models served by LMStudio's Anthropic-compatible `POST /v1/messages` endpoint, so a low-capability local model (e.g. `google/gemma-3n-e4b`) can be benchmarked alongside the Anthropic API models.

**Architecture:** Reuse `ClaudeClient` by adding `base_url` and `cache_system_prompt` parameters that are forwarded to the `anthropic` SDK. Introduce a `ModelSpec(display_name, api_model_name, client)` so the benchmark loop is provider-agnostic, and a `build_model_specs(...)` CLI helper that parses an `lmstudio:` prefix on `--models` entries.

**Tech Stack:** Python 3.13, `anthropic` SDK (already a dep — supports `base_url`), `pytest`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-18-lmstudio-backend-design.md`

**Files touched:**

- Modify: `src/llm_relations/runner/client.py` (add `base_url`, `cache_system_prompt` params)
- Create: `src/llm_relations/runner/specs.py` (`ModelSpec`, `build_model_specs`, `LMSTUDIO_PREFIX`)
- Modify: `src/llm_relations/runner/benchmark.py` (switch to `ModelSpec`-based loop)
- Modify: `scripts/run_benchmark.py` (CLI uses `build_model_specs`, new `--lmstudio-url` flag)
- Modify: `tests/test_runner_client.py` (cover `base_url` and `cache_system_prompt`)
- Modify: `tests/test_runner_benchmark.py` (switch to `ModelSpec`)
- Create: `tests/test_runner_specs.py` (cover `build_model_specs`)
- Create: `tests/test_smoke_lmstudio.py` (opt-in smoke test against real LMStudio)

---

## Task 1: Add `base_url` parameter to `ClaudeClient`

**Files:**
- Modify: `src/llm_relations/runner/client.py`
- Modify: `tests/test_runner_client.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_runner_client.py`:

```python
def test_client_constructs_anthropic_with_base_url_when_provided(mocker):
    fake_anthropic_cls = mocker.patch("llm_relations.runner.client.Anthropic")
    ClaudeClient(api_key="test-key", base_url="http://127.0.0.1:1234")
    fake_anthropic_cls.assert_called_once_with(
        api_key="test-key", base_url="http://127.0.0.1:1234"
    )


def test_client_constructs_anthropic_without_base_url_when_omitted(mocker):
    fake_anthropic_cls = mocker.patch("llm_relations.runner.client.Anthropic")
    ClaudeClient(api_key="test-key")
    fake_anthropic_cls.assert_called_once_with(api_key="test-key")
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_runner_client.py::test_client_constructs_anthropic_with_base_url_when_provided tests/test_runner_client.py::test_client_constructs_anthropic_without_base_url_when_omitted -v`

Expected: FAIL — `ClaudeClient.__init__()` got an unexpected keyword argument `'base_url'` for the first; the second may pass already (it documents the existing default), which is fine.

- [ ] **Step 3: Implement `base_url` in `ClaudeClient.__init__`**

Replace the `__init__` method in `src/llm_relations/runner/client.py`:

```python
class ClaudeClient:
    def __init__(
        self,
        api_key: str,
        max_retries: int = 5,
        base_delay: float = 2.0,
        base_url: str | None = None,
    ):
        if base_url is None:
            self._client = Anthropic(api_key=api_key)
        else:
            self._client = Anthropic(api_key=api_key, base_url=base_url)
        self._max_retries = max_retries
        self._base_delay = base_delay
```

- [ ] **Step 4: Run the full client test module**

Run: `uv run pytest tests/test_runner_client.py -v`

Expected: PASS — all tests, including the two new ones and all pre-existing ones.

- [ ] **Step 5: Commit**

```bash
git add src/llm_relations/runner/client.py tests/test_runner_client.py
git commit -m "feat: ClaudeClient accepts optional base_url"
```

---

## Task 2: Add `cache_system_prompt` parameter to `ClaudeClient`

**Files:**
- Modify: `src/llm_relations/runner/client.py`
- Modify: `tests/test_runner_client.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_runner_client.py`:

```python
def test_client_call_sends_plain_string_system_when_caching_disabled(mocker):
    fake_anthropic = MagicMock()
    fake_anthropic.messages.create.return_value = _mock_message("ok")
    mocker.patch("llm_relations.runner.client.Anthropic", return_value=fake_anthropic)

    client = ClaudeClient(api_key="test-key", cache_system_prompt=False)
    client.call(model="some-model", user_prompt="Solve.", system_prompt="SYS")

    call_kwargs = fake_anthropic.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "SYS"


def test_client_call_default_still_sends_cached_system_block(mocker):
    fake_anthropic = MagicMock()
    fake_anthropic.messages.create.return_value = _mock_message("ok")
    mocker.patch("llm_relations.runner.client.Anthropic", return_value=fake_anthropic)

    client = ClaudeClient(api_key="test-key")
    client.call(model="some-model", user_prompt="Solve.", system_prompt="SYS")

    call_kwargs = fake_anthropic.messages.create.call_args.kwargs
    assert call_kwargs["system"] == [
        {"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}
    ]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_runner_client.py::test_client_call_sends_plain_string_system_when_caching_disabled tests/test_runner_client.py::test_client_call_default_still_sends_cached_system_block -v`

Expected: FAIL — `cache_system_prompt` is an unexpected keyword argument for the first; the second may pass (documents existing behavior).

- [ ] **Step 3: Implement `cache_system_prompt` in `ClaudeClient`**

Update `__init__` and `call` in `src/llm_relations/runner/client.py`. The full updated class body:

```python
class ClaudeClient:
    def __init__(
        self,
        api_key: str,
        max_retries: int = 5,
        base_delay: float = 2.0,
        base_url: str | None = None,
        cache_system_prompt: bool = True,
    ):
        if base_url is None:
            self._client = Anthropic(api_key=api_key)
        else:
            self._client = Anthropic(api_key=api_key, base_url=base_url)
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._cache_system_prompt = cache_system_prompt

    def call(
        self,
        model: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> CallResult:
        if self._cache_system_prompt:
            system_arg = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_arg = system_prompt

        attempt = 0
        while True:
            start = time.time()
            try:
                msg = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_arg,
                    messages=[{"role": "user", "content": user_prompt}],
                )
            except (RateLimitError, APIStatusError) as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status not in (429, 529):
                    raise
                attempt += 1
                if attempt >= self._max_retries:
                    raise
                _sleep(self._base_delay * (2 ** (attempt - 1)))
                continue

            latency_ms = int((time.time() - start) * 1000)
            text = "".join(
                block.text for block in msg.content if getattr(block, "type", None) == "text"
            )
            return CallResult(
                response_text=text,
                input_tokens=msg.usage.input_tokens,
                output_tokens=msg.usage.output_tokens,
                cache_read_input_tokens=getattr(msg.usage, "cache_read_input_tokens", 0),
                latency_ms=latency_ms,
            )
```

- [ ] **Step 4: Run the full client test module**

Run: `uv run pytest tests/test_runner_client.py -v`

Expected: PASS — all tests including the two new ones and the unchanged retry/argument tests.

- [ ] **Step 5: Commit**

```bash
git add src/llm_relations/runner/client.py tests/test_runner_client.py
git commit -m "feat: ClaudeClient can disable system-prompt cache_control"
```

---

## Task 3: Introduce `ModelSpec` and refactor `run_benchmark`

This task is atomic: changing the `run_benchmark` signature requires updating both its tests and `scripts/run_benchmark.py` in the same commit, otherwise the script will not import.

**Files:**
- Create: `src/llm_relations/runner/specs.py`
- Modify: `src/llm_relations/runner/benchmark.py`
- Modify: `scripts/run_benchmark.py`
- Modify: `tests/test_runner_benchmark.py`

- [ ] **Step 1: Update `tests/test_runner_benchmark.py` to use `ModelSpec`**

Replace the entire contents of `tests/test_runner_benchmark.py` with:

```python
import json
from pathlib import Path
from unittest.mock import MagicMock

from llm_relations.problem import Problem, save_problem
from llm_relations.runner.benchmark import run_benchmark, SampleRecord
from llm_relations.runner.client import CallResult
from llm_relations.runner.specs import ModelSpec


def _problem(pid: str = "baseline_00") -> Problem:
    return Problem(
        problem_id=pid,
        variant="baseline",
        prompt_text="prompt",
        correct_answer={"analog": "mek", "button_color": "blue"},
        metadata={
            "n_objects": 3,
            "feature_match_answer": {"analog": "zop", "button_color": "blue"},
            "positional_match_answer": {"analog": "quib", "button_color": "green"},
        },
    )


def _spec(display_name: str, client, api_model_name: str | None = None) -> ModelSpec:
    return ModelSpec(
        display_name=display_name,
        api_model_name=api_model_name or display_name,
        client=client,
    )


def _client_returning(text: str) -> MagicMock:
    client = MagicMock()
    client.call.return_value = CallResult(
        response_text=text,
        input_tokens=1,
        output_tokens=1,
        cache_read_input_tokens=0,
        latency_ms=1,
    )
    return client


def test_run_benchmark_writes_one_file_per_sample(tmp_path: Path):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")

    results_dir = tmp_path / "results"
    client = _client_returning(
        'Reasoning...\n```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )

    run_benchmark(
        problems_dir=problems_dir,
        results_dir=results_dir,
        model_specs=[_spec("claude-haiku-4-5-20251001", client)],
        n_samples=3,
    )

    sample_files = list((results_dir / "raw").rglob("sample_*.json"))
    assert len(sample_files) == 3
    for f in sample_files:
        rec = json.loads(f.read_text())
        assert rec["is_correct"] is True
        assert rec["error_type"] is None
        assert rec["problem_id"] == "baseline_00"


def test_run_benchmark_records_parse_error_when_answer_missing(tmp_path: Path):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    client = _client_returning("I cannot solve this.")

    run_benchmark(
        problems_dir=problems_dir,
        results_dir=results_dir,
        model_specs=[_spec("claude-haiku-4-5-20251001", client)],
        n_samples=1,
    )

    rec = json.loads(
        (results_dir / "raw" / "cot" / "claude-haiku-4-5-20251001"
         / p.problem_id / "sample_0.json").read_text()
    )
    assert rec["parse_error"] is True
    assert rec["is_correct"] is False
    assert rec["error_type"] == "parse_error"
    assert rec["prompt_variant"] == "cot"


def test_run_benchmark_nests_raw_under_prompt_variant(tmp_path: Path):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    client = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )

    run_benchmark(
        problems_dir=problems_dir,
        results_dir=results_dir,
        model_specs=[_spec("claude-haiku-4-5-20251001", client)],
        n_samples=1,
        use_cot=False,
    )

    path = (results_dir / "raw" / "no_cot" / "claude-haiku-4-5-20251001"
            / p.problem_id / "sample_0.json")
    assert path.exists()
    rec = json.loads(path.read_text())
    assert rec["prompt_variant"] == "no_cot"


def test_run_benchmark_summary_includes_prompt_variant_column(tmp_path: Path):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    client = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )

    run_benchmark(
        problems_dir=problems_dir,
        results_dir=results_dir,
        model_specs=[_spec("claude-haiku-4-5-20251001", client)],
        n_samples=1,
        use_cot=False,
    )

    summary = (results_dir / "summary.csv").read_text()
    header = summary.splitlines()[0]
    assert "prompt_variant" in header
    assert "no_cot" in summary.splitlines()[1]


def test_run_benchmark_summary_merges_existing_rows(tmp_path: Path):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    client = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )
    spec = _spec("claude-haiku-4-5-20251001", client)

    run_benchmark(
        problems_dir=problems_dir, results_dir=results_dir,
        model_specs=[spec], n_samples=1, use_cot=True,
    )
    run_benchmark(
        problems_dir=problems_dir, results_dir=results_dir,
        model_specs=[spec], n_samples=1, use_cot=False,
    )

    import csv as _csv
    with (results_dir / "summary.csv").open() as f:
        rows = list(_csv.DictReader(f))
    variants_in_summary = {r["prompt_variant"] for r in rows}
    assert variants_in_summary == {"cot", "no_cot"}


def test_run_benchmark_passes_no_cot_system_prompt_to_client(tmp_path: Path):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    client = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )

    run_benchmark(
        problems_dir=problems_dir,
        results_dir=results_dir,
        model_specs=[_spec("claude-haiku-4-5-20251001", client)],
        n_samples=1,
        use_cot=False,
    )

    call_kwargs = client.call.call_args.kwargs
    assert "Think step by step" not in call_kwargs["system_prompt"]
    assert "```json" in call_kwargs["system_prompt"]


def test_run_benchmark_defaults_to_cot_system_prompt(tmp_path: Path):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    client = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )

    run_benchmark(
        problems_dir=problems_dir,
        results_dir=results_dir,
        model_specs=[_spec("claude-haiku-4-5-20251001", client)],
        n_samples=1,
    )

    call_kwargs = client.call.call_args.kwargs
    assert "Think step by step" in call_kwargs["system_prompt"]


def test_run_benchmark_writes_summary_csv(tmp_path: Path):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    client = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )

    run_benchmark(
        problems_dir=problems_dir,
        results_dir=results_dir,
        model_specs=[_spec("claude-haiku-4-5-20251001", client)],
        n_samples=2,
    )

    summary = (results_dir / "summary.csv").read_text()
    header = summary.splitlines()[0]
    for col in ["model", "variant", "problem_id", "n_samples", "n_correct", "accuracy"]:
        assert col in header


def test_run_benchmark_uses_api_model_name_for_call_and_display_name_for_record(tmp_path: Path):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    client = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )

    run_benchmark(
        problems_dir=problems_dir,
        results_dir=results_dir,
        model_specs=[_spec(
            display_name="lmstudio:google/gemma-3n-e4b",
            client=client,
            api_model_name="google/gemma-3n-e4b",
        )],
        n_samples=1,
    )

    # The SDK was called with the bare api_model_name.
    assert client.call.call_args.kwargs["model"] == "google/gemma-3n-e4b"
    # The on-disk record + path use the prefixed display_name.
    sample_path = (results_dir / "raw" / "cot" / "lmstudio:google/gemma-3n-e4b"
                   / p.problem_id / "sample_0.json")
    assert sample_path.exists()
    rec = json.loads(sample_path.read_text())
    assert rec["model"] == "lmstudio:google/gemma-3n-e4b"
    summary = (results_dir / "summary.csv").read_text()
    assert "lmstudio:google/gemma-3n-e4b" in summary


def test_run_benchmark_uses_each_specs_own_client(tmp_path: Path):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    client_a = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )
    client_b = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )

    run_benchmark(
        problems_dir=problems_dir,
        results_dir=results_dir,
        model_specs=[
            _spec("alpha", client_a),
            _spec("beta", client_b),
        ],
        n_samples=1,
    )

    assert client_a.call.call_count == 1
    assert client_b.call.call_count == 1
    assert client_a.call.call_args.kwargs["model"] == "alpha"
    assert client_b.call.call_args.kwargs["model"] == "beta"
```

- [ ] **Step 2: Run the benchmark tests to verify they fail**

Run: `uv run pytest tests/test_runner_benchmark.py -v`

Expected: FAIL — `cannot import name 'ModelSpec' from 'llm_relations.runner.specs'` (the module does not exist yet) and `run_benchmark` does not accept `model_specs=`.

- [ ] **Step 3: Create `src/llm_relations/runner/specs.py`**

Write `src/llm_relations/runner/specs.py` (new file):

```python
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
```

- [ ] **Step 4: Refactor `src/llm_relations/runner/benchmark.py`**

Replace the contents of `src/llm_relations/runner/benchmark.py` with:

```python
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from llm_relations.parser import parse_answer, ParseError
from llm_relations.problem import Problem, load_problem
from llm_relations.runner.client import build_system_prompt
from llm_relations.runner.specs import ModelSpec
from llm_relations.scorer import score_answer


@dataclass(frozen=True)
class SampleRecord:
    problem_id: str
    model: str
    sample: int
    variant: str
    prompt_variant: str
    prompt: str
    response_text: str
    parsed_answer: Optional[dict[str, str]]
    correct_answer: dict[str, str]
    is_correct: bool
    error_type: Optional[str]
    parse_error: bool
    input_tokens: int
    output_tokens: int
    latency_ms: int
    timestamp: str


def _run_one_sample(
    spec: ModelSpec,
    sample: int,
    problem: Problem,
    system_prompt: str,
    prompt_variant: str,
) -> SampleRecord:
    result = spec.client.call(
        model=spec.api_model_name,
        user_prompt=problem.prompt_text,
        system_prompt=system_prompt,
    )
    try:
        parsed = parse_answer(result.response_text)
    except ParseError:
        parsed = None
    score = score_answer(problem, parsed)
    return SampleRecord(
        problem_id=problem.problem_id,
        model=spec.display_name,
        sample=sample,
        variant=problem.variant,
        prompt_variant=prompt_variant,
        prompt=problem.prompt_text,
        response_text=result.response_text,
        parsed_answer=parsed,
        correct_answer=problem.correct_answer,
        is_correct=score.is_correct,
        error_type=score.error_type,
        parse_error=parsed is None,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _write_sample(results_dir: Path, record: SampleRecord) -> None:
    out_dir = results_dir / "raw" / record.prompt_variant / record.model / record.problem_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"sample_{record.sample}.json").write_text(
        json.dumps(asdict(record), indent=2, sort_keys=True)
    )


SUMMARY_FIELDNAMES = [
    "prompt_variant", "model", "variant", "problem_id",
    "n_samples", "n_correct", "accuracy",
    "n_feature_match", "n_positional_match", "n_other", "n_parse_error",
    "mean_output_tokens", "mean_latency_ms",
]


def _aggregate(records: list[SampleRecord]) -> dict[tuple[str, str, str], dict]:
    rows: dict[tuple[str, str, str], dict] = {}
    for r in records:
        key = (r.prompt_variant, r.model, r.problem_id)
        if key not in rows:
            rows[key] = {
                "prompt_variant": r.prompt_variant,
                "model": r.model,
                "variant": r.variant,
                "problem_id": r.problem_id,
                "n_samples": 0,
                "n_correct": 0,
                "n_feature_match": 0,
                "n_positional_match": 0,
                "n_other": 0,
                "n_parse_error": 0,
                "total_output_tokens": 0,
                "total_latency_ms": 0,
            }
        agg = rows[key]
        agg["n_samples"] += 1
        agg["n_correct"] += int(r.is_correct)
        if r.error_type == "feature_match":
            agg["n_feature_match"] += 1
        elif r.error_type == "positional_match":
            agg["n_positional_match"] += 1
        elif r.error_type == "parse_error":
            agg["n_parse_error"] += 1
        elif r.error_type == "other":
            agg["n_other"] += 1
        agg["total_output_tokens"] += r.output_tokens
        agg["total_latency_ms"] += r.latency_ms
    return rows


def _agg_to_csv_row(agg: dict) -> dict:
    n = agg["n_samples"]
    return {
        "prompt_variant": agg["prompt_variant"],
        "model": agg["model"],
        "variant": agg["variant"],
        "problem_id": agg["problem_id"],
        "n_samples": n,
        "n_correct": agg["n_correct"],
        "accuracy": agg["n_correct"] / n if n else 0.0,
        "n_feature_match": agg["n_feature_match"],
        "n_positional_match": agg["n_positional_match"],
        "n_other": agg["n_other"],
        "n_parse_error": agg["n_parse_error"],
        "mean_output_tokens": agg["total_output_tokens"] / n if n else 0,
        "mean_latency_ms": agg["total_latency_ms"] / n if n else 0,
    }


def _read_existing_summary(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_summary(results_dir: Path, records: list[SampleRecord]) -> None:
    csv_path = results_dir / "summary.csv"
    new_rows = _aggregate(records)
    preserved = [
        row for row in _read_existing_summary(csv_path)
        if (row.get("prompt_variant", ""), row["model"], row["problem_id"]) not in new_rows
    ]

    results_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in preserved:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDNAMES})
        for agg in new_rows.values():
            writer.writerow(_agg_to_csv_row(agg))


def run_benchmark(
    problems_dir: Path,
    results_dir: Path,
    model_specs: list[ModelSpec],
    n_samples: int,
    use_cot: bool = True,
) -> None:
    problems = [load_problem(f) for f in sorted(problems_dir.glob("*.json"))]
    system_prompt = build_system_prompt(use_cot=use_cot)
    prompt_variant = "cot" if use_cot else "no_cot"
    records: list[SampleRecord] = []
    for spec in model_specs:
        for problem in problems:
            for s in range(n_samples):
                record = _run_one_sample(
                    spec, s, problem, system_prompt, prompt_variant
                )
                _write_sample(results_dir, record)
                records.append(record)
    _write_summary(results_dir, records)
```

- [ ] **Step 5: Update `scripts/run_benchmark.py` to construct `ModelSpec`s**

Replace the contents of `scripts/run_benchmark.py` with:

```python
"""CLI entry point: run the full benchmark."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from llm_relations.runner.benchmark import run_benchmark
from llm_relations.runner.client import ClaudeClient
from llm_relations.runner.specs import ModelSpec


DEFAULT_MODELS = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems-dir", type=Path, default=Path("problems"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument(
        "--no-cot",
        action="store_true",
        help="Omit the 'think step by step' instruction from the system prompt.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    client = ClaudeClient(api_key=api_key)
    specs = [ModelSpec(display_name=m, api_model_name=m, client=client) for m in args.models]

    run_benchmark(
        problems_dir=args.problems_dir,
        results_dir=args.results_dir,
        model_specs=specs,
        n_samples=args.n_samples,
        use_cot=not args.no_cot,
    )
    print(f"Done. Summary at {args.results_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
```

(LMStudio CLI plumbing comes in Task 4 — for this task, the script preserves its current behavior using only Anthropic models.)

- [ ] **Step 6: Run all tests to verify the refactor passes end-to-end**

Run: `uv run pytest -v`

Expected: PASS — all tests, including the rewritten `test_runner_benchmark.py` and untouched generator/parser/scorer tests.

- [ ] **Step 7: Commit**

```bash
git add src/llm_relations/runner/specs.py src/llm_relations/runner/benchmark.py scripts/run_benchmark.py tests/test_runner_benchmark.py
git commit -m "refactor: drive run_benchmark from ModelSpecs"
```

---

## Task 4: Add `build_model_specs` helper and `--lmstudio-url` CLI flag

**Files:**
- Modify: `src/llm_relations/runner/specs.py`
- Modify: `scripts/run_benchmark.py`
- Create: `tests/test_runner_specs.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_runner_specs.py`:

```python
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
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_runner_specs.py -v`

Expected: FAIL — `cannot import name 'build_model_specs' from 'llm_relations.runner.specs'`.

- [ ] **Step 3: Implement `build_model_specs` in `src/llm_relations/runner/specs.py`**

Replace the contents of `src/llm_relations/runner/specs.py` with:

```python
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
```

- [ ] **Step 4: Wire `build_model_specs` and `--lmstudio-url` into the CLI**

Replace the contents of `scripts/run_benchmark.py` with:

```python
"""CLI entry point: run the full benchmark."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from llm_relations.runner.benchmark import run_benchmark
from llm_relations.runner.specs import build_model_specs


DEFAULT_MODELS = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems-dir", type=Path, default=Path("problems"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Model identifiers. Bare names (e.g. 'claude-opus-4-7') hit the "
            "Anthropic API. Entries prefixed with 'lmstudio:' (e.g. "
            "'lmstudio:google/gemma-3n-e4b') hit the LMStudio Anthropic-"
            "compatible endpoint."
        ),
    )
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument(
        "--no-cot",
        action="store_true",
        help="Omit the 'think step by step' instruction from the system prompt.",
    )
    parser.add_argument(
        "--lmstudio-url",
        default="http://127.0.0.1:1234",
        help="Base URL for LMStudio's Anthropic-compatible /v1/messages endpoint.",
    )
    args = parser.parse_args()

    specs = build_model_specs(
        args.models,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        lmstudio_url=args.lmstudio_url,
    )

    run_benchmark(
        problems_dir=args.problems_dir,
        results_dir=args.results_dir,
        model_specs=specs,
        n_samples=args.n_samples,
        use_cot=not args.no_cot,
    )
    print(f"Done. Summary at {args.results_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run all tests to verify the build_model_specs path works end-to-end**

Run: `uv run pytest -v`

Expected: PASS — all tests, including the new `tests/test_runner_specs.py` and unchanged tests.

- [ ] **Step 6: Commit**

```bash
git add src/llm_relations/runner/specs.py scripts/run_benchmark.py tests/test_runner_specs.py
git commit -m "feat: route lmstudio:* models to local Anthropic-compatible endpoint"
```

---

## Task 5: Add opt-in smoke test for LMStudio

**Files:**
- Create: `tests/test_smoke_lmstudio.py`

- [ ] **Step 1: Create the smoke test**

Write `tests/test_smoke_lmstudio.py` (new file):

```python
"""Opt-in smoke test that hits a real LMStudio server.

Run with: uv run pytest -m smoke -v

Skipped unless the smoke marker is selected AND a model is loaded in
LMStudio at the configured URL. Configure with environment variables:
- LMSTUDIO_URL (default: http://127.0.0.1:1234)
- LMSTUDIO_MODEL (default: google/gemma-3n-e4b)
"""
import os

import pytest

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
    except Exception as e:
        pytest.skip(f"LMStudio not reachable at {base_url} ({type(e).__name__}: {e})")

    assert result.output_tokens > 0
    assert isinstance(result.response_text, str)
    assert len(result.response_text) > 0
```

- [ ] **Step 2: Verify the smoke test is collected but skipped by default**

Run: `uv run pytest -v`

Expected: PASS — the smoke test does not run because the `smoke` marker is not selected. It does not affect the rest of the suite.

- [ ] **Step 3: (Optional, manual) verify against a running LMStudio**

If LMStudio is running with a model loaded, run:
`uv run pytest -m smoke tests/test_smoke_lmstudio.py -v`

Expected: PASS, or `SKIPPED` with a clear reason if LMStudio is unreachable.

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke_lmstudio.py
git commit -m "test: opt-in smoke test for LMStudio backend"
```

---

## Self-review

**Spec coverage:**

- `ClaudeClient` `base_url` parameter → Task 1.
- `ClaudeClient` `cache_system_prompt` parameter → Task 2.
- `ModelSpec` dataclass → Task 3.
- `run_benchmark` signature change → Task 3.
- `display_name` written into `SampleRecord.model` and per-sample paths → Task 3 (Step 1 test `test_run_benchmark_uses_api_model_name_for_call_and_display_name_for_record`).
- `build_model_specs` helper, `lmstudio:` prefix parsing → Task 4.
- `--lmstudio-url` CLI flag (default `http://127.0.0.1:1234`) → Task 4.
- `ANTHROPIC_API_KEY` optional when only LMStudio models are requested → Task 4 (`test_lmstudio_only_does_not_require_anthropic_key`).
- `ANTHROPIC_API_KEY` still required when at least one Anthropic model is requested → Task 4 (`test_missing_anthropic_key_raises_when_anthropic_model_requested`).
- One shared client per provider → Task 4 (`test_mixed_list_shares_one_client_per_provider`).
- Optional smoke test → Task 5.

All spec items covered.

**Placeholder scan:** No "TBD", "TODO", "implement later". All steps contain executable code or commands.

**Type/name consistency:** `ModelSpec(display_name, api_model_name, client)` is referenced consistently across Tasks 3 and 4 (tests, `specs.py`, `benchmark.py`, `scripts/run_benchmark.py`). `LMSTUDIO_PREFIX = "lmstudio:"` is referenced consistently. `build_model_specs(model_args, anthropic_api_key, lmstudio_url)` signature is identical across the helper, its tests, and the CLI call site.
