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
    # The on-disk record + path use the prefixed display_name (slash sanitized for path).
    sample_path = (results_dir / "raw" / "cot" / "lmstudio:google_gemma-3n-e4b"
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


def test_run_benchmark_sanitizes_slashes_in_display_name_for_on_disk_path(tmp_path: Path):
    """display_name like 'lmstudio:google/gemma-3n-e4b' must not create nested dirs."""
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

    # The model directory must be a single, flat segment with the slash replaced.
    model_dir = results_dir / "raw" / "cot" / "lmstudio:google_gemma-3n-e4b"
    assert model_dir.is_dir(), f"expected flat model dir, got: {list((results_dir / 'raw' / 'cot').iterdir())}"
    # No nested 'gemma-3n-e4b' directory under a 'lmstudio:google' parent.
    assert not (results_dir / "raw" / "cot" / "lmstudio:google").exists()
    # The JSON record still carries the unsanitized display_name.
    rec = json.loads((model_dir / p.problem_id / "sample_0.json").read_text())
    assert rec["model"] == "lmstudio:google/gemma-3n-e4b"
    # The summary CSV also carries the unsanitized name.
    summary = (results_dir / "summary.csv").read_text()
    assert "lmstudio:google/gemma-3n-e4b" in summary


def test_run_benchmark_appends_samples_when_prior_exist(tmp_path: Path):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    client = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )
    spec = _spec("claude-haiku-4-5-20251001", client)

    # First run: writes sample_0.json and sample_1.json
    run_benchmark(
        problems_dir=problems_dir, results_dir=results_dir,
        model_specs=[spec], n_samples=2, use_cot=True,
    )
    # Second run: should write sample_2.json and sample_3.json (not overwrite)
    run_benchmark(
        problems_dir=problems_dir, results_dir=results_dir,
        model_specs=[spec], n_samples=2, use_cot=True,
    )

    target_dir = (
        results_dir / "raw" / "cot" / "claude-haiku-4-5-20251001" / p.problem_id
    )
    files = sorted(f.name for f in target_dir.glob("sample_*.json"))
    assert files == [
        "sample_0.json", "sample_1.json", "sample_2.json", "sample_3.json"
    ], files

    # Each record's `sample` field matches its filename index.
    for name in files:
        idx = int(name.removeprefix("sample_").removesuffix(".json"))
        rec = json.loads((target_dir / name).read_text())
        assert rec["sample"] == idx
