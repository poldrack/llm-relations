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
        prompt_variant="no_cot",
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
        prompt_variant="no_cot",
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
        model_specs=[spec], n_samples=1, prompt_variant="cot",
    )
    run_benchmark(
        problems_dir=problems_dir, results_dir=results_dir,
        model_specs=[spec], n_samples=1, prompt_variant="no_cot",
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
        prompt_variant="no_cot",
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
        model_specs=[spec], n_samples=2, prompt_variant="cot",
    )
    # Second run: should write sample_2.json and sample_3.json (not overwrite)
    run_benchmark(
        problems_dir=problems_dir, results_dir=results_dir,
        model_specs=[spec], n_samples=2, prompt_variant="cot",
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


def test_run_benchmark_summary_aggregates_across_prior_and_new_samples(
    tmp_path: Path,
):
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
        model_specs=[spec], n_samples=2, prompt_variant="cot",
    )
    run_benchmark(
        problems_dir=problems_dir, results_dir=results_dir,
        model_specs=[spec], n_samples=2, prompt_variant="cot",
    )

    import csv as _csv
    with (results_dir / "summary.csv").open() as f:
        rows = list(_csv.DictReader(f))
    matching = [
        r for r in rows
        if r["prompt_variant"] == "cot"
        and r["model"] == "claude-haiku-4-5-20251001"
        and r["problem_id"] == p.problem_id
    ]
    assert len(matching) == 1, matching
    row = matching[0]
    assert int(row["n_samples"]) == 4, row
    assert int(row["n_correct"]) == 4, row
    assert float(row["accuracy"]) == 1.0, row


def _problem_with_variant(pid: str, variant: str) -> Problem:
    return Problem(
        problem_id=pid,
        variant=variant,
        prompt_text="prompt",
        correct_answer={"analog": "mek", "button_color": "blue"},
        metadata={
            "n_objects": 3,
            "feature_match_answer": {"analog": "zop", "button_color": "blue"},
            "positional_match_answer": {"analog": "quib", "button_color": "green"},
        },
    )


def test_run_benchmark_variants_filter_restricts_to_matching_problems(tmp_path: Path):
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(_problem_with_variant("baseline_00", "baseline"), problems_dir / "baseline_00.json")
    save_problem(_problem_with_variant("cross_domain_00", "cross_domain"), problems_dir / "cross_domain_00.json")
    save_problem(_problem_with_variant("cross_domain_01", "cross_domain"), problems_dir / "cross_domain_01.json")
    save_problem(_problem_with_variant("control_00", "control"), problems_dir / "control_00.json")

    results_dir = tmp_path / "results"
    client = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )

    run_benchmark(
        problems_dir=problems_dir,
        results_dir=results_dir,
        model_specs=[_spec("claude-haiku-4-5-20251001", client)],
        n_samples=1,
        variants=["cross_domain"],
    )

    # Only the two cross_domain problems should have been called.
    assert client.call.call_count == 2
    written = sorted(
        f.parent.name for f in (results_dir / "raw").rglob("sample_*.json")
    )
    assert written == ["cross_domain_00", "cross_domain_01"]


def test_run_benchmark_variants_filter_rejects_unknown_variant(tmp_path: Path):
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(_problem_with_variant("baseline_00", "baseline"), problems_dir / "baseline_00.json")

    import pytest

    with pytest.raises(ValueError, match="Unknown variant"):
        run_benchmark(
            problems_dir=problems_dir,
            results_dir=tmp_path / "results",
            model_specs=[_spec("claude-haiku-4-5-20251001", _client_returning(""))],
            n_samples=1,
            variants=["nonsense"],
        )


def test_run_benchmark_variants_none_means_run_all(tmp_path: Path):
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(_problem_with_variant("baseline_00", "baseline"), problems_dir / "baseline_00.json")
    save_problem(_problem_with_variant("cross_domain_00", "cross_domain"), problems_dir / "cross_domain_00.json")

    client = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )

    run_benchmark(
        problems_dir=problems_dir,
        results_dir=tmp_path / "results",
        model_specs=[_spec("claude-haiku-4-5-20251001", client)],
        n_samples=1,
        # variants omitted
    )

    assert client.call.call_count == 2


def test_run_benchmark_summary_splits_mean_latency_by_correctness(tmp_path: Path):
    """mean_latency_ms_correct and _error are averaged over matching samples only."""
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    # Interleave three correct (latency 100) and two incorrect (latency 400)
    # responses. The client returns each in sequence.
    correct_body = '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    wrong_body = '```json\n{"analog": "zop", "button_color": "blue"}\n```'
    call_results = [
        CallResult(response_text=correct_body, input_tokens=1, output_tokens=1,
                   cache_read_input_tokens=0, latency_ms=100),
        CallResult(response_text=wrong_body, input_tokens=1, output_tokens=1,
                   cache_read_input_tokens=0, latency_ms=400),
        CallResult(response_text=correct_body, input_tokens=1, output_tokens=1,
                   cache_read_input_tokens=0, latency_ms=100),
        CallResult(response_text=wrong_body, input_tokens=1, output_tokens=1,
                   cache_read_input_tokens=0, latency_ms=400),
        CallResult(response_text=correct_body, input_tokens=1, output_tokens=1,
                   cache_read_input_tokens=0, latency_ms=100),
    ]
    client = MagicMock()
    client.call.side_effect = call_results

    run_benchmark(
        problems_dir=problems_dir,
        results_dir=results_dir,
        model_specs=[_spec("claude-haiku-4-5-20251001", client)],
        n_samples=5,
    )

    import csv as _csv
    with (results_dir / "summary.csv").open() as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert int(row["n_samples"]) == 5
    assert int(row["n_correct"]) == 3
    # Overall mean is the weighted avg of all five samples.
    assert float(row["mean_latency_ms"]) == (3 * 100 + 2 * 400) / 5
    # Mean over correct samples only: 100.
    assert float(row["mean_latency_ms_correct"]) == 100.0
    # Mean over error samples only: 400.
    assert float(row["mean_latency_ms_error"]) == 400.0


def test_run_benchmark_summary_latency_columns_empty_when_bucket_empty(tmp_path: Path):
    """If all samples are correct (or all are errors), the empty bucket is blank."""
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

    import csv as _csv
    with (results_dir / "summary.csv").open() as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert int(row["n_correct"]) == 2
    # Correct bucket populated, error bucket empty.
    assert row["mean_latency_ms_correct"] != ""
    assert row["mean_latency_ms_error"] == ""


def test_run_benchmark_sample_index_skips_gaps(tmp_path: Path):
    """Gaps in sample numbering are not filled — new samples land at max+1."""
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    # Pre-create sample_0.json and sample_2.json by hand.
    target_dir = (
        results_dir / "raw" / "cot" / "claude-haiku-4-5-20251001" / p.problem_id
    )
    target_dir.mkdir(parents=True)

    def _stub_record(idx: int) -> dict:
        return {
            "problem_id": p.problem_id,
            "model": "claude-haiku-4-5-20251001",
            "sample": idx,
            "variant": "baseline",
            "prompt_variant": "cot",
            "prompt": "prompt",
            "response_text": "stub",
            "parsed_answer": {"analog": "mek", "button_color": "blue"},
            "correct_answer": {"analog": "mek", "button_color": "blue"},
            "is_correct": True,
            "error_type": None,
            "parse_error": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0,
            "timestamp": "1970-01-01T00:00:00+00:00",
        }

    (target_dir / "sample_0.json").write_text(
        json.dumps(_stub_record(0), indent=2, sort_keys=True)
    )
    (target_dir / "sample_2.json").write_text(
        json.dumps(_stub_record(2), indent=2, sort_keys=True)
    )

    client = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )

    run_benchmark(
        problems_dir=problems_dir, results_dir=results_dir,
        model_specs=[_spec("claude-haiku-4-5-20251001", client)],
        n_samples=1, prompt_variant="cot",
    )

    files = sorted(f.name for f in target_dir.glob("sample_*.json"))
    assert files == [
        "sample_0.json", "sample_2.json", "sample_3.json"
    ], files
    # The newly written record is index 3.
    rec = json.loads((target_dir / "sample_3.json").read_text())
    assert rec["sample"] == 3


def test_run_benchmark_writes_samples_under_graphical_model_variant(tmp_path: Path):
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
        prompt_variant="graphical_model",
    )

    # Directory tree uses the variant name verbatim.
    sample_path = (
        results_dir / "raw" / "graphical_model" / "claude-haiku-4-5-20251001"
        / p.problem_id / "sample_0.json"
    )
    assert sample_path.exists()
    rec = json.loads(sample_path.read_text())
    assert rec["prompt_variant"] == "graphical_model"

    # The graphical-model instruction was sent in the system prompt.
    sys_prompt = client.call.call_args.kwargs["system_prompt"]
    assert "graph" in sys_prompt.lower()

    # Summary CSV carries the variant too.
    summary = (results_dir / "summary.csv").read_text()
    assert "graphical_model" in summary


def test_run_benchmark_rejects_unknown_prompt_variant_before_calling_client(
    tmp_path: Path,
):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    client = _client_returning("unused")

    import pytest
    with pytest.raises(ValueError, match="bogus"):
        run_benchmark(
            problems_dir=problems_dir,
            results_dir=results_dir,
            model_specs=[_spec("claude-haiku-4-5-20251001", client)],
            n_samples=1,
            prompt_variant="bogus",
        )

    # No API calls were made — validation happened first.
    assert client.call.call_count == 0
