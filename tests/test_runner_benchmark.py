import json
from pathlib import Path
from unittest.mock import MagicMock

from llm_relations.problem import Problem, save_problem
from llm_relations.runner.benchmark import run_benchmark, SampleRecord
from llm_relations.runner.client import CallResult


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


def test_run_benchmark_writes_one_file_per_sample(tmp_path: Path):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")

    results_dir = tmp_path / "results"

    client = MagicMock()
    client.call.return_value = CallResult(
        response_text='Reasoning...\n```json\n{"analog": "mek", "button_color": "blue"}\n```',
        input_tokens=500,
        output_tokens=200,
        cache_read_input_tokens=0,
        latency_ms=1234,
    )

    run_benchmark(
        problems_dir=problems_dir,
        results_dir=results_dir,
        models=["claude-haiku-4-5-20251001"],
        n_samples=3,
        client=client,
    )

    sample_files = list((results_dir / "raw").rglob("sample_*.json"))
    assert len(sample_files) == 3
    # Each file parses cleanly and matches SampleRecord schema
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

    client = MagicMock()
    client.call.return_value = CallResult(
        response_text="I cannot solve this.",
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=0,
        latency_ms=500,
    )

    run_benchmark(
        problems_dir=problems_dir,
        results_dir=results_dir,
        models=["claude-haiku-4-5-20251001"],
        n_samples=1,
        client=client,
    )

    rec = json.loads((results_dir / "raw" / "claude-haiku-4-5-20251001" / p.problem_id / "sample_0.json").read_text())
    assert rec["parse_error"] is True
    assert rec["is_correct"] is False
    assert rec["error_type"] == "parse_error"


def test_run_benchmark_writes_summary_csv(tmp_path: Path):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    client = MagicMock()
    client.call.return_value = CallResult(
        response_text='```json\n{"analog": "mek", "button_color": "blue"}\n```',
        input_tokens=500,
        output_tokens=200,
        cache_read_input_tokens=0,
        latency_ms=1234,
    )

    run_benchmark(
        problems_dir=problems_dir,
        results_dir=results_dir,
        models=["claude-haiku-4-5-20251001"],
        n_samples=2,
        client=client,
    )

    summary = (results_dir / "summary.csv").read_text()
    header = summary.splitlines()[0]
    for col in ["model", "variant", "problem_id", "n_samples", "n_correct", "accuracy"]:
        assert col in header
