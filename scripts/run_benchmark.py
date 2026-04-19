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
