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
        "--variants",
        nargs="+",
        default=None,
        metavar="VARIANT",
        help=(
            "If provided, only run problems with these variant labels "
            "(e.g. --variants cross_domain baseline). "
            "Known variants: baseline, feature_misleading, adversarial, "
            "cross_domain, scale, control. Default: run all variants present "
            "in the problems directory."
        ),
    )
    parser.add_argument(
        "--prompt-variant",
        default="cot",
        help=(
            "System-prompt variant. One of: cot (default), no_cot, "
            "graphical_model. See src/llm_relations/runner/client.py "
            "for the registry."
        ),
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
        prompt_variant=args.prompt_variant,
        variants=args.variants,
    )
    print(f"Done. Summary at {args.results_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
