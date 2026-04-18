# Relational Reasoning Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a benchmark that runs 25 relational-reasoning problems across Claude Opus 4.7, Sonnet 4.6, and Haiku 4.5, quantifies per-model × per-variant accuracy, and classifies errors against Hummel & Heaton's predicted failure modes.

**Architecture:** Four-layer Python package — variant-specific problem generators → frozen JSON corpus → API runner with retry/caching → scorer + Jupyter notebook report. Generators are seeded for reproducibility; committed corpus is ground truth.

**Tech Stack:** Python 3.13, `uv` for packaging, `anthropic` SDK, `pydantic` for schema, `pytest` for tests, `pandas`/`matplotlib` for analysis notebook.

**Spec:** `docs/superpowers/specs/2026-04-18-relational-benchmark-design.md`

**File layout:**

```
llm-relations/
├── src/llm_relations/
│   ├── __init__.py                     # empty
│   ├── problem.py                      # Problem model + JSON load/save
│   ├── palette.py                      # fixed pools: colors, positions, nonsense words
│   ├── generator/
│   │   ├── __init__.py                 # empty
│   │   ├── baseline.py
│   │   ├── feature_misleading.py
│   │   ├── scale.py
│   │   ├── cross_domain.py
│   │   └── adversarial.py
│   ├── parser.py                       # extract JSON answer from response text
│   ├── scorer.py                       # classify correct / feature / positional / other
│   └── runner/
│       ├── __init__.py                 # empty
│       ├── client.py                   # Anthropic wrapper: retry, caching
│       └── benchmark.py                # orchestration over models × problems × samples
├── scripts/
│   ├── freeze_corpus.py                # instantiate + freeze 25 problems to JSON
│   └── run_benchmark.py                # CLI entry point
├── problems/                           # committed corpus (25 JSON files)
├── analysis/
│   └── report.ipynb
├── results/                            # raw results (gitignored), summary.csv (committed)
├── tests/
│   ├── test_problem.py
│   ├── test_palette.py
│   ├── test_generator_baseline.py
│   ├── test_generator_feature_misleading.py
│   ├── test_generator_scale.py
│   ├── test_generator_cross_domain.py
│   ├── test_generator_adversarial.py
│   ├── test_corpus.py
│   ├── test_parser.py
│   ├── test_scorer.py
│   ├── test_runner_client.py
│   └── test_smoke_api.py               # opt-in, real API
└── pyproject.toml
```

---

## Task 1: Project scaffold

**Files:**
- Modify: `pyproject.toml`
- Create: `.gitignore`, `src/llm_relations/__init__.py`, `src/llm_relations/generator/__init__.py`, `src/llm_relations/runner/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Update pyproject.toml with dependencies and pytest config**

Replace `pyproject.toml` with:

```toml
[project]
name = "llm-relations"
version = "0.1.0"
description = "Relational reasoning benchmark for Claude models"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "anthropic>=0.39.0",
    "pydantic>=2.9.0",
    "tenacity>=9.0.0",
    "pandas>=2.2.0",
    "matplotlib>=3.9.0",
]

[dependency-groups]
dev = [
    "pytest>=8.3.0",
    "pytest-mock>=3.14.0",
    "jupyter>=1.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/llm_relations"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
    "smoke: hits the real Anthropic API (opt-in)",
]
```

- [ ] **Step 2: Create empty __init__.py files**

The project rule forbids code in `__init__.py`. Create four empty files:

```bash
mkdir -p src/llm_relations/generator src/llm_relations/runner tests problems results analysis scripts
touch src/llm_relations/__init__.py src/llm_relations/generator/__init__.py src/llm_relations/runner/__init__.py tests/__init__.py
```

- [ ] **Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.pytest_cache/
.venv/
.python-version
results/raw/
.ipynb_checkpoints/
```

Note: `results/raw/` is gitignored; `results/summary.csv` is committed.

- [ ] **Step 4: Install dependencies**

Run: `uv sync`
Expected: creates `.venv/`, installs packages, no errors.

- [ ] **Step 5: Smoke-check pytest**

Run: `uv run pytest --collect-only`
Expected: "collected 0 items" — no errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/ tests/ problems/ analysis/ scripts/
git commit -m "chore: scaffold project structure and dependencies"
```

---

## Task 2: Problem dataclass and JSON serialization

**Files:**
- Create: `src/llm_relations/problem.py`
- Create: `tests/test_problem.py`

- [ ] **Step 1: Write the failing test**

`tests/test_problem.py`:

```python
import json
from pathlib import Path

from llm_relations.problem import Problem, load_problem, save_problem


def test_problem_constructs_with_required_fields():
    p = Problem(
        problem_id="baseline_00",
        variant="baseline",
        prompt_text="...",
        correct_answer={"analog": "mek", "button_color": "blue"},
        metadata={"n_objects": 3, "seed": 42},
    )
    assert p.variant == "baseline"
    assert p.correct_answer["analog"] == "mek"


def test_problem_round_trips_through_json(tmp_path: Path):
    p = Problem(
        problem_id="baseline_00",
        variant="baseline",
        prompt_text="prompt",
        correct_answer={"analog": "mek", "button_color": "blue"},
        metadata={
            "n_objects": 3,
            "seed": 42,
            "feature_match_answer": {"analog": "zop", "button_color": "blue"},
            "positional_match_answer": {"analog": "zop", "button_color": "green"},
        },
    )
    path = tmp_path / "p.json"
    save_problem(p, path)
    loaded = load_problem(path)
    assert loaded == p


def test_problem_rejects_missing_correct_answer_keys():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Problem(
            problem_id="x",
            variant="baseline",
            prompt_text="p",
            correct_answer={"analog": "mek"},  # missing button_color
            metadata={},
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_problem.py -v`
Expected: ImportError — `llm_relations.problem` does not exist.

- [ ] **Step 3: Implement Problem**

`src/llm_relations/problem.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Answer(BaseModel):
    analog: str
    button_color: str


class Problem(BaseModel):
    problem_id: str
    variant: str
    prompt_text: str
    correct_answer: dict[str, str]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("correct_answer")
    @classmethod
    def _check_answer_keys(cls, v: dict[str, str]) -> dict[str, str]:
        if set(v.keys()) != {"analog", "button_color"}:
            raise ValueError("correct_answer must have exactly keys: analog, button_color")
        return v


def save_problem(problem: Problem, path: Path) -> None:
    path.write_text(json.dumps(problem.model_dump(), indent=2, sort_keys=True))


def load_problem(path: Path) -> Problem:
    return Problem.model_validate_json(path.read_text())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_problem.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_relations/problem.py tests/test_problem.py
git commit -m "feat: add Problem model with JSON persistence"
```

---

## Task 3: Palette module (fixed pools)

**Files:**
- Create: `src/llm_relations/palette.py`
- Create: `tests/test_palette.py`

- [ ] **Step 1: Write the failing test**

`tests/test_palette.py`:

```python
from llm_relations.palette import (
    COLORS,
    POSITIONS,
    NONSENSE_WORDS,
    draw_nonsense_words,
    draw_colors,
)


def test_pools_are_nonempty_and_unique():
    assert len(COLORS) >= 5 and len(set(COLORS)) == len(COLORS)
    assert len(POSITIONS) >= 5 and len(set(POSITIONS)) == len(POSITIONS)
    assert len(NONSENSE_WORDS) >= 50 and len(set(NONSENSE_WORDS)) == len(NONSENSE_WORDS)


def test_draw_nonsense_words_is_seeded_and_unique():
    import random

    rng = random.Random(42)
    a = draw_nonsense_words(rng, n=6)
    rng2 = random.Random(42)
    b = draw_nonsense_words(rng2, n=6)
    assert a == b
    assert len(a) == 6 and len(set(a)) == 6


def test_draw_colors_returns_requested_count():
    import random

    rng = random.Random(0)
    colors = draw_colors(rng, n=3)
    assert len(colors) == 3
    assert all(c in COLORS for c in colors)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_palette.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement palette**

`src/llm_relations/palette.py`:

```python
from __future__ import annotations

import random

COLORS: list[str] = ["blue", "red", "green", "yellow", "purple"]

POSITIONS: list[str] = ["top", "bottom", "side", "front", "back"]

NONSENSE_WORDS: list[str] = [
    "florp", "greeble", "wix", "zop", "quib", "mek", "tarn", "blix", "drog", "pell",
    "krink", "vorp", "snig", "thorp", "glim", "rask", "plon", "twix", "vask", "grup",
    "sploof", "zink", "brall", "muk", "flen", "jorp", "skiv", "trob", "yup", "nop",
    "wren", "clop", "fip", "hask", "lurn", "moob", "nurp", "oxim", "prit", "quan",
    "rast", "scob", "tunk", "ulmp", "vro", "woz", "xav", "yib", "zant", "arn",
    "bep", "corp", "dask", "emp", "fro", "gox", "hint", "ikk", "jax", "klon",
]


def draw_nonsense_words(rng: random.Random, n: int) -> list[str]:
    return rng.sample(NONSENSE_WORDS, n)


def draw_colors(rng: random.Random, n: int) -> list[str]:
    return rng.sample(COLORS, n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_palette.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_relations/palette.py tests/test_palette.py
git commit -m "feat: add fixed palette of colors, positions, and nonsense words"
```

---

## Task 4: Baseline generator

**Files:**
- Create: `src/llm_relations/generator/baseline.py`
- Create: `tests/test_generator_baseline.py`

Design:
- Memory scenario: 3 memory objects `m0, m1, m2` with relations `m0 underneath m1` and `m1 to-the-left-of m2`. Memory target is always `m0`, activated by its *target-color* button on *top*.
- Perception scenario: 3 perception objects `p_a, p_b, p_c`. Listed in a scrambled order so the *positional* cue (first listed) and the *correct* cue (matches by relational role) diverge. Exactly one perception object preserves the target's button configuration (correct analog). One other object carries a prominent button of the target color but on the wrong position (feature-match distractor).
- `correct_slot_index` picks which of the 3 *listed* perception objects is the correct analog (0, 1, or 2).

- [ ] **Step 1: Write the failing test**

`tests/test_generator_baseline.py`:

```python
from llm_relations.generator.baseline import generate_baseline


def test_baseline_returns_well_formed_problem():
    p = generate_baseline(seed=1, index=0, correct_slot_index=2, feature_distractor_slot=0)
    assert p.variant == "baseline"
    assert p.problem_id == "baseline_00"
    assert p.metadata["n_objects"] == 3
    assert p.correct_answer["analog"] and p.correct_answer["button_color"]
    assert p.metadata["feature_match_answer"]["analog"] != p.correct_answer["analog"]
    assert p.metadata["positional_match_answer"]["analog"] != p.correct_answer["analog"]


def test_baseline_is_seed_reproducible():
    a = generate_baseline(seed=7, index=0, correct_slot_index=1, feature_distractor_slot=2)
    b = generate_baseline(seed=7, index=0, correct_slot_index=1, feature_distractor_slot=2)
    assert a == b


def test_baseline_correct_analog_appears_in_prompt_with_target_button():
    p = generate_baseline(seed=3, index=0, correct_slot_index=0, feature_distractor_slot=1)
    analog = p.correct_answer["analog"]
    color = p.correct_answer["button_color"]
    assert analog in p.prompt_text
    # The correct analog should have the target color on top
    assert f"The {analog} has a {color} button on top" in p.prompt_text


def test_baseline_feature_distractor_has_target_color_button():
    p = generate_baseline(seed=5, index=0, correct_slot_index=2, feature_distractor_slot=0)
    distractor = p.metadata["feature_match_answer"]["analog"]
    color = p.correct_answer["button_color"]
    # Find the distractor's button-description sentence and check the target color is in it.
    marker = f"The {distractor} has"
    line_start = p.prompt_text.index(marker)
    line_end = p.prompt_text.index(".", line_start)
    assert color in p.prompt_text[line_start:line_end]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generator_baseline.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement baseline generator**

`src/llm_relations/generator/baseline.py`:

```python
from __future__ import annotations

import random

from llm_relations.problem import Problem
from llm_relations.palette import draw_nonsense_words, draw_colors


_INSTRUCTION = (
    "I'm going to describe two scenarios. In the memory scenario, a novel object called "
    "a {target_memory} has a property: it can be activated by pressing one of its buttons. "
    "Your job is to figure out which object in the perception scenario is the {target_memory}-analog, "
    "and therefore which button on it can be activated."
)


def _describe_object(name: str, buttons: list[tuple[str, str]]) -> str:
    # buttons: list of (color, position)
    parts = [f"a {c} button on {p}" for c, p in buttons]
    if len(parts) == 1:
        body = parts[0]
    elif len(parts) == 2:
        body = f"{parts[0]} and {parts[1]}"
    else:
        body = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"The {name} has {body}."


def generate_baseline(
    seed: int,
    index: int,
    correct_slot_index: int,
    feature_distractor_slot: int,
) -> Problem:
    assert 0 <= correct_slot_index < 3
    assert 0 <= feature_distractor_slot < 3
    assert correct_slot_index != feature_distractor_slot

    rng = random.Random(seed)
    words = draw_nonsense_words(rng, n=6)
    m0, m1, m2 = words[:3]
    perception_words = words[3:]  # three perception objects by *relational role* (under, middle, right)

    colors = draw_colors(rng, n=3)
    target_color, other1, other2 = colors

    # Memory: m0 is under m1, m1 is left of m2. m0 is target.
    m0_buttons = [(target_color, "top"), (other1, "side"), (other2, "bottom")]
    m1_buttons = [(other1, "top"), (target_color, "bottom")]
    m2_buttons = [(other2, "top"), (target_color, "side")]

    memory_text = " ".join([
        f"Memory scenario: There are three objects on a table: a {m0}, a {m1}, and a {m2}.",
        f"The {m0} is underneath the {m1}.",
        f"The {m1} is to-the-left-of the {m2}.",
        _describe_object(m0, m0_buttons),
        _describe_object(m1, m1_buttons),
        _describe_object(m2, m2_buttons),
        f"Pressing the {target_color} button on the {m0} activates it.",
    ])

    # Perception: same relational structure. perception_words[0] is under perception_words[1],
    # perception_words[1] is left of perception_words[2]. But list them in a scrambled order.
    # The correct analog of m0 is perception_words[0] (same relational role).
    correct_analog = perception_words[0]

    # Decide list order so that correct_analog appears at correct_slot_index in the list.
    list_order = [None, None, None]
    list_order[correct_slot_index] = correct_analog
    remaining = [perception_words[1], perception_words[2]]
    # feature_distractor_slot places the distractor in some other slot
    distractor = remaining[0]  # we'll designate perception_words[1] as the feature distractor
    list_order[feature_distractor_slot] = distractor
    other_slot = [i for i in range(3) if list_order[i] is None][0]
    list_order[other_slot] = remaining[1]

    # Perception buttons:
    # correct_analog gets the same config as m0 (target_color on top).
    # distractor gets target_color somewhere *not* on top (feature lure).
    # third object gets no target_color button.
    perception_buttons = {
        correct_analog: [(target_color, "top"), (other1, "side"), (other2, "bottom")],
        distractor: [(other1, "top"), (target_color, "side")],
        remaining[1]: [(other2, "top"), (other1, "bottom")],
    }

    perception_text = " ".join([
        f"Perception scenario: There are three objects on a shelf: "
        f"a {list_order[0]}, a {list_order[1]}, and a {list_order[2]}.",
        f"The {perception_words[0]} is underneath the {perception_words[1]}.",
        f"The {perception_words[1]} is to-the-left-of the {perception_words[2]}.",
        _describe_object(list_order[0], perception_buttons[list_order[0]]),
        _describe_object(list_order[1], perception_buttons[list_order[1]]),
        _describe_object(list_order[2], perception_buttons[list_order[2]]),
    ])

    question = f"Which object in the perception scenario is the {m0}-analog, and which button activates it?"

    prompt = "\n\n".join([
        _INSTRUCTION.format(target_memory=m0),
        memory_text,
        perception_text,
        question,
    ])

    # Positional match: the first-listed perception object.
    positional_analog = list_order[0]
    # Its "target" button color is whatever that object has on top.
    positional_button = next(c for c, p in perception_buttons[positional_analog] if p == "top")

    return Problem(
        problem_id=f"baseline_{index:02d}",
        variant="baseline",
        prompt_text=prompt,
        correct_answer={"analog": correct_analog, "button_color": target_color},
        metadata={
            "n_objects": 3,
            "seed": seed,
            "correct_slot_index": correct_slot_index,
            "feature_distractor_slot": feature_distractor_slot,
            "feature_match_answer": {"analog": distractor, "button_color": target_color},
            "positional_match_answer": {"analog": positional_analog, "button_color": positional_button},
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generator_baseline.py -v`
Expected: 4 passed. If a test fails, inspect whether the string-match expectation matches the generated text exactly — fix the generator, not the test.

- [ ] **Step 5: Commit**

```bash
git add src/llm_relations/generator/baseline.py tests/test_generator_baseline.py
git commit -m "feat: add baseline variant generator"
```

---

## Task 5: Feature-misleading generator

**Files:**
- Create: `src/llm_relations/generator/feature_misleading.py`
- Create: `tests/test_generator_feature_misleading.py`

Design: Same as baseline structure, but the feature-match distractor gets **two** buttons of the target color (one on top, one elsewhere) to strengthen the feature lure.

- [ ] **Step 1: Write the failing test**

`tests/test_generator_feature_misleading.py`:

```python
from llm_relations.generator.feature_misleading import generate_feature_misleading


def test_feature_misleading_returns_well_formed_problem():
    p = generate_feature_misleading(seed=2, index=0, correct_slot_index=1, feature_distractor_slot=0)
    assert p.variant == "feature_misleading"
    assert p.problem_id == "feature_misleading_00"
    assert p.metadata["n_objects"] == 3


def test_feature_misleading_distractor_has_two_target_color_buttons():
    p = generate_feature_misleading(seed=9, index=0, correct_slot_index=2, feature_distractor_slot=0)
    color = p.correct_answer["button_color"]
    distractor = p.metadata["feature_match_answer"]["analog"]
    # Find the distractor's button-description sentence; target color should appear >=2 times.
    marker = f"The {distractor} has"
    line_start = p.prompt_text.index(marker)
    line_end = p.prompt_text.index(".", line_start)
    segment = p.prompt_text[line_start:line_end]
    assert segment.count(color) >= 2


def test_feature_misleading_is_seed_reproducible():
    a = generate_feature_misleading(seed=1, index=0, correct_slot_index=0, feature_distractor_slot=1)
    b = generate_feature_misleading(seed=1, index=0, correct_slot_index=0, feature_distractor_slot=1)
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generator_feature_misleading.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement feature-misleading generator**

`src/llm_relations/generator/feature_misleading.py`:

```python
from __future__ import annotations

import random

from llm_relations.problem import Problem
from llm_relations.palette import draw_nonsense_words, draw_colors


_INSTRUCTION = (
    "I'm going to describe two scenarios. In the memory scenario, a novel object called "
    "a {target_memory} has a property: it can be activated by pressing one of its buttons. "
    "Your job is to figure out which object in the perception scenario is the {target_memory}-analog, "
    "and therefore which button on it can be activated."
)


def _describe_object(name: str, buttons: list[tuple[str, str]]) -> str:
    parts = [f"a {c} button on {p}" for c, p in buttons]
    if len(parts) == 1:
        body = parts[0]
    elif len(parts) == 2:
        body = f"{parts[0]} and {parts[1]}"
    else:
        body = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"The {name} has {body}."


def generate_feature_misleading(
    seed: int,
    index: int,
    correct_slot_index: int,
    feature_distractor_slot: int,
) -> Problem:
    assert 0 <= correct_slot_index < 3
    assert 0 <= feature_distractor_slot < 3
    assert correct_slot_index != feature_distractor_slot

    rng = random.Random(seed)
    words = draw_nonsense_words(rng, n=6)
    m0, m1, m2 = words[:3]
    perception_words = words[3:]

    colors = draw_colors(rng, n=3)
    target_color, other1, other2 = colors

    m0_buttons = [(target_color, "top"), (other1, "side"), (other2, "bottom")]
    m1_buttons = [(other1, "top"), (target_color, "bottom")]
    m2_buttons = [(other2, "top"), (target_color, "side")]

    memory_text = " ".join([
        f"Memory scenario: There are three objects on a table: a {m0}, a {m1}, and a {m2}.",
        f"The {m0} is underneath the {m1}.",
        f"The {m1} is to-the-left-of the {m2}.",
        _describe_object(m0, m0_buttons),
        _describe_object(m1, m1_buttons),
        _describe_object(m2, m2_buttons),
        f"Pressing the {target_color} button on the {m0} activates it.",
    ])

    correct_analog = perception_words[0]
    list_order = [None, None, None]
    list_order[correct_slot_index] = correct_analog
    remaining = [perception_words[1], perception_words[2]]
    distractor = remaining[0]
    list_order[feature_distractor_slot] = distractor
    other_slot = [i for i in range(3) if list_order[i] is None][0]
    list_order[other_slot] = remaining[1]

    # Distractor now has TWO target-color buttons (top + side), to strengthen feature lure.
    perception_buttons = {
        correct_analog: [(target_color, "top"), (other1, "side"), (other2, "bottom")],
        distractor: [(target_color, "top"), (target_color, "side"), (other1, "bottom")],
        remaining[1]: [(other2, "top"), (other1, "bottom")],
    }

    perception_text = " ".join([
        f"Perception scenario: There are three objects on a shelf: "
        f"a {list_order[0]}, a {list_order[1]}, and a {list_order[2]}.",
        f"The {perception_words[0]} is underneath the {perception_words[1]}.",
        f"The {perception_words[1]} is to-the-left-of the {perception_words[2]}.",
        _describe_object(list_order[0], perception_buttons[list_order[0]]),
        _describe_object(list_order[1], perception_buttons[list_order[1]]),
        _describe_object(list_order[2], perception_buttons[list_order[2]]),
    ])

    question = f"Which object in the perception scenario is the {m0}-analog, and which button activates it?"

    prompt = "\n\n".join([
        _INSTRUCTION.format(target_memory=m0),
        memory_text,
        perception_text,
        question,
    ])

    positional_analog = list_order[0]
    positional_button = next(c for c, p in perception_buttons[positional_analog] if p == "top")

    return Problem(
        problem_id=f"feature_misleading_{index:02d}",
        variant="feature_misleading",
        prompt_text=prompt,
        correct_answer={"analog": correct_analog, "button_color": target_color},
        metadata={
            "n_objects": 3,
            "seed": seed,
            "correct_slot_index": correct_slot_index,
            "feature_distractor_slot": feature_distractor_slot,
            "feature_match_answer": {"analog": distractor, "button_color": target_color},
            "positional_match_answer": {"analog": positional_analog, "button_color": positional_button},
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generator_feature_misleading.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_relations/generator/feature_misleading.py tests/test_generator_feature_misleading.py
git commit -m "feat: add feature-misleading variant generator"
```

---

## Task 6: Scale generator

**Files:**
- Create: `src/llm_relations/generator/scale.py`
- Create: `tests/test_generator_scale.py`

Design: n objects in a chain. Relations form a linear chain: `obj_0 underneath obj_1`, `obj_1 to-the-left-of obj_2`, `obj_2 above obj_3`, `obj_3 to-the-right-of obj_4`, etc. (alternate between vertical and horizontal to avoid repetition). Memory target is `m0`. Correct analog is perception object in the same relational role (`p_0`). Feature distractor is placed at a fixed offset.

- [ ] **Step 1: Write the failing test**

`tests/test_generator_scale.py`:

```python
import pytest

from llm_relations.generator.scale import generate_scale


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8])
def test_scale_generates_n_objects(n: int):
    p = generate_scale(seed=1, index=0, n_objects=n)
    assert p.variant == "scale"
    assert p.metadata["n_objects"] == n
    # Prompt should reference all n memory + n perception objects
    # (we verify by checking the memory scenario mentions n distinct nonsense words)
    memory_section = p.prompt_text.split("Perception scenario")[0]
    # Count nonsense-word mentions by checking each metadata word
    memory_words = p.metadata["memory_words"]
    assert len(memory_words) == n
    for w in memory_words:
        assert w in memory_section


def test_scale_correct_analog_in_prompt():
    p = generate_scale(seed=2, index=0, n_objects=5)
    assert p.correct_answer["analog"] in p.prompt_text


def test_scale_is_seed_reproducible():
    a = generate_scale(seed=3, index=0, n_objects=6)
    b = generate_scale(seed=3, index=0, n_objects=6)
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generator_scale.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement scale generator**

`src/llm_relations/generator/scale.py`:

```python
from __future__ import annotations

import random

from llm_relations.problem import Problem
from llm_relations.palette import draw_nonsense_words, draw_colors


_INSTRUCTION = (
    "I'm going to describe two scenarios. In the memory scenario, a novel object called "
    "a {target_memory} has a property: it can be activated by pressing one of its buttons. "
    "Your job is to figure out which object in the perception scenario is the {target_memory}-analog, "
    "and therefore which button on it can be activated."
)

_RELATION_CYCLE = [
    "is underneath",
    "is to-the-left-of",
    "is above",
    "is to-the-right-of",
]


def _describe_object(name: str, buttons: list[tuple[str, str]]) -> str:
    parts = [f"a {c} button on {p}" for c, p in buttons]
    if len(parts) == 1:
        body = parts[0]
    elif len(parts) == 2:
        body = f"{parts[0]} and {parts[1]}"
    else:
        body = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"The {name} has {body}."


def _chain_relations(names: list[str]) -> list[str]:
    # Produces n-1 relation sentences linking consecutive objects.
    out = []
    for i in range(len(names) - 1):
        rel = _RELATION_CYCLE[i % len(_RELATION_CYCLE)]
        out.append(f"The {names[i]} {rel} the {names[i+1]}.")
    return out


def generate_scale(seed: int, index: int, n_objects: int) -> Problem:
    assert n_objects >= 3
    rng = random.Random(seed)
    words = draw_nonsense_words(rng, n=2 * n_objects)
    memory_words = words[:n_objects]
    perception_words = words[n_objects:]

    colors = draw_colors(rng, n=3)
    target_color, other1, other2 = colors

    # Memory: chain of n objects; m0 is target. m0 has target_color on top.
    memory_buttons = {memory_words[0]: [(target_color, "top"), (other1, "side"), (other2, "bottom")]}
    for i in range(1, n_objects):
        # Non-target memory objects get one target_color elsewhere + other colors.
        memory_buttons[memory_words[i]] = [
            (rng.choice([other1, other2]), "top"),
            (target_color, rng.choice(["bottom", "side", "front", "back"])),
        ]

    memory_text_parts = [
        f"Memory scenario: There are {n_objects} objects on a table: "
        + ", ".join(f"a {w}" for w in memory_words[:-1])
        + f", and a {memory_words[-1]}.",
        *_chain_relations(memory_words),
        *(_describe_object(w, memory_buttons[w]) for w in memory_words),
        f"Pressing the {target_color} button on the {memory_words[0]} activates it.",
    ]
    memory_text = " ".join(memory_text_parts)

    # Perception: same chain structure, correct analog is perception_words[0].
    correct_analog = perception_words[0]

    # Shuffle listing order for perception objects.
    perception_list_order = perception_words[:]
    rng.shuffle(perception_list_order)

    # Correct analog buttons: same config as memory target.
    perception_buttons = {
        correct_analog: [(target_color, "top"), (other1, "side"), (other2, "bottom")]
    }
    # Designate perception_words[1] as feature distractor: target color elsewhere.
    distractor = perception_words[1]
    perception_buttons[distractor] = [(other1, "top"), (target_color, "side"), (other2, "bottom")]
    # Others: no target_color at all.
    for w in perception_words[2:]:
        perception_buttons[w] = [(rng.choice([other1, other2]), "top"), (rng.choice([other1, other2]), "bottom")]

    perception_text_parts = [
        f"Perception scenario: There are {n_objects} objects on a shelf: "
        + ", ".join(f"a {w}" for w in perception_list_order[:-1])
        + f", and a {perception_list_order[-1]}.",
        *_chain_relations(perception_words),
        *(_describe_object(w, perception_buttons[w]) for w in perception_list_order),
    ]
    perception_text = " ".join(perception_text_parts)

    question = (
        f"Which object in the perception scenario is the {memory_words[0]}-analog, "
        "and which button activates it?"
    )

    prompt = "\n\n".join([
        _INSTRUCTION.format(target_memory=memory_words[0]),
        memory_text,
        perception_text,
        question,
    ])

    positional_analog = perception_list_order[0]
    positional_button = next(c for c, p in perception_buttons[positional_analog] if p == "top")

    return Problem(
        problem_id=f"scale_{index:02d}",
        variant="scale",
        prompt_text=prompt,
        correct_answer={"analog": correct_analog, "button_color": target_color},
        metadata={
            "n_objects": n_objects,
            "seed": seed,
            "memory_words": memory_words,
            "perception_words": perception_words,
            "perception_list_order": perception_list_order,
            "feature_match_answer": {"analog": distractor, "button_color": target_color},
            "positional_match_answer": {"analog": positional_analog, "button_color": positional_button},
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generator_scale.py -v`
Expected: 7 passed (5 parametrized + 2 others).

- [ ] **Step 5: Commit**

```bash
git add src/llm_relations/generator/scale.py tests/test_generator_scale.py
git commit -m "feat: add scale variant generator"
```

---

## Task 7: Cross-domain generator

**Files:**
- Create: `src/llm_relations/generator/cross_domain.py`
- Create: `tests/test_generator_cross_domain.py`

Design: Preserves baseline's 3-object relational structure, but re-skins the domain. Five domains: `org_chart`, `garden`, `building`, `enclosure`, `vehicle_lot`. Each domain defines: object category name, container noun (table/garden/building/enclosure/lot), two relation phrasings (vertical and horizontal), feature noun (button → skill/leaf/fixture/marking/panel), and "activate" verb. The correct analog still varies by `correct_slot_index`.

- [ ] **Step 1: Write the failing test**

`tests/test_generator_cross_domain.py`:

```python
import pytest

from llm_relations.generator.cross_domain import generate_cross_domain, DOMAINS


def test_domains_list_has_five_entries():
    assert set(DOMAINS) == {"org_chart", "garden", "building", "enclosure", "vehicle_lot"}


@pytest.mark.parametrize("domain", sorted(["org_chart", "garden", "building", "enclosure", "vehicle_lot"]))
def test_cross_domain_generates_for_each_domain(domain: str):
    p = generate_cross_domain(seed=1, index=0, domain=domain, correct_slot_index=1, feature_distractor_slot=0)
    assert p.variant == "cross_domain"
    assert p.metadata["domain"] == domain
    assert p.correct_answer["analog"] in p.prompt_text


def test_cross_domain_is_seed_reproducible():
    a = generate_cross_domain(seed=5, index=0, domain="garden", correct_slot_index=2, feature_distractor_slot=1)
    b = generate_cross_domain(seed=5, index=0, domain="garden", correct_slot_index=2, feature_distractor_slot=1)
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generator_cross_domain.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement cross-domain generator**

`src/llm_relations/generator/cross_domain.py`:

```python
from __future__ import annotations

import random
from dataclasses import dataclass

from llm_relations.problem import Problem
from llm_relations.palette import draw_nonsense_words, draw_colors


@dataclass(frozen=True)
class Domain:
    name: str
    container: str                # "on a table"
    category: str                 # "objects"
    feature_noun: str             # "button"
    feature_prefix: str           # what modifier describes the feature (e.g., a color)
    relation_vertical: str        # "is underneath"
    relation_horizontal: str      # "is to-the-left-of"
    activation_phrase: str        # "Pressing the {feature} {feature_noun} on the {name}"


_DOMAINS: dict[str, Domain] = {
    "org_chart": Domain(
        name="org_chart",
        container="in an organization",
        category="employees",
        feature_noun="specialty",
        feature_prefix="color-coded",
        relation_vertical="reports-to",
        relation_horizontal="sits-beside",
        activation_phrase="Assigning the {feature} {feature_noun} to the {name}",
    ),
    "garden": Domain(
        name="garden",
        container="in a garden",
        category="plants",
        feature_noun="leaf",
        feature_prefix="colored",
        relation_vertical="is-growing-under",
        relation_horizontal="is-to-the-left-of",
        activation_phrase="Clipping the {feature} {feature_noun} from the {name}",
    ),
    "building": Domain(
        name="building",
        container="in a building",
        category="rooms",
        feature_noun="fixture",
        feature_prefix="colored",
        relation_vertical="is-directly-below",
        relation_horizontal="is-adjacent-to",
        activation_phrase="Operating the {feature} {feature_noun} in the {name}",
    ),
    "enclosure": Domain(
        name="enclosure",
        container="in an enclosure",
        category="animals",
        feature_noun="marking",
        feature_prefix="colored",
        relation_vertical="is-beneath",
        relation_horizontal="is-to-the-left-of",
        activation_phrase="Touching the {feature} {feature_noun} on the {name}",
    ),
    "vehicle_lot": Domain(
        name="vehicle_lot",
        container="in a lot",
        category="vehicles",
        feature_noun="panel",
        feature_prefix="colored",
        relation_vertical="is-parked-beneath",
        relation_horizontal="is-parked-to-the-left-of",
        activation_phrase="Switching the {feature} {feature_noun} on the {name}",
    ),
}

DOMAINS = list(_DOMAINS.keys())

_POSITIONS = ["top", "bottom", "side"]


def _describe(name: str, feature_noun: str, feature_prefix: str, features: list[tuple[str, str]]) -> str:
    parts = [f"a {feat} {feature_prefix} {feature_noun} on {pos}" for feat, pos in features]
    if len(parts) == 2:
        body = f"{parts[0]} and {parts[1]}"
    else:
        body = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"The {name} has {body}."


def generate_cross_domain(
    seed: int,
    index: int,
    domain: str,
    correct_slot_index: int,
    feature_distractor_slot: int,
) -> Problem:
    assert domain in _DOMAINS, f"unknown domain: {domain}"
    assert 0 <= correct_slot_index < 3
    assert 0 <= feature_distractor_slot < 3
    assert correct_slot_index != feature_distractor_slot

    d = _DOMAINS[domain]
    rng = random.Random(seed)
    words = draw_nonsense_words(rng, n=6)
    m0, m1, m2 = words[:3]
    perception_words = words[3:]

    colors = draw_colors(rng, n=3)
    target_color, other1, other2 = colors

    m0_feats = [(target_color, "top"), (other1, "side"), (other2, "bottom")]
    m1_feats = [(other1, "top"), (target_color, "bottom")]
    m2_feats = [(other2, "top"), (target_color, "side")]

    memory_text = " ".join([
        f"Memory scenario: There are three {d.category} {d.container}: a {m0}, a {m1}, and a {m2}.",
        f"The {m0} {d.relation_vertical} the {m1}.",
        f"The {m1} {d.relation_horizontal} the {m2}.",
        _describe(m0, d.feature_noun, d.feature_prefix, m0_feats),
        _describe(m1, d.feature_noun, d.feature_prefix, m1_feats),
        _describe(m2, d.feature_noun, d.feature_prefix, m2_feats),
        d.activation_phrase.format(feature=target_color, feature_noun=d.feature_noun, name=m0) + " activates it.",
    ])

    correct_analog = perception_words[0]
    list_order = [None, None, None]
    list_order[correct_slot_index] = correct_analog
    remaining = [perception_words[1], perception_words[2]]
    distractor = remaining[0]
    list_order[feature_distractor_slot] = distractor
    other_slot = [i for i in range(3) if list_order[i] is None][0]
    list_order[other_slot] = remaining[1]

    perception_feats = {
        correct_analog: [(target_color, "top"), (other1, "side"), (other2, "bottom")],
        distractor: [(other1, "top"), (target_color, "side")],
        remaining[1]: [(other2, "top"), (other1, "bottom")],
    }

    perception_text = " ".join([
        f"Perception scenario: There are three {d.category} {d.container}: "
        f"a {list_order[0]}, a {list_order[1]}, and a {list_order[2]}.",
        f"The {perception_words[0]} {d.relation_vertical} the {perception_words[1]}.",
        f"The {perception_words[1]} {d.relation_horizontal} the {perception_words[2]}.",
        _describe(list_order[0], d.feature_noun, d.feature_prefix, perception_feats[list_order[0]]),
        _describe(list_order[1], d.feature_noun, d.feature_prefix, perception_feats[list_order[1]]),
        _describe(list_order[2], d.feature_noun, d.feature_prefix, perception_feats[list_order[2]]),
    ])

    question = (
        f"Which {d.category[:-1]} in the perception scenario is the {m0}-analog, "
        f"and which {d.feature_prefix} {d.feature_noun} activates it?"
    )

    instruction = (
        f"I'm going to describe two scenarios. In the memory scenario, a novel {d.category[:-1]} called "
        f"a {m0} has a property: it can be activated by one of its {d.feature_prefix} {d.feature_noun}s. "
        f"Your job is to figure out which {d.category[:-1]} in the perception scenario is the {m0}-analog, "
        f"and therefore which {d.feature_prefix} {d.feature_noun} on it can be activated."
    )

    prompt = "\n\n".join([instruction, memory_text, perception_text, question])

    positional_analog = list_order[0]
    positional_feature = next(c for c, p in perception_feats[positional_analog] if p == "top")

    return Problem(
        problem_id=f"cross_domain_{index:02d}",
        variant="cross_domain",
        prompt_text=prompt,
        correct_answer={"analog": correct_analog, "button_color": target_color},
        metadata={
            "n_objects": 3,
            "seed": seed,
            "domain": domain,
            "correct_slot_index": correct_slot_index,
            "feature_distractor_slot": feature_distractor_slot,
            "feature_match_answer": {"analog": distractor, "button_color": target_color},
            "positional_match_answer": {"analog": positional_analog, "button_color": positional_feature},
        },
    )
```

Note: the prompt parser still expects JSON with keys `analog` and `button_color` even in cross-domain — the "button_color" key semantically holds whatever feature color is relevant. The system prompt in Task 12 tells the model to answer in that schema regardless of domain.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generator_cross_domain.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_relations/generator/cross_domain.py tests/test_generator_cross_domain.py
git commit -m "feat: add cross-domain variant generator"
```

---

## Task 8: Adversarial generator

**Files:**
- Create: `src/llm_relations/generator/adversarial.py`
- Create: `tests/test_generator_adversarial.py`

Design: Same structure as feature-misleading (two target-color buttons on the distractor), but an extra *linguistic* decoy: inject a sentence at the end of the perception description such as *"The distractor's {target_color} button lights up when pressed."* The phrasing mimics training-data patterns for "activated by" without asserting that the distractor is the answer. The correct answer is still the structure-match analog.

- [ ] **Step 1: Write the failing test**

`tests/test_generator_adversarial.py`:

```python
from llm_relations.generator.adversarial import generate_adversarial, LINGUISTIC_DECOYS


def test_adversarial_has_linguistic_decoy_sentence():
    p = generate_adversarial(seed=1, index=0, correct_slot_index=1, feature_distractor_slot=0, decoy_index=0)
    distractor = p.metadata["feature_match_answer"]["analog"]
    color = p.correct_answer["button_color"]
    # At least one of the known decoy templates should appear in the prompt,
    # instantiated for the distractor.
    assert any(
        template.format(name=distractor, color=color) in p.prompt_text
        for template in LINGUISTIC_DECOYS
    )


def test_adversarial_correct_answer_is_structure_match_not_distractor():
    p = generate_adversarial(seed=2, index=0, correct_slot_index=2, feature_distractor_slot=0, decoy_index=1)
    assert p.correct_answer["analog"] != p.metadata["feature_match_answer"]["analog"]


def test_adversarial_is_seed_reproducible():
    a = generate_adversarial(seed=9, index=0, correct_slot_index=1, feature_distractor_slot=2, decoy_index=3)
    b = generate_adversarial(seed=9, index=0, correct_slot_index=1, feature_distractor_slot=2, decoy_index=3)
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generator_adversarial.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement adversarial generator**

`src/llm_relations/generator/adversarial.py`:

```python
from __future__ import annotations

import random

from llm_relations.problem import Problem
from llm_relations.palette import draw_nonsense_words, draw_colors


_INSTRUCTION = (
    "I'm going to describe two scenarios. In the memory scenario, a novel object called "
    "a {target_memory} has a property: it can be activated by pressing one of its buttons. "
    "Your job is to figure out which object in the perception scenario is the {target_memory}-analog, "
    "and therefore which button on it can be activated."
)

LINGUISTIC_DECOYS: list[str] = [
    "Notably, the {name}'s {color} button glows softly when anyone walks by.",
    "The {name}'s {color} button is described in the catalog as its primary control.",
    "People who have interacted with the {name} report that its {color} button is the one you press first.",
    "The {name} is famous for its prominent {color} button.",
    "The {name}'s {color} button is the one usually associated with activation in similar objects.",
]


def _describe_object(name: str, buttons: list[tuple[str, str]]) -> str:
    parts = [f"a {c} button on {p}" for c, p in buttons]
    if len(parts) == 2:
        body = f"{parts[0]} and {parts[1]}"
    else:
        body = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"The {name} has {body}."


def generate_adversarial(
    seed: int,
    index: int,
    correct_slot_index: int,
    feature_distractor_slot: int,
    decoy_index: int,
) -> Problem:
    assert 0 <= correct_slot_index < 3
    assert 0 <= feature_distractor_slot < 3
    assert correct_slot_index != feature_distractor_slot
    assert 0 <= decoy_index < len(LINGUISTIC_DECOYS)

    rng = random.Random(seed)
    words = draw_nonsense_words(rng, n=6)
    m0, m1, m2 = words[:3]
    perception_words = words[3:]

    colors = draw_colors(rng, n=3)
    target_color, other1, other2 = colors

    m0_buttons = [(target_color, "top"), (other1, "side"), (other2, "bottom")]
    m1_buttons = [(other1, "top"), (target_color, "bottom")]
    m2_buttons = [(other2, "top"), (target_color, "side")]

    memory_text = " ".join([
        f"Memory scenario: There are three objects on a table: a {m0}, a {m1}, and a {m2}.",
        f"The {m0} is underneath the {m1}.",
        f"The {m1} is to-the-left-of the {m2}.",
        _describe_object(m0, m0_buttons),
        _describe_object(m1, m1_buttons),
        _describe_object(m2, m2_buttons),
        f"Pressing the {target_color} button on the {m0} activates it.",
    ])

    correct_analog = perception_words[0]
    list_order = [None, None, None]
    list_order[correct_slot_index] = correct_analog
    remaining = [perception_words[1], perception_words[2]]
    distractor = remaining[0]
    list_order[feature_distractor_slot] = distractor
    other_slot = [i for i in range(3) if list_order[i] is None][0]
    list_order[other_slot] = remaining[1]

    perception_buttons = {
        correct_analog: [(target_color, "top"), (other1, "side"), (other2, "bottom")],
        distractor: [(target_color, "top"), (target_color, "side"), (other1, "bottom")],
        remaining[1]: [(other2, "top"), (other1, "bottom")],
    }

    decoy_sentence = LINGUISTIC_DECOYS[decoy_index].format(name=distractor, color=target_color)

    perception_text = " ".join([
        f"Perception scenario: There are three objects on a shelf: "
        f"a {list_order[0]}, a {list_order[1]}, and a {list_order[2]}.",
        f"The {perception_words[0]} is underneath the {perception_words[1]}.",
        f"The {perception_words[1]} is to-the-left-of the {perception_words[2]}.",
        _describe_object(list_order[0], perception_buttons[list_order[0]]),
        _describe_object(list_order[1], perception_buttons[list_order[1]]),
        _describe_object(list_order[2], perception_buttons[list_order[2]]),
        decoy_sentence,
    ])

    question = f"Which object in the perception scenario is the {m0}-analog, and which button activates it?"

    prompt = "\n\n".join([
        _INSTRUCTION.format(target_memory=m0),
        memory_text,
        perception_text,
        question,
    ])

    positional_analog = list_order[0]
    positional_button = next(c for c, p in perception_buttons[positional_analog] if p == "top")

    return Problem(
        problem_id=f"adversarial_{index:02d}",
        variant="adversarial",
        prompt_text=prompt,
        correct_answer={"analog": correct_analog, "button_color": target_color},
        metadata={
            "n_objects": 3,
            "seed": seed,
            "correct_slot_index": correct_slot_index,
            "feature_distractor_slot": feature_distractor_slot,
            "decoy_index": decoy_index,
            "feature_match_answer": {"analog": distractor, "button_color": target_color},
            "positional_match_answer": {"analog": positional_analog, "button_color": positional_button},
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generator_adversarial.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_relations/generator/adversarial.py tests/test_generator_adversarial.py
git commit -m "feat: add adversarial variant generator"
```

---

## Task 9: Freeze corpus script and corpus validation

**Files:**
- Create: `scripts/freeze_corpus.py`
- Create: `tests/test_corpus.py`

- [ ] **Step 1: Write the freeze script**

`scripts/freeze_corpus.py`:

```python
"""Instantiate all 25 problems with fixed seeds and write them to problems/."""
from __future__ import annotations

from pathlib import Path

from llm_relations.problem import save_problem
from llm_relations.generator.baseline import generate_baseline
from llm_relations.generator.feature_misleading import generate_feature_misleading
from llm_relations.generator.scale import generate_scale
from llm_relations.generator.cross_domain import generate_cross_domain
from llm_relations.generator.adversarial import generate_adversarial


PROBLEMS_DIR = Path(__file__).resolve().parent.parent / "problems"


# (seed, correct_slot_index, feature_distractor_slot) per instance.
# 5 combos; covers all (correct, distractor) pairs at least once.
BASELINE_CONFIGS = [
    (1001, 0, 1),
    (1002, 1, 0),
    (1003, 2, 0),
    (1004, 1, 2),
    (1005, 2, 1),
]

FEATURE_MISLEADING_CONFIGS = [
    (2001, 0, 2),
    (2002, 1, 0),
    (2003, 2, 1),
    (2004, 0, 1),
    (2005, 2, 0),
]

# Scale test: one instance per size.
SCALE_CONFIGS = [
    (3001, 4),
    (3002, 5),
    (3003, 6),
    (3004, 7),
    (3005, 8),
]

# One instance per domain.
CROSS_DOMAIN_CONFIGS = [
    (4001, "org_chart", 2, 0),
    (4002, "garden", 1, 2),
    (4003, "building", 0, 1),
    (4004, "enclosure", 2, 1),
    (4005, "vehicle_lot", 1, 0),
]

# Five linguistic decoys (one each).
ADVERSARIAL_CONFIGS = [
    (5001, 1, 0, 0),
    (5002, 2, 1, 1),
    (5003, 0, 2, 2),
    (5004, 2, 0, 3),
    (5005, 1, 2, 4),
]


def main() -> None:
    PROBLEMS_DIR.mkdir(exist_ok=True)

    for i, (seed, correct, distractor) in enumerate(BASELINE_CONFIGS):
        p = generate_baseline(seed=seed, index=i, correct_slot_index=correct, feature_distractor_slot=distractor)
        save_problem(p, PROBLEMS_DIR / f"{p.problem_id}.json")

    for i, (seed, correct, distractor) in enumerate(FEATURE_MISLEADING_CONFIGS):
        p = generate_feature_misleading(seed=seed, index=i, correct_slot_index=correct, feature_distractor_slot=distractor)
        save_problem(p, PROBLEMS_DIR / f"{p.problem_id}.json")

    for i, (seed, n) in enumerate(SCALE_CONFIGS):
        p = generate_scale(seed=seed, index=i, n_objects=n)
        save_problem(p, PROBLEMS_DIR / f"{p.problem_id}.json")

    for i, (seed, domain, correct, distractor) in enumerate(CROSS_DOMAIN_CONFIGS):
        p = generate_cross_domain(seed=seed, index=i, domain=domain, correct_slot_index=correct, feature_distractor_slot=distractor)
        save_problem(p, PROBLEMS_DIR / f"{p.problem_id}.json")

    for i, (seed, correct, distractor, decoy) in enumerate(ADVERSARIAL_CONFIGS):
        p = generate_adversarial(
            seed=seed, index=i, correct_slot_index=correct,
            feature_distractor_slot=distractor, decoy_index=decoy,
        )
        save_problem(p, PROBLEMS_DIR / f"{p.problem_id}.json")

    print(f"Wrote {len(list(PROBLEMS_DIR.glob('*.json')))} problems to {PROBLEMS_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write corpus validation tests**

`tests/test_corpus.py`:

```python
from pathlib import Path

import pytest

from llm_relations.problem import load_problem

PROBLEMS_DIR = Path(__file__).resolve().parent.parent / "problems"


def _all_problems():
    files = sorted(PROBLEMS_DIR.glob("*.json"))
    return [load_problem(f) for f in files]


def test_corpus_has_25_problems():
    if not PROBLEMS_DIR.exists() or not list(PROBLEMS_DIR.glob("*.json")):
        pytest.skip("Corpus not yet frozen — run scripts/freeze_corpus.py")
    assert len(list(PROBLEMS_DIR.glob("*.json"))) == 25


def test_corpus_covers_all_five_variants():
    if not list(PROBLEMS_DIR.glob("*.json")):
        pytest.skip("Corpus not yet frozen")
    problems = _all_problems()
    variants = {p.variant for p in problems}
    assert variants == {"baseline", "feature_misleading", "scale", "cross_domain", "adversarial"}
    for variant in variants:
        assert sum(1 for p in problems if p.variant == variant) == 5


def test_each_problem_has_internally_consistent_ground_truth():
    if not list(PROBLEMS_DIR.glob("*.json")):
        pytest.skip("Corpus not yet frozen")
    for p in _all_problems():
        # The correct analog must appear in the prompt text.
        assert p.correct_answer["analog"] in p.prompt_text, f"{p.problem_id}: analog not in prompt"
        # Distractors must differ from the correct answer.
        fma = p.metadata["feature_match_answer"]["analog"]
        pma = p.metadata["positional_match_answer"]["analog"]
        assert fma != p.correct_answer["analog"], f"{p.problem_id}: feature_match == correct"
        assert pma != p.correct_answer["analog"], f"{p.problem_id}: positional_match == correct"


def test_each_problem_has_unique_id():
    if not list(PROBLEMS_DIR.glob("*.json")):
        pytest.skip("Corpus not yet frozen")
    ids = [p.problem_id for p in _all_problems()]
    assert len(ids) == len(set(ids))
```

- [ ] **Step 3: Run the freeze script**

Run: `uv run python scripts/freeze_corpus.py`
Expected: "Wrote 25 problems to .../problems". Directory populated.

- [ ] **Step 4: Run corpus tests**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: 4 passed.

- [ ] **Step 5: Manual review**

Open 3 random problem JSONs (one small, one scale_04, one cross_domain). Read the `prompt_text`. Confirm:
- The correct analog is unambiguously identified by relational role.
- The feature and positional distractors are genuinely tempting.
- The prompt reads naturally.

If any problem looks degenerate, adjust the seed/config in `freeze_corpus.py`, re-run, and commit. Do not silently regenerate to paper over the issue.

- [ ] **Step 6: Commit the corpus**

```bash
git add scripts/freeze_corpus.py tests/test_corpus.py problems/
git commit -m "feat: freeze and validate 25-problem corpus"
```

---

## Task 10: Response parser

**Files:**
- Create: `src/llm_relations/parser.py`
- Create: `tests/test_parser.py`

- [ ] **Step 1: Write the failing test**

`tests/test_parser.py`:

```python
import pytest

from llm_relations.parser import parse_answer, ParseError


def test_parses_trailing_fenced_json():
    text = """Let me think...

The mapping is mek↔florp.

```json
{"analog": "mek", "button_color": "blue"}
```"""
    result = parse_answer(text)
    assert result == {"analog": "mek", "button_color": "blue"}


def test_parses_last_fenced_json_when_multiple_present():
    text = """First attempt:
```json
{"analog": "zop", "button_color": "blue"}
```
Actually, on reflection:
```json
{"analog": "mek", "button_color": "blue"}
```"""
    result = parse_answer(text)
    assert result == {"analog": "mek", "button_color": "blue"}


def test_parses_plain_fenced_block_without_json_tag():
    text = """Answer:
```
{"analog": "mek", "button_color": "blue"}
```"""
    result = parse_answer(text)
    assert result == {"analog": "mek", "button_color": "blue"}


def test_raises_on_missing_fence():
    with pytest.raises(ParseError):
        parse_answer("I think the answer is mek with the blue button.")


def test_raises_on_malformed_json():
    text = "```json\n{analog: mek, button_color: blue}\n```"
    with pytest.raises(ParseError):
        parse_answer(text)


def test_raises_on_missing_keys():
    text = '```json\n{"analog": "mek"}\n```'
    with pytest.raises(ParseError):
        parse_answer(text)


def test_raises_on_non_string_values():
    text = '```json\n{"analog": 42, "button_color": "blue"}\n```'
    with pytest.raises(ParseError):
        parse_answer(text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_parser.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement parser**

`src/llm_relations/parser.py`:

```python
from __future__ import annotations

import json
import re


class ParseError(Exception):
    """Raised when a model response cannot be parsed into the expected answer format."""


_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


def parse_answer(text: str) -> dict[str, str]:
    """Extract the last fenced JSON block from `text` and return it as a dict.

    Raises ParseError if:
    - No fenced block is found
    - The block is not valid JSON
    - Required keys are missing
    - Values are not strings
    """
    matches = _FENCE_RE.findall(text)
    if not matches:
        raise ParseError("no fenced code block found in response")
    raw = matches[-1].strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ParseError(f"expected JSON object, got {type(obj).__name__}")
    if set(obj.keys()) != {"analog", "button_color"}:
        raise ParseError(f"expected keys {{'analog', 'button_color'}}, got {set(obj.keys())}")
    if not isinstance(obj["analog"], str) or not isinstance(obj["button_color"], str):
        raise ParseError("values must be strings")
    return obj
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_parser.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_relations/parser.py tests/test_parser.py
git commit -m "feat: add response parser with robust JSON extraction"
```

---

## Task 11: Scorer

**Files:**
- Create: `src/llm_relations/scorer.py`
- Create: `tests/test_scorer.py`

- [ ] **Step 1: Write the failing test**

`tests/test_scorer.py`:

```python
from llm_relations.problem import Problem
from llm_relations.scorer import score_answer, ScoreResult


def _make_problem() -> Problem:
    return Problem(
        problem_id="baseline_00",
        variant="baseline",
        prompt_text="...",
        correct_answer={"analog": "mek", "button_color": "blue"},
        metadata={
            "n_objects": 3,
            "feature_match_answer": {"analog": "zop", "button_color": "blue"},
            "positional_match_answer": {"analog": "quib", "button_color": "green"},
        },
    )


def test_correct_answer_scored_as_correct():
    p = _make_problem()
    r = score_answer(p, {"analog": "mek", "button_color": "blue"})
    assert r == ScoreResult(is_correct=True, error_type=None)


def test_feature_match_error_classified():
    p = _make_problem()
    r = score_answer(p, {"analog": "zop", "button_color": "blue"})
    assert r == ScoreResult(is_correct=False, error_type="feature_match")


def test_positional_match_error_classified():
    p = _make_problem()
    r = score_answer(p, {"analog": "quib", "button_color": "green"})
    assert r == ScoreResult(is_correct=False, error_type="positional_match")


def test_other_error_classified():
    p = _make_problem()
    r = score_answer(p, {"analog": "mek", "button_color": "red"})
    assert r.is_correct is False
    assert r.error_type == "other"


def test_parse_error_classified():
    p = _make_problem()
    r = score_answer(p, None)
    assert r == ScoreResult(is_correct=False, error_type="parse_error")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scorer.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement scorer**

`src/llm_relations/scorer.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from llm_relations.problem import Problem


ErrorType = Literal["feature_match", "positional_match", "other", "parse_error"]


@dataclass(frozen=True)
class ScoreResult:
    is_correct: bool
    error_type: Optional[ErrorType]


def score_answer(problem: Problem, answer: Optional[dict[str, str]]) -> ScoreResult:
    """Score a parsed model answer against the problem's ground truth.

    `answer=None` indicates a parse failure upstream.
    Error types correspond to Hummel & Heaton's predicted failure modes:
    - feature_match: picked the object with matching surface features
    - positional_match: picked the object in the same list position
    - other: some other wrong answer
    - parse_error: model did not emit a parseable answer
    """
    if answer is None:
        return ScoreResult(is_correct=False, error_type="parse_error")

    if answer == problem.correct_answer:
        return ScoreResult(is_correct=True, error_type=None)

    if answer == problem.metadata.get("feature_match_answer"):
        return ScoreResult(is_correct=False, error_type="feature_match")

    if answer == problem.metadata.get("positional_match_answer"):
        return ScoreResult(is_correct=False, error_type="positional_match")

    return ScoreResult(is_correct=False, error_type="other")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scorer.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_relations/scorer.py tests/test_scorer.py
git commit -m "feat: add scorer with Hummel & Heaton error-type classification"
```

---

## Task 12: API client wrapper

**Files:**
- Create: `src/llm_relations/runner/client.py`
- Create: `tests/test_runner_client.py`

- [ ] **Step 1: Write the failing test**

`tests/test_runner_client.py`:

```python
from unittest.mock import MagicMock

import pytest

from llm_relations.runner.client import ClaudeClient, SYSTEM_PROMPT, CallResult


def _mock_message(text: str, input_tokens: int = 100, output_tokens: int = 200) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text, type="text")]
    msg.usage.input_tokens = input_tokens
    msg.usage.output_tokens = output_tokens
    msg.usage.cache_creation_input_tokens = 0
    msg.usage.cache_read_input_tokens = 0
    return msg


def test_client_calls_messages_create_with_expected_arguments(mocker):
    fake_anthropic = MagicMock()
    fake_anthropic.messages.create.return_value = _mock_message("hello")
    mocker.patch("llm_relations.runner.client.Anthropic", return_value=fake_anthropic)

    client = ClaudeClient(api_key="test-key")
    result = client.call(model="claude-haiku-4-5-20251001", user_prompt="Solve this.")

    assert isinstance(result, CallResult)
    assert result.response_text == "hello"
    assert result.input_tokens == 100
    assert result.output_tokens == 200

    call_kwargs = fake_anthropic.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
    assert call_kwargs["max_tokens"] == 4096
    assert call_kwargs["temperature"] == 1.0
    # System prompt uses cache_control
    assert call_kwargs["system"][0]["text"] == SYSTEM_PROMPT
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call_kwargs["messages"] == [{"role": "user", "content": "Solve this."}]


def test_client_retries_on_rate_limit(mocker):
    from anthropic import RateLimitError

    fake_anthropic = MagicMock()
    # Fail twice with rate limit, then succeed
    err = RateLimitError(
        message="rate limited", response=MagicMock(status_code=429), body=None
    )
    fake_anthropic.messages.create.side_effect = [err, err, _mock_message("ok")]
    mocker.patch("llm_relations.runner.client.Anthropic", return_value=fake_anthropic)
    # Patch the sleep-between-retries to avoid slow tests
    mocker.patch("llm_relations.runner.client._sleep", return_value=None)

    client = ClaudeClient(api_key="test-key")
    result = client.call(model="claude-haiku-4-5-20251001", user_prompt="Solve.")

    assert result.response_text == "ok"
    assert fake_anthropic.messages.create.call_count == 3


def test_client_retries_then_fails_after_max_attempts(mocker):
    from anthropic import APIStatusError

    fake_anthropic = MagicMock()
    err = APIStatusError(
        message="overloaded", response=MagicMock(status_code=529), body=None
    )
    fake_anthropic.messages.create.side_effect = err
    mocker.patch("llm_relations.runner.client.Anthropic", return_value=fake_anthropic)
    mocker.patch("llm_relations.runner.client._sleep", return_value=None)

    client = ClaudeClient(api_key="test-key", max_retries=3)
    with pytest.raises(APIStatusError):
        client.call(model="claude-haiku-4-5-20251001", user_prompt="Solve.")
    assert fake_anthropic.messages.create.call_count == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runner_client.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement client**

`src/llm_relations/runner/client.py`:

```python
from __future__ import annotations

import time
from dataclasses import dataclass

from anthropic import Anthropic, APIStatusError, RateLimitError


SYSTEM_PROMPT = (
    "You are solving relational reasoning problems. Each problem has a memory scenario "
    "and a perception scenario. Your task is to map objects in the perception scenario "
    "to objects in the memory scenario based on their relational structure (how they "
    "relate to each other), then answer a specific question.\n\n"
    "Think step by step: first identify the relations in each scenario, then find the "
    "mapping that preserves relational structure, then answer.\n\n"
    "End your response with a JSON block in this exact format:\n"
    "```json\n"
    '{"analog": "<object_name>", "button_color": "<color>"}\n'
    "```"
)


@dataclass(frozen=True)
class CallResult:
    response_text: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    latency_ms: int


def _sleep(seconds: float) -> None:
    # Wrapper so tests can patch it out.
    time.sleep(seconds)


class ClaudeClient:
    def __init__(self, api_key: str, max_retries: int = 5, base_delay: float = 2.0):
        self._client = Anthropic(api_key=api_key)
        self._max_retries = max_retries
        self._base_delay = base_delay

    def call(
        self,
        model: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> CallResult:
        attempt = 0
        while True:
            start = time.time()
            try:
                msg = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_prompt}],
                )
            except (RateLimitError, APIStatusError) as e:
                # Retry on 429 and 529 (overloaded); re-raise on other status codes.
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

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_runner_client.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_relations/runner/client.py tests/test_runner_client.py
git commit -m "feat: add Anthropic API client with retry and prompt caching"
```

---

## Task 13: Benchmark runner orchestration

**Files:**
- Create: `src/llm_relations/runner/benchmark.py`
- Create: `scripts/run_benchmark.py`
- Create: `tests/test_runner_benchmark.py`

- [ ] **Step 1: Write the failing test**

`tests/test_runner_benchmark.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runner_benchmark.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement benchmark orchestration**

`src/llm_relations/runner/benchmark.py`:

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
from llm_relations.runner.client import ClaudeClient
from llm_relations.scorer import score_answer


@dataclass(frozen=True)
class SampleRecord:
    problem_id: str
    model: str
    sample: int
    variant: str
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
    client: ClaudeClient,
    model: str,
    sample: int,
    problem: Problem,
) -> SampleRecord:
    result = client.call(model=model, user_prompt=problem.prompt_text)
    try:
        parsed = parse_answer(result.response_text)
    except ParseError:
        parsed = None
    score = score_answer(problem, parsed)
    return SampleRecord(
        problem_id=problem.problem_id,
        model=model,
        sample=sample,
        variant=problem.variant,
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
    out_dir = results_dir / "raw" / record.model / record.problem_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"sample_{record.sample}.json").write_text(
        json.dumps(asdict(record), indent=2, sort_keys=True)
    )


def _write_summary(results_dir: Path, records: list[SampleRecord]) -> None:
    # Aggregate to one row per (model, problem_id).
    rows: dict[tuple[str, str], dict] = {}
    for r in records:
        key = (r.model, r.problem_id)
        if key not in rows:
            rows[key] = {
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

    fieldnames = [
        "model", "variant", "problem_id", "n_samples", "n_correct", "accuracy",
        "n_feature_match", "n_positional_match", "n_other", "n_parse_error",
        "mean_output_tokens", "mean_latency_ms",
    ]
    csv_path = results_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for agg in rows.values():
            n = agg["n_samples"]
            writer.writerow({
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
            })


def run_benchmark(
    problems_dir: Path,
    results_dir: Path,
    models: list[str],
    n_samples: int,
    client: ClaudeClient,
) -> None:
    problems = [load_problem(f) for f in sorted(problems_dir.glob("*.json"))]
    records: list[SampleRecord] = []
    for model in models:
        for problem in problems:
            for s in range(n_samples):
                record = _run_one_sample(client, model, s, problem)
                _write_sample(results_dir, record)
                records.append(record)
    _write_summary(results_dir, records)
```

- [ ] **Step 4: Write the CLI script**

`scripts/run_benchmark.py`:

```python
"""CLI entry point: run the full benchmark."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from llm_relations.runner.benchmark import run_benchmark
from llm_relations.runner.client import ClaudeClient


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
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    client = ClaudeClient(api_key=api_key)
    run_benchmark(
        problems_dir=args.problems_dir,
        results_dir=args.results_dir,
        models=args.models,
        n_samples=args.n_samples,
        client=client,
    )
    print(f"Done. Summary at {args.results_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_runner_benchmark.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/llm_relations/runner/benchmark.py scripts/run_benchmark.py tests/test_runner_benchmark.py
git commit -m "feat: add benchmark orchestration with per-sample logs and summary CSV"
```

---

## Task 14: Smoke test (opt-in real API)

**Files:**
- Create: `tests/test_smoke_api.py`

- [ ] **Step 1: Write the smoke test**

`tests/test_smoke_api.py`:

```python
"""Opt-in smoke test that hits the real Anthropic API.

Run with: uv run pytest -m smoke -v
Skipped unless the smoke marker is selected AND ANTHROPIC_API_KEY is set.
"""
import os

import pytest

from llm_relations.runner.client import ClaudeClient
from llm_relations.parser import parse_answer, ParseError


@pytest.mark.smoke
def test_real_haiku_call_returns_parseable_response():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    client = ClaudeClient(api_key=api_key)
    result = client.call(
        model="claude-haiku-4-5-20251001",
        user_prompt=(
            "Memory scenario: a florp is underneath a greeble. "
            "The florp has a blue button on top. Pressing the blue button on the florp activates it.\n\n"
            "Perception scenario: a mek is underneath a quib. "
            "The mek has a blue button on top.\n\n"
            "Which object in the perception scenario is the florp-analog?"
        ),
    )

    assert result.output_tokens > 0
    # Response should include a fenced JSON answer per the system prompt.
    try:
        answer = parse_answer(result.response_text)
    except ParseError as e:
        pytest.fail(f"Response not parseable: {e}\n\nResponse:\n{result.response_text}")
    assert "analog" in answer and "button_color" in answer
```

- [ ] **Step 2: Run the smoke test**

Run: `ANTHROPIC_API_KEY=<key> uv run pytest -m smoke -v`
Expected: 1 passed. If it fails, inspect the real response — either the system prompt needs adjustment or the parser is too strict.

- [ ] **Step 3: Verify normal test runs do NOT execute the smoke test**

Run: `uv run pytest -v`
Expected: smoke test is deselected (no smoke marker in default run). All other tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke_api.py
git commit -m "test: add opt-in smoke test against real Anthropic API"
```

---

## Task 15: Analysis notebook

**Files:**
- Create: `analysis/report.ipynb`

- [ ] **Step 1: Author the notebook**

Create `analysis/report.ipynb` with the following cells (use `jupyter notebook analysis/report.ipynb` or write JSON directly). The notebook is committed with outputs cleared.

**Cell 1 (markdown):**
```
# Relational Reasoning Benchmark — Report

Loads `results/summary.csv` and `results/raw/` and renders per-variant × per-model accuracy, scale-test curve, error-type breakdown, and per-problem drill-down. Rerun all cells after each benchmark run.
```

**Cell 2 (code):**
```python
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path.cwd().parent if Path.cwd().name == "analysis" else Path.cwd()
RESULTS = REPO / "results"
RAW = RESULTS / "raw"

summary = pd.read_csv(RESULTS / "summary.csv")
summary.head()
```

**Cell 3 (markdown):** `## Accuracy by variant and model`

**Cell 4 (code):**
```python
pivot = (
    summary.groupby(["variant", "model"], as_index=False)
    .agg(accuracy=("accuracy", "mean"),
         se=("accuracy", lambda x: x.std(ddof=1) / (len(x) ** 0.5)),
         n_instances=("problem_id", "nunique"))
)
display_pivot = pivot.pivot(index="variant", columns="model", values="accuracy").round(3)
display_pivot
```

**Cell 5 (markdown):** `## Scale-test curve`

**Cell 6 (code):**
```python
scale_rows = summary[summary["variant"] == "scale"].copy()
# Extract n_objects from the problem JSON.
def _n_objects(problem_id):
    path = REPO / "problems" / f"{problem_id}.json"
    return json.loads(path.read_text())["metadata"]["n_objects"]

scale_rows["n_objects"] = scale_rows["problem_id"].map(_n_objects)

fig, ax = plt.subplots(figsize=(7, 4))
for model, g in scale_rows.groupby("model"):
    g = g.sort_values("n_objects")
    ax.plot(g["n_objects"], g["accuracy"], marker="o", label=model)
ax.set_xlabel("Number of objects")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Scale-test accuracy vs. problem size")
ax.legend()
fig.tight_layout()
plt.show()
```

**Cell 7 (markdown):** `## Error-type breakdown`

**Cell 8 (code):**
```python
error_cols = ["n_correct", "n_feature_match", "n_positional_match", "n_other", "n_parse_error"]
by_model = summary.groupby("model")[error_cols].sum()
# Normalize to proportions of samples.
by_model_pct = by_model.div(by_model.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(7, 4))
by_model_pct.plot(kind="bar", stacked=True, ax=ax)
ax.set_ylabel("Proportion of samples")
ax.set_title("Outcome and error-type breakdown by model")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
plt.show()
```

**Cell 9 (markdown):** `## Per-problem drill-down`

**Cell 10 (code):**
```python
drill = summary.sort_values(["variant", "problem_id", "model"])
drill[["model", "variant", "problem_id", "accuracy", "n_feature_match", "n_positional_match", "n_parse_error"]]
```

**Cell 11 (markdown):**
```
## Inspect a single response

Change the `model_id`, `problem_id`, and `sample_n` below to view a specific model output.
```

**Cell 12 (code):**
```python
model_id = "claude-haiku-4-5-20251001"
problem_id = "baseline_00"
sample_n = 0

path = RAW / model_id / problem_id / f"sample_{sample_n}.json"
rec = json.loads(path.read_text())
print("=== PROMPT ===\n")
print(rec["prompt"])
print("\n=== RESPONSE ===\n")
print(rec["response_text"])
print("\n=== SCORE ===")
print({k: rec[k] for k in ["is_correct", "error_type", "parsed_answer", "correct_answer"]})
```

- [ ] **Step 2: Clear outputs and commit**

Run:
```bash
uv run jupyter nbconvert --clear-output --inplace analysis/report.ipynb
git add analysis/report.ipynb
git commit -m "feat: add analysis notebook for benchmark report"
```

---

## Task 16: Full benchmark run + final report

This task is operational rather than code-authoring. It runs the whole benchmark end-to-end and verifies the notebook regenerates cleanly.

- [ ] **Step 1: Confirm all tests still pass**

Run: `uv run pytest -v`
Expected: all tests pass (smoke test is deselected).

- [ ] **Step 2: Execute the benchmark**

Requires `ANTHROPIC_API_KEY` set. Estimated cost: ~$16–25 at n=5.

Run: `ANTHROPIC_API_KEY=<key> uv run python scripts/run_benchmark.py`
Expected: ~375 API calls; `results/summary.csv` written; `results/raw/{model}/{problem_id}/sample_{n}.json` populated.

If any call fails with a non-retryable error, the run aborts. To resume, you would re-run — the current design is non-resumable (acceptable at this scale). If this becomes painful, add resumability as a follow-up.

- [ ] **Step 3: Render the notebook**

Run:
```bash
uv run jupyter nbconvert --to notebook --execute analysis/report.ipynb --output report-executed.ipynb
```
Expected: `analysis/report-executed.ipynb` created with all outputs populated. Inspect figures for sanity (accuracy numbers in [0, 1], scale curve roughly monotone, no errors).

- [ ] **Step 4: Commit results**

```bash
git add results/summary.csv analysis/report-executed.ipynb
git commit -m "feat: run full benchmark and render report"
```

Note: `results/raw/` is gitignored — keep raw files locally for inspection but don't commit them.

- [ ] **Step 5: (Optional) Briefly interpret in README**

If you want a pointer for future readers, add a short section to `README.md` with: one-line summary of per-model accuracy, link to the executed notebook, and a note on what the error-type breakdown showed (does it match Hummel & Heaton's prediction of feature-match errors dominating?).

---

## Self-review checklist (for the plan author)

- [x] Every variant has a generator task with tests written first.
- [x] Every task has concrete code; no "implement later" placeholders.
- [x] `Problem` schema (Task 2) matches usage in generators (Tasks 4–8), scorer (Task 11), and runner (Task 13).
- [x] Error-type names (`feature_match`, `positional_match`, `other`, `parse_error`) are consistent across scorer, runner, and notebook.
- [x] System prompt text in runner matches spec Section "Prompt and response."
- [x] Corpus is frozen (Task 9) before the benchmark runs (Task 16), and the benchmark loads from `problems/` not from generators.
- [x] Smoke test is opt-in via pytest marker; default `pytest` run does not hit the API.
- [x] Notebook (Task 15) consumes `summary.csv` and raw JSON; both are produced by Task 13.
- [x] `results/raw/` is gitignored; `results/summary.csv` is committed (see Task 1 .gitignore and Task 16 Step 4).
