from __future__ import annotations

import time
from dataclasses import dataclass

from anthropic import Anthropic, APIStatusError, RateLimitError


_TASK_DESCRIPTION = (
    "You are solving relational reasoning problems. Each problem has a memory scenario "
    "and a perception scenario. Your task is to map objects in the perception scenario "
    "to objects in the memory scenario based on their relational structure (how they "
    "relate to each other), then answer a specific question."
)

_COT_INSTRUCTION = (
    "Think step by step: first identify the relations in each scenario, then find the "
    "mapping that preserves relational structure, then answer."
)

_GRAPHICAL_MODEL_INSTRUCTION = (
"""
You will solve an analogy problem that is specifically designed to defeat
two kinds of shortcuts: (a) matching objects by shared colours, and
(b) matching relations by shared meaning. Neither of those shortcuts will
give the correct answer. To solve it reliably, follow this protocol
exactly and write your work out step by step.

STEP 1 — Extract the relation sentences.
  Ignore the feature sentences for now. Write down just the two relation
  sentences from the memory scenario, in the order they appear, and the
  two relation sentences from the perception scenario, in the order they
  appear.

STEP 2 — Assign roles by template position, NOT by predicate meaning.
  The problem is built from a fixed template with three roles: 0, 1, 2.
  The template rule is:
      • Sentence 1 links role 0 (first argument) to role 1 (second argument).
      • Sentence 2 links role 1 (first argument) to role 2 (second argument).
  So role 1 is always the entity that appears in BOTH sentences.
  Apply this rule to the memory scenario:
      memory role 0 = the subject of memory sentence 1
      memory role 1 = the entity that appears in both memory sentences
      memory role 2 = the non-shared entity in memory sentence 2
  Apply the *same* rule, independently, to the perception scenario:
      perception role 0 = the subject of perception sentence 1
      perception role 1 = the entity that appears in both perception sentences
      perception role 2 = the non-shared entity in perception sentence 2
  Write out all six assignments explicitly.

  WARNING — common traps to ignore at this step:
    • Do NOT try to match "reports-to" with "is-growing-under" because they
      both sound hierarchical. The two scenarios may use different slot
      orders for the vertical and lateral predicates.
    • Do NOT try to match objects by their feature colours. One perception
      object will typically share a colour triple with the memory target;
      this is a decoy, not the analog.
    • Do NOT use the listing order from the "There are three X: a, b, c"
      sentence. Only the relation sentences determine roles.

STEP 3 — Build the mapping.
  The analog of a memory object is the perception object with the SAME
  role index. State the three pairs explicitly:
      memory role 0  ↔  perception role 0
      memory role 1  ↔  perception role 1
      memory role 2  ↔  perception role 2

STEP 4 — Identify the target and its analog.
  The memory scenario names one object as the target (the one whose
  activation rule is given). Find its role index in memory. Look up the
  perception object at that same role index — that object is the analog.

STEP 5 — Read off the activating feature.
  The memory activation rule specifies a position (top, side, or bottom).
  Look at the analog object in the perception scenario and report the
  colour of its feature at the SAME position. Do not substitute the
  memory target's colour; use the analog's own colour at that position.

STEP 6 — Sanity check.
  Before answering, verify: is the analog you chose the one whose colour
  triple matches the memory target? If yes, you probably fell into the
  feature-matching trap — recheck step 2. Did you pair relations by
  meaning (hierarchical with hierarchical)? If yes, you probably fell
  into the predicate-semantics trap — recheck step 2.
"""
)

_ANSWER_FORMAT = (
    "End your response with a JSON block in this exact format:\n"
    "```json\n"
    '{"analog": "<object_name>", "button_color": "<color>"}\n'
    "```"
)


PROMPT_VARIANTS: dict[str, str] = {
    "cot": _COT_INSTRUCTION,
    "no_cot": "",
    "graphical_model": _GRAPHICAL_MODEL_INSTRUCTION,
}


def build_system_prompt(prompt_variant: str = "cot") -> str:
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"Unknown prompt_variant {prompt_variant!r}. "
            f"Valid variants: {sorted(PROMPT_VARIANTS)}"
        )
    instruction = PROMPT_VARIANTS[prompt_variant]
    parts = [_TASK_DESCRIPTION]
    if instruction:
        parts.append(instruction)
    parts.append(_ANSWER_FORMAT)
    return "\n\n".join(parts)


SYSTEM_PROMPT = build_system_prompt("cot")


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
    def __init__(
        self,
        api_key: str,
        max_retries: int = 5,
        base_delay: float = 2.0,
        base_url: str | None = None,
        cache_system_prompt: bool = True,
    ):
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
