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

_ANSWER_FORMAT = (
    "End your response with a JSON block in this exact format:\n"
    "```json\n"
    '{"analog": "<object_name>", "button_color": "<color>"}\n'
    "```"
)


def build_system_prompt(use_cot: bool = True) -> str:
    parts = [_TASK_DESCRIPTION]
    if use_cot:
        parts.append(_COT_INSTRUCTION)
    parts.append(_ANSWER_FORMAT)
    return "\n\n".join(parts)


SYSTEM_PROMPT = build_system_prompt(use_cot=True)


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
    ):
        if base_url is None:
            self._client = Anthropic(api_key=api_key)
        else:
            self._client = Anthropic(api_key=api_key, base_url=base_url)
        self._max_retries = max_retries
        self._base_delay = base_delay

    def call(
        self,
        model: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system_prompt: str = SYSTEM_PROMPT,
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
                            "text": system_prompt,
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
