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
