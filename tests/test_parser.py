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
