"""Single-turn extraction call, run once per user message.

extract_turn() takes the FULL conversation history (not just the
latest message -- see session_repository.py for why) and returns a
validated ExtractionResult. It never trusts the LLM's raw text: JSON
parsing and Pydantic validation both have to succeed before anything
is returned to the caller.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from orchestrator.api.schemas.common import ExtractionResult
from orchestrator.api.services.prompts import EXTRACTION_SYSTEM_PROMPT
from orchestrator.indexing.gateway_client import chat_completion


class ExtractionError(Exception):
    """Raised when the LLM's response can't be trusted -- bad JSON,
    or JSON that doesn't match ExtractionResult's shape. Callers
    should treat this as "retry or surface an error", never as
    something to silently paper over.
    """


def _strip_code_fences(text: str) -> str:
    """Models add markdown fences around JSON surprisingly often
    even when told not to -- defend against it rather than trust
    the instruction alone.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_turn(history: list[dict]) -> ExtractionResult:
    """Run one extraction turn.

    `history` is the full conversation so far, standard OpenAI format:
      [{"role": "user", "content": "..."},
       {"role": "assistant", "content": "..."}, ...]

    The system prompt is prepended fresh every call -- it is NOT
    stored in session history, so it can't be edited/lost across turns.
    """
    messages = [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}, *history]

    raw = chat_completion(messages, temperature=0)
    cleaned = _strip_code_fences(raw)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"LLM did not return valid JSON: {exc}\nRaw response: {raw!r}"
        ) from exc

    try:
        return ExtractionResult.model_validate(parsed)
    except ValidationError as exc:
        raise ExtractionError(
            f"LLM JSON did not match expected shape: {exc}"
        ) from exc