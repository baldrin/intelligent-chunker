"""Thin helpers around the Anthropic Messages API.

Centralizes client creation and the structured-output call both passes use, so
the schema-in / parsed-dict-out contract lives in one place.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import anthropic

logger = logging.getLogger(__name__)

# Transient failures worth retrying at the pipeline level, on top of the SDK's
# own 2 built-in retries: rate limits, 5xx/529, and network drops. A single
# burst mid-run must not discard the sections already chunked.
_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)


class StructuredOutputError(RuntimeError):
    """The response arrived but its structured output was unusable.

    Truncation, a missing text block, or JSON that fails to parse. Unlike a
    refusal, re-issuing the request is a reasonable repair.
    """


def make_client() -> "anthropic.Anthropic":
    """Create a client. Reads ANTHROPIC_API_KEY (or an `ant` profile) from env."""
    return anthropic.Anthropic()


def structured_call(
    client: "anthropic.Anthropic",
    *,
    model: str,
    system: str,
    content: List[Dict[str, Any]],
    schema: Dict[str, Any],
    max_tokens: int,
    repair_attempts: int = 1,
    transient_retries: int = 2,
) -> Dict[str, Any]:
    """Run one structured-output request and return the parsed JSON object.

    ``content`` is the user-turn content blocks (e.g. a PDF document block plus
    a text instruction). ``schema`` is a JSON Schema the response must satisfy;
    structured outputs guarantee the first text block is conforming JSON.

    Two layers of resilience so one bad call can't sink a multi-call run:
    ``transient_retries`` backs off and retries rate limits / 5xx / network
    errors that survive the SDK's built-in retries; ``repair_attempts``
    re-issues the request when the output came back unusable (truncated or
    unparseable). Refusals are never retried.
    """
    repair_left = max(0, repair_attempts)
    transient_left = max(0, transient_retries)
    delay = 2.0
    while True:
        try:
            return _structured_call_once(
                client,
                model=model,
                system=system,
                content=content,
                schema=schema,
                max_tokens=max_tokens,
            )
        except _RETRYABLE as exc:
            if transient_left <= 0:
                raise
            transient_left -= 1
            logger.warning(
                "Transient API error (%s); retrying in %.0fs",
                type(exc).__name__,
                delay,
            )
            time.sleep(delay)
            delay *= 2
        except StructuredOutputError as exc:
            if repair_left <= 0:
                raise
            repair_left -= 1
            logger.warning("Structured output unusable (%s); re-requesting", exc)


def _structured_call_once(
    client: "anthropic.Anthropic",
    *,
    model: str,
    system: str,
    content: List[Dict[str, Any]],
    schema: Dict[str, Any],
    max_tokens: int,
) -> Dict[str, Any]:
    # Streamed so large ``max_tokens`` values don't hit HTTP timeouts, with an
    # explicit guard for truncation (which would otherwise surface as a
    # confusing JSON parse error on a cut-off document).
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        raise RuntimeError(
            "Model refused the request"
            + (
                f" ({response.stop_details.category})"
                if getattr(response, "stop_details", None)
                else ""
            )
        )
    if response.stop_reason == "max_tokens":
        raise StructuredOutputError(
            f"Structured output was truncated at max_tokens={max_tokens}. "
            "Increase ChunkerConfig.max_output_tokens, or lower "
            "max_pages_per_batch so each request produces less output."
        )

    text = _first_text(response)
    if text is None:
        raise StructuredOutputError("No text block in structured-output response")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"Response JSON failed to parse: {exc}") from exc


def _first_text(response: Any) -> Optional[str]:
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return None
