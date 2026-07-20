"""Thin helpers around the Anthropic Messages API.

Centralizes client creation and the structured-output call both passes use, so
the schema-in / parsed-dict-out contract lives in one place.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import anthropic

logger = logging.getLogger(__name__)

# USD per million tokens, keyed by model-id prefix:
# (input, output, cache write @5min TTL, cache read).
# Verified against Anthropic pricing 2026-07 (cache write = 1.25x input,
# cache read = 0.1x input). Unknown models report tokens without a dollar
# estimate rather than guessing.
_PRICING = {
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
    "claude-sonnet-5": (3.00, 15.00, 3.75, 0.30),
    "claude-opus-4": (5.00, 25.00, 6.25, 0.50),
}


def _pricing_for(model: str):
    for prefix, rates in _PRICING.items():
        if model.startswith(prefix):
            return rates
    return None


class UsageTracker:
    """Thread-safe accumulator of per-model token usage across a run.

    Both passes record into one tracker (Pass 1 batches and Pass 2 sections
    run on worker threads, hence the lock). Counters are kept per model so a
    run mixing models (e.g. a stronger Pass 1 model) still prices correctly.
    """

    _FIELDS = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.by_model: Dict[str, Dict[str, int]] = {}

    def record(self, model: str, usage: Any) -> None:
        """Add one response's ``usage`` object (tolerates missing fields)."""
        if usage is None:
            return
        with self._lock:
            self.calls += 1
            counters = self.by_model.setdefault(
                model, {f: 0 for f in self._FIELDS}
            )
            for field in self._FIELDS:
                counters[field] += getattr(usage, field, None) or 0

    def totals(self) -> Dict[str, int]:
        with self._lock:
            out = {f: 0 for f in self._FIELDS}
            for counters in self.by_model.values():
                for field in self._FIELDS:
                    out[field] += counters[field]
            return out

    def estimated_cost_usd(self) -> Optional[float]:
        """Dollar estimate, or None when any used model has unknown pricing."""
        with self._lock:
            total = 0.0
            for model, c in self.by_model.items():
                rates = _pricing_for(model)
                if rates is None:
                    return None
                p_in, p_out, p_write, p_read = rates
                total += (
                    c["input_tokens"] * p_in
                    + c["output_tokens"] * p_out
                    + c["cache_creation_input_tokens"] * p_write
                    + c["cache_read_input_tokens"] * p_read
                ) / 1_000_000
            return total

    def summary(self) -> str:
        """One human-readable line for the end of a run."""
        t = self.totals()
        parts = (
            f"{self.calls} API calls: "
            f"{t['input_tokens']:,} in, {t['output_tokens']:,} out, "
            f"{t['cache_creation_input_tokens']:,} cache-write, "
            f"{t['cache_read_input_tokens']:,} cache-read tokens"
        )
        cost = self.estimated_cost_usd()
        if cost is not None:
            parts += f" (~${cost:.4f})"
        return parts

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
    usage: Optional[UsageTracker] = None,
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
                usage=usage,
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
    usage: Optional[UsageTracker] = None,
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

    if usage is not None:
        usage.record(model, getattr(response, "usage", None))

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
