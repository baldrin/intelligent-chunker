"""Shared test helpers: in-memory PDFs and a fake Anthropic client."""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List

from pypdf import PdfWriter


def make_pdf(num_pages: int) -> bytes:
    """Build a valid in-memory PDF with ``num_pages`` blank pages."""
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class WordCounter:
    """Deterministic token counter for tests: one token per whitespace word."""

    def count(self, text: str) -> int:
        return len(text.split())


class _FakeBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class FakeUsage:
    """Deterministic usage object matching anthropic.types.Usage fields."""

    def __init__(self):
        self.input_tokens = 100
        self.output_tokens = 10
        self.cache_creation_input_tokens = 5
        self.cache_read_input_tokens = 50


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any]):
        self.stop_reason = "end_turn"
        self.stop_details = None
        self.content = [_FakeBlock(json.dumps(payload))]
        self.usage = FakeUsage()


class _FakeStream:
    """Context-manager stand-in for client.messages.stream(...)."""

    def __init__(self, response: _FakeResponse):
        self._response = response

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def get_final_message(self) -> _FakeResponse:
        return self._response


class _FakeMessages:
    def __init__(self, payloads: List[Dict[str, Any]]):
        self._payloads = payloads
        self.calls: List[Dict[str, Any]] = []

    def _next(self, kwargs: Dict[str, Any]) -> _FakeResponse:
        self.calls.append(kwargs)
        # Cycle through canned payloads (last one repeats).
        idx = min(len(self.calls) - 1, len(self._payloads) - 1)
        return _FakeResponse(self._payloads[idx])

    def stream(self, **kwargs: Any) -> _FakeStream:
        return _FakeStream(self._next(kwargs))

    def create(self, **kwargs: Any) -> _FakeResponse:
        return self._next(kwargs)


class FakeClient:
    """Stands in for anthropic.Anthropic; returns canned structured payloads."""

    def __init__(self, payloads: List[Dict[str, Any]]):
        self.messages = _FakeMessages(payloads)
