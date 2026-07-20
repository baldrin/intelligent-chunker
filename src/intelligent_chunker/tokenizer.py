"""Pluggable token counting.

Chunk sizes must match what the *embedder* sees, so the default counter loads
the GTE-large v1.5 tokenizer. When ``tokenizers`` isn't installed we fall back
to a conservative heuristic and warn -- good enough to keep the pipeline
running, but install the real tokenizer before trusting chunk sizes.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

logger = logging.getLogger(__name__)


class TokenCounter(Protocol):
    """Anything that can count tokens for a string."""

    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Cheap, dependency-free approximation.

    Counts word-ish pieces and punctuation. Subword tokenizers usually emit
    *more* tokens than whitespace words, so we scale up a little to avoid
    under-counting (which would let chunks slip over the embedder limit).
    """

    _TOKEN_RE = re.compile(r"\w+|[^\w\s]")
    _SCALE = 1.3

    def count(self, text: str) -> int:
        pieces = self._TOKEN_RE.findall(text)
        return int(len(pieces) * self._SCALE) + 1


class HFTokenCounter:
    """Exact counts via the embedder's own Hugging Face tokenizer."""

    def __init__(self, tokenizer_id: str):
        from tokenizers import Tokenizer  # imported lazily

        # ``from_pretrained`` pulls the tokenizer.json for the model id.
        self._tok = Tokenizer.from_pretrained(tokenizer_id)
        # The GTE tokenizer ships with padding/truncation enabled to its max
        # length -- with those on, ``encode`` pads every input to 512 ids and
        # the count is meaningless. Disable both so we get the true length.
        self._tok.no_padding()
        self._tok.no_truncation()

    def count(self, text: str) -> int:
        return len(self._tok.encode(text).ids)


def get_token_counter(tokenizer_id: str) -> TokenCounter:
    """Return the exact GTE counter if available, else the heuristic."""
    try:
        return HFTokenCounter(tokenizer_id)
    except Exception as exc:  # ImportError, network, missing model, ...
        logger.warning(
            "Falling back to heuristic token counting (could not load "
            "tokenizer %r: %s). Install the 'tokenizer' extra and ensure "
            "network access for exact GTE sizing.",
            tokenizer_id,
            exc,
        )
        return HeuristicTokenCounter()
