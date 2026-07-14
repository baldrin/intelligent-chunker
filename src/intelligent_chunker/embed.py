"""Embedding interface -- STUB for this phase.

The chunker's job right now ends at producing chunks. Embedding is wired in a
later phase against an existing GTE-large v1.5 model behind an HTTP endpoint.
The protocol below is the contract that phase will implement; ``HTTPEmbedder``
is a deliberately-unfinished skeleton marking the integration point.
"""

from __future__ import annotations

import os
from typing import List, Optional, Protocol


class Embedder(Protocol):
    """Maps texts to fixed-length vectors. GTE-large v1.5 -> 1024 dims."""

    def embed(self, texts: List[str]) -> List[List[float]]: ...


class HTTPEmbedder:
    """Skeleton client for the GTE-large v1.5 HTTP endpoint.

    TODO(embedding phase): implement ``embed`` -- POST batches of ``texts`` to
    ``endpoint_url`` and parse the returned vectors. Left unimplemented on
    purpose so the chunker phase has no live embedding dependency.
    """

    def __init__(self, endpoint_url: Optional[str] = None):
        self.endpoint_url = endpoint_url or os.environ.get("GTE_ENDPOINT_URL")

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError(
            "HTTPEmbedder is a stub. Embedding is out of scope for the "
            "chunker phase; wire this up against the GTE-large v1.5 endpoint "
            "when adding the embedding step."
        )
