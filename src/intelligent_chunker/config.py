"""Pipeline configuration.

One dataclass holds every knob: per-pass model ids, batching, and the token
ceiling enforced against the eventual embedder (GTE-large v1.5).
"""

from __future__ import annotations

from dataclasses import dataclass

# Claude Haiku 4.5 -- cheap/fast, supports native PDF document blocks (text +
# vision), structured outputs, and citations. Used for both passes by default.
DEFAULT_MODEL = "claude-haiku-4-5"

# GTE-large v1.5 allows 8192 tokens, but retrieval quality is better with
# smaller chunks. This is the *hard ceiling* the deterministic token guard
# enforces; the model is asked to aim well below it.
DEFAULT_MAX_TOKENS = 1024

# What the model is asked to aim for per chunk (CLI shares this default).
DEFAULT_TARGET_TOKENS = 512

# Per-request payload budget in MB, measured against the base64-encoded PDF
# block (the dominant part of a request). Platform request-size limits:
#   - Anthropic API / Azure AI Foundry: 32 MB per request
#   - Databricks model serving:         ~4 MB per request
# The default leaves headroom under the 32 MB platforms; drop this to ~3 if
# calls are routed through Databricks model serving.
DEFAULT_MAX_REQUEST_MB = 25.0

# Pages per Pass 1 batch for long documents (CLI shares this default).
DEFAULT_MAX_PAGES_PER_BATCH = 50

# Hugging Face id whose tokenizer matches the eventual embedder, so chunk
# sizing reflects exactly what the embedder will see.
GTE_TOKENIZER_ID = "Alibaba-NLP/gte-large-en-v1.5"


@dataclass
class ChunkerConfig:
    # Models (per pass, so Pass 1 can later use a stronger model if desired).
    pass1_model: str = DEFAULT_MODEL
    pass2_model: str = DEFAULT_MODEL

    # Token budget for model calls. Pass 1 can emit a large profile (many
    # sections + glossary) for long documents; streamed so this stays safe.
    max_output_tokens: int = 16000

    # Chunk sizing (tokens), enforced by the deterministic guard.
    max_tokens: int = DEFAULT_MAX_TOKENS
    target_tokens: int = DEFAULT_TARGET_TOKENS

    # Batching for long documents. Native PDF blocks cap at 100 pages for
    # 200K-context models; scanned pages cost more tokens, so keep batches
    # modest. One page of overlap keeps sections that straddle a boundary
    # from being lost.
    max_pages_per_batch: int = DEFAULT_MAX_PAGES_PER_BATCH
    batch_overlap_pages: int = 1

    # Request payload budget (see DEFAULT_MAX_REQUEST_MB for platform limits).
    # Batches/section slices whose encoded PDF exceeds this are split further.
    max_request_mb: float = DEFAULT_MAX_REQUEST_MB

    # Pass 1 batches are independent, so long documents analyze in parallel.
    # Keep modest: each request carries a ~MB-scale PDF slice, and too much
    # concurrency just trades 429 retries for wall-clock time.
    pass1_concurrency: int = 4

    # Pass 2 section calls are independent too. In cached full-document mode
    # the first call runs alone to populate the prompt cache before fanning
    # out, so concurrency never forfeits the cached-read discount.
    pass2_concurrency: int = 4

    # Tokenizer.
    tokenizer_id: str = GTE_TOKENIZER_ID

    # Extra resilience on top of the SDK's built-in retries:
    # - max_repair_attempts: re-issue a request whose structured output came
    #   back unusable (truncated / unparseable / missing text).
    # - max_transient_retries: our own backoff for rate limits, 5xx, and
    #   network drops that survive the SDK's 2 default retries.
    max_repair_attempts: int = 1
    max_transient_retries: int = 2

    @property
    def max_encoded_request_bytes(self) -> int:
        """The payload budget in bytes, compared against base64-encoded size."""
        return int(self.max_request_mb * 1024 * 1024)
