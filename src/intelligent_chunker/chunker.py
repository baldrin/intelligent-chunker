"""Pass 2 -- context-aware chunking.

For each section in the global map, re-read just that section's pages *with the
map as context* and have the model emit coherent, boundary-respecting chunks.
A deterministic token guard then enforces the embedder's hard limit, so the
intelligent chunking stays model-driven but nothing is ever silently truncated.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .config import ChunkerConfig
from .llm import structured_call
from .models import Chunk, DocumentProfile, Section
from .pdf_io import document_block, encoded_size, slice_to_fit
from .tokenizer import TokenCounter

logger = logging.getLogger(__name__)

# Native PDF blocks cap at 100 pages on 200K-context models (e.g. Haiku 4.5).
_FULL_DOC_PAGE_LIMIT = 100

_PASS2_RULES = (
    "Extract the section's text and split it into coherent, self-contained "
    "chunks suitable for embedding. Rules: (1) never split mid-word or "
    "mid-sentence; (2) break only at meaningful boundaries -- paragraphs, "
    "sub-headings, list items; (3) each chunk should stand on its own; (4) "
    "preserve wording faithfully -- do not summarize or invent text; (5) for "
    "every chunk, set page_start/page_end to the physical page(s) its text "
    "appears on, counting the FIRST page you were given as page 1. Use the "
    "global map to resolve references and pick relevant keywords."
)

PASS2_SYSTEM = (
    "You are an expert at preparing benefits/SPD documents for retrieval. "
    "You are given ONE section of a document (as a PDF of just its pages) plus "
    "a global map of the whole document for context. " + _PASS2_RULES + "\n\n"
    "IMPORTANT -- scope: the PDF pages you receive may include the tail of the "
    "previous section or the start of the next one (sections can share a "
    "page). Extract ONLY the content that belongs to the CURRENT SECTION named "
    "below. Skip any text that belongs to an adjacent section."
)

PASS2_SYSTEM_FULL_DOC = (
    "You are an expert at preparing benefits/SPD documents for retrieval. "
    "You are given the FULL document as a PDF plus a global map of it for "
    "context. Work on ONE section at a time, identified by name and physical "
    "page range below. " + _PASS2_RULES + "\n\n"
    "IMPORTANT -- scope: extract ONLY the content that belongs to the CURRENT "
    "SECTION named below (its pages may share a page with adjacent sections). "
    "Skip any text that belongs to other sections."
)

_CHUNKS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "cross_references": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "page_start": {"type": "integer"},
                    "page_end": {"type": "integer"},
                },
                "required": [
                    "text",
                    "keywords",
                    "cross_references",
                    "page_start",
                    "page_end",
                ],
            },
        }
    },
    "required": ["chunks"],
}


def _global_context(profile: DocumentProfile, section: Section) -> str:
    """Compact context block describing the document and the current section."""
    lines = ["DOCUMENT MAP (for context):"]
    if profile.title:
        lines.append(f"- Title: {profile.title}")
    if profile.doc_type:
        lines.append(f"- Type: {profile.doc_type}")
    if profile.plan_name:
        lines.append(f"- Plan: {profile.plan_name}")
    if profile.sponsor:
        lines.append(f"- Sponsor: {profile.sponsor}")
    if profile.effective_dates:
        lines.append(f"- Effective dates: {', '.join(profile.effective_dates)}")
    if profile.sections:
        outline = "; ".join(s.title for s in profile.sections)
        lines.append(f"- Section outline: {outline}")
    if profile.glossary:
        terms = "; ".join(
            f"{g.term}: {g.definition}" for g in profile.glossary[:40]
        )
        lines.append(f"- Defined terms: {terms}")
    lines.append("")
    lines.append("CURRENT SECTION:")
    lines.append(f"- Title: {section.title}")
    lines.append(f"- Type: {section.section_type}")
    if section.summary:
        lines.append(f"- Summary: {section.summary}")
    return "\n".join(lines)


def _sizing_instruction(config: ChunkerConfig) -> str:
    return (
        "Split THIS section into chunks. Make chunks SUBSTANTIAL: aim for "
        f"about {config.target_tokens} tokens each by keeping related "
        "paragraphs, list items, and sub-points together in one chunk. Do "
        "not emit a separate chunk for every sentence or short paragraph. "
        "Start a new chunk only at a real topic boundary or when nearing "
        f"the target, and never exceed {config.max_tokens} tokens. Return "
        "JSON matching the schema."
    )


def _normalize_chunk_pages(
    raw_chunks: List[Dict[str, Any]],
    offset: int,
    sec_start: int,
    sec_end: int,
) -> List[Dict[str, Any]]:
    """Shift model-reported chunk pages to absolute and clamp to the section.

    The model numbers pages relative to the PDF it was given (first page = 1),
    so sliced sections need ``offset`` added. A missing or nonsensical range
    falls back to the whole section's range -- provenance degrades to what we
    guaranteed before per-chunk pages existed, never to garbage.
    """
    for rc in raw_chunks:
        try:
            ps = int(rc.get("page_start", 0)) + offset
            pe = int(rc.get("page_end", 0)) + offset
        except (TypeError, ValueError):
            ps, pe = 0, 0
        if ps < sec_start or ps > sec_end or pe < ps:
            ps, pe = sec_start, sec_end
        else:
            pe = min(pe, sec_end)
        rc["page_start"], rc["page_end"] = ps, pe
    return raw_chunks


def chunk_section(
    client: Any,
    config: ChunkerConfig,
    pdf_bytes: bytes,
    profile: DocumentProfile,
    section: Section,
    full_doc_block: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Ask the model for this section's chunks (raw dicts, pre token-guard).

    When ``full_doc_block`` is given (cached full-document mode), every call
    reuses the same prompt-cached PDF block and only the trailing instruction
    varies, so calls after the first read the document at ~0.1x input price.
    Otherwise the section's pages are sliced out (split further if a slice
    would exceed the request-size budget) and sent per call.
    """
    page_start = max(1, min(section.page_start, profile.page_count))
    page_end = max(page_start, min(section.page_end, profile.page_count))

    if full_doc_block is not None:
        instruction = (
            _global_context(profile, section)
            + f"\n- Physical pages: {page_start}-{page_end}\n\n"
            + _sizing_instruction(config)
        )
        raw = structured_call(
            client,
            model=config.pass2_model,
            system=PASS2_SYSTEM_FULL_DOC,
            content=[full_doc_block, {"type": "text", "text": instruction}],
            schema=_CHUNKS_SCHEMA,
            max_tokens=config.max_output_tokens,
            repair_attempts=config.max_repair_attempts,
            transient_retries=config.max_transient_retries,
        )
        # Full-document mode: the model saw the whole PDF, pages are absolute.
        return _normalize_chunk_pages(
            raw.get("chunks", []), 0, page_start, page_end
        )

    instruction = _global_context(profile, section) + "\n\n" + _sizing_instruction(
        config
    )
    chunks: List[Dict[str, Any]] = []
    # Usually one slice; oversized (e.g. scanned) sections split to fit the
    # platform request budget, and each piece is chunked separately.
    for batch in slice_to_fit(
        pdf_bytes, page_start, page_end, config.max_encoded_request_bytes
    ):
        content = [
            document_block(batch.pdf_bytes),  # citations off (structured outputs)
            {"type": "text", "text": instruction},
        ]
        raw = structured_call(
            client,
            model=config.pass2_model,
            system=PASS2_SYSTEM,
            content=content,
            schema=_CHUNKS_SCHEMA,
            max_tokens=config.max_output_tokens,
            repair_attempts=config.max_repair_attempts,
            transient_retries=config.max_transient_retries,
        )
        # Sliced mode: the model saw only this slice, so its page 1 is the
        # slice's first absolute page.
        chunks.extend(
            _normalize_chunk_pages(
                raw.get("chunks", []),
                batch.page_offset,
                batch.page_start,
                batch.page_end,
            )
        )
    return chunks


def chunk_document(
    client: Any,
    config: ChunkerConfig,
    pdf_bytes: bytes,
    profile: DocumentProfile,
    counter: TokenCounter,
    on_section: Optional[Callable[[List[Chunk]], None]] = None,
) -> List[Chunk]:
    """Run Pass 2 over every section and return density-packed chunks.

    Per section: get the model's chunks, split any that exceed the hard token
    ceiling, then greedily pack consecutive pieces up to the target size so
    chunks are substantial rather than tiny. Packing never crosses a section.

    ``on_section`` (if given) receives each section's finished chunks as soon
    as they exist, so callers can persist incrementally -- a failure partway
    through a long run then costs only the unfinished sections.
    """
    # Never pack past the hard ceiling even if target is misconfigured above it.
    pack_target = min(config.target_tokens, config.max_tokens)

    # Cached full-document mode: when the whole PDF fits both the native PDF
    # page cap and the request budget, send the *same* document block (with a
    # cache breakpoint) on every section call. Only the trailing instruction
    # varies, so calls 2..N read the document from the prompt cache instead of
    # re-paying full input price per section. Otherwise fall back to slicing
    # each section's pages.
    full_doc_block: Optional[Dict[str, Any]] = None
    if (
        profile.page_count <= _FULL_DOC_PAGE_LIMIT
        and encoded_size(pdf_bytes) <= config.max_encoded_request_bytes
    ):
        full_doc_block = document_block(pdf_bytes)
        full_doc_block["cache_control"] = {"type": "ephemeral"}
        logger.info("Pass 2: using cached full-document mode")

    chunks: List[Chunk] = []
    index = 0
    for section in profile.sections:
        try:
            raw_chunks = chunk_section(
                client, config, pdf_bytes, profile, section,
                full_doc_block=full_doc_block,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Pass 2 failed on section {section.title!r} "
                f"(pages {section.page_start}-{section.page_end}): {exc}"
            ) from exc
        if not raw_chunks:
            logger.warning(
                "Section %r (pages %d-%d) produced no chunks",
                section.title,
                section.page_start,
                section.page_end,
            )

        # 1) Normalize + enforce the hard token ceiling, into ordered pieces.
        pieces: List[Dict[str, Any]] = []
        for rc in raw_chunks:
            text = (rc.get("text") or "").strip()
            if not text:
                continue
            for sub in enforce_token_limit(text, config.max_tokens, counter):
                pieces.append(
                    {
                        "text": sub,
                        "keywords": list(rc.get("keywords", [])),
                        "cross_references": list(rc.get("cross_references", [])),
                        # Splits inherit the parent chunk's page range.
                        "page_start": rc.get("page_start"),
                        "page_end": rc.get("page_end"),
                    }
                )

        # 2) Pack small pieces toward the target density.
        section_chunks: List[Chunk] = []
        for pc in pack_chunks(pieces, pack_target, counter):
            section_chunks.append(
                Chunk(
                    text=pc["text"],
                    source_file=profile.source_file,
                    chunk_index=index,
                    section_title=section.title,
                    section_type=section.section_type,
                    section_summary=section.summary,
                    page_start=pc.get("page_start") or section.page_start,
                    page_end=pc.get("page_end") or section.page_end,
                    keywords=pc["keywords"],
                    cross_references=pc["cross_references"],
                    token_count=counter.count(pc["text"]),
                )
            )
            index += 1
        if on_section is not None:
            on_section(section_chunks)
        chunks.extend(section_chunks)
    return chunks


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def pack_chunks(
    pieces: List[Dict[str, Any]], target_tokens: int, counter: TokenCounter
) -> List[Dict[str, Any]]:
    """Greedily merge consecutive pieces up to ``target_tokens``.

    Pieces are joined with a blank line (preserving paragraph boundaries),
    their keyword/cross-reference lists are unioned, and their page ranges
    merge to the min/max span. A piece already at/over the target stands
    alone -- packing only ever combines, never splits, so it cannot push a
    chunk past the hard ceiling the caller already enforced.
    """
    result: List[Dict[str, Any]] = []
    buf_text: List[str] = []
    buf_kw: List[str] = []
    buf_xref: List[str] = []
    buf_ps: List[int] = []
    buf_pe: List[int] = []

    def flush() -> None:
        if buf_text:
            result.append(
                {
                    "text": "\n\n".join(buf_text),
                    "keywords": _dedupe_preserve_order(buf_kw),
                    "cross_references": _dedupe_preserve_order(buf_xref),
                    "page_start": min(buf_ps) if buf_ps else None,
                    "page_end": max(buf_pe) if buf_pe else None,
                }
            )
            buf_text.clear()
            buf_kw.clear()
            buf_xref.clear()
            buf_ps.clear()
            buf_pe.clear()

    for piece in pieces:
        text = piece["text"]
        if buf_text:
            candidate = "\n\n".join(buf_text + [text])
            if counter.count(candidate) > target_tokens:
                flush()
        buf_text.append(text)
        buf_kw.extend(piece.get("keywords", []))
        buf_xref.extend(piece.get("cross_references", []))
        if piece.get("page_start"):
            buf_ps.append(int(piece["page_start"]))
        if piece.get("page_end"):
            buf_pe.append(int(piece["page_end"]))
    flush()
    return result


# --- deterministic token guard (pure; unit-tested without the API) ----------

_segmenter = None


def _segment_sentences(text: str) -> List[str]:
    """Split into sentences with pysbd (handles abbreviations cleanly)."""
    global _segmenter
    if _segmenter is None:
        import pysbd

        _segmenter = pysbd.Segmenter(language="en", clean=False)
    parts = [s.strip() for s in _segmenter.segment(text)]
    return [s for s in parts if s]


def _split_long_sentence(
    sentence: str, max_tokens: int, counter: TokenCounter
) -> List[str]:
    """Last resort: split an over-length sentence at whitespace only.

    Words are never broken -- we pack whole whitespace-separated tokens up to
    the limit.
    """
    words = sentence.split()
    pieces: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and counter.count(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
        if len(current) == 1 and counter.count(current[0]) > max_tokens:
            # A single unbreakable token (long URL, pasted table row, ID
            # string) over the limit: emit it alone rather than break
            # mid-word. This is the one case the <= max_tokens guarantee
            # cannot hold; the embedder will truncate it.
            logger.warning(
                "Single unbreakable token exceeds max_tokens=%d: %.60r...",
                max_tokens,
                current[0],
            )
            pieces.append(current[0])
            current = []
    if current:
        pieces.append(" ".join(current))
    return pieces


def enforce_token_limit(
    text: str, max_tokens: int, counter: TokenCounter
) -> List[str]:
    """Guarantee every returned piece is <= max_tokens.

    Splits only at sentence boundaries (and, for a single over-length
    sentence, at whitespace). A chunk already within the limit is returned
    unchanged. The single exception to the guarantee: one unbreakable
    whitespace-free token longer than the limit is emitted as-is (with a
    warning) rather than broken mid-word.
    """
    if counter.count(text) <= max_tokens:
        return [text]

    pieces: List[str] = []
    current: List[str] = []

    def flush() -> None:
        if current:
            pieces.append(" ".join(current))
            current.clear()

    for sentence in _segment_sentences(text):
        if counter.count(sentence) > max_tokens:
            flush()
            pieces.extend(_split_long_sentence(sentence, max_tokens, counter))
            continue
        candidate = " ".join(current + [sentence])
        if current and counter.count(candidate) > max_tokens:
            flush()
            current.append(sentence)
        else:
            current.append(sentence)
    flush()
    return pieces or [text]
