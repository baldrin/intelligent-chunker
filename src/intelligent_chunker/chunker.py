"""Pass 2 -- context-aware chunking.

For each section in the global map, re-read just that section's pages *with the
map as context* and have the model emit coherent, boundary-respecting chunks.
A deterministic token guard then enforces the embedder's hard limit, so the
intelligent chunking stays model-driven but nothing is ever silently truncated.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from .config import ChunkerConfig
from .fidelity import extract_page_texts, match_key
from .llm import UsageTracker, structured_call
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
    "preserve wording faithfully -- do not summarize or invent text, and "
    "NEVER add headings, titles, or labels of your own: every line of a "
    "chunk must be text that appears in the document; (5) for "
    "every chunk, set page_start/page_end to the physical page(s) its text "
    "appears on, counting the FIRST page you were given as page 1 -- IGNORE "
    "page numbers printed in headers, footers, or the text itself, which "
    "often differ from physical position; (6) never "
    "emit the same text twice -- each passage belongs in exactly one chunk, "
    "and when the text cross-references another subsection, keep the "
    "reference as written instead of copying the referenced text in. Use the "
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


# --- text-layer grounding of per-chunk page ranges ---------------------------
#
# The model's per-chunk page_start/page_end are self-reported and drift
# (observed: chunks labeled two pages off even inside a correct section
# range). When the PDF has a text layer, verify each chunk's edges against
# where its text physically appears.

# How many normalized characters of a chunk's head/tail to search for (~6-8
# words: long enough to be distinctive, short enough to sit on one page).
_CHUNK_MATCH_CHARS = 40

# Chunks whose whole normalized text is shorter than this are too generic to
# locate reliably (stray headings, table fragments).
_CHUNK_MATCH_MIN_CHARS = 15

# When an edge finds nothing inside the section's pages, retry this many
# pages beyond each end. Models drift page labels (printed page numbers often
# differ from physical position by several pages of unnumbered front matter),
# and drifted text sits just outside the claimed range, where the in-range
# search can't see it.
_GROUND_MARGIN_PAGES = 3

# Word-overlap fallback (order-immune) for when substring matching fails:
# extraction can scramble word order (hanging-indent lists, table layouts),
# which defeats contiguous matching while leaving the words themselves in the
# layer. Only words appearing on few pages document-wide count, so shared
# boilerplate vocabulary ("plan", "coverage", ...) can't produce false hits.
_RARE_WORD_MAX_DOC_PAGES = 3  # a word on <= this many pages is distinctive
_RARE_WORDS_MIN = 5  # need at least this many distinctive words to try
_WORD_FALLBACK_MIN_COVERED = 0.5  # located pages must hold half the rare words
_WORD_FALLBACK_MAX_SPAN = 3  # a single piece never spans more pages than this
_ADJACENT_PAGE_SCORE_FRACTION = 0.25  # extend to neighbors with >= this share

_WORDISH_RE = re.compile(r"[a-z0-9]{4,}")


def _locate_by_words(
    text: str,
    page_words: List[set],
    doc_freq: Dict[str, int],
    lo: int,
    hi: int,
) -> Optional[tuple]:
    """Locate ``text`` in pages [lo, hi] by its distinctive words, or None."""
    rare = {
        w
        for w in _WORDISH_RE.findall(text.lower())
        if 0 < doc_freq.get(w, 0) <= _RARE_WORD_MAX_DOC_PAGES
    }
    if len(rare) < _RARE_WORDS_MIN:
        return None
    scores = {p: len(rare & page_words[p - 1]) for p in range(lo, hi + 1)}
    best = max(scores, key=lambda p: scores[p])
    if scores[best] == 0:
        return None
    # Grow a contiguous run around the best page: a piece can straddle pages,
    # so neighbors holding a meaningful share of the rare words belong too.
    floor = max(1, int(scores[best] * _ADJACENT_PAGE_SCORE_FRACTION))
    start = end = best
    while (
        start - 1 >= lo
        and scores[start - 1] >= floor
        and end - start + 1 < _WORD_FALLBACK_MAX_SPAN
    ):
        start -= 1
    while (
        end + 1 <= hi
        and scores[end + 1] >= floor
        and end - start + 1 < _WORD_FALLBACK_MAX_SPAN
    ):
        end += 1
    covered = set()
    for p in range(start, end + 1):
        covered |= rare & page_words[p - 1]
    if len(covered) / len(rare) < _WORD_FALLBACK_MIN_COVERED:
        return None
    return start, end


def ground_chunk_pages(
    pieces: List[Dict[str, Any]],
    norm_pages: List[str],
    sec_start: int,
    sec_end: int,
    raw_pages: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Snap each piece's page range to where its text appears in the section.

    A piece's first/last ``_CHUNK_MATCH_CHARS`` normalized characters are
    searched across the section's pages (``norm_pages`` is the full
    document's normalized text layer, ``match_key``-style). An edge found on
    exactly one page pins that end of the range; multiple hits keep the
    model's value. Zero hits retry with ``_GROUND_MARGIN_PAGES`` extra pages
    on each end, so text whose label drifted out of the section can still be
    located; the narrow pass stays first because the section range is also
    what disambiguates text repeated elsewhere in the document.

    When ``raw_pages`` (the unnormalized text layer) is given and neither
    edge matched at all, a word-overlap fallback locates the piece by its
    distinctive words -- immune to the extraction word-scrambling (hanging
    indents, tables) that defeats substring matching.

    Contradictory hits (start after end) distrust both. Pieces are mutated
    in place and returned.
    """
    lo = max(1, sec_start)
    hi = min(sec_end, len(norm_pages))
    wide_lo = max(1, sec_start - _GROUND_MARGIN_PAGES)
    wide_hi = min(sec_end + _GROUND_MARGIN_PAGES, len(norm_pages))

    page_words: Optional[List[set]] = None
    doc_freq: Dict[str, int] = {}
    if raw_pages is not None:
        page_words = [set(_WORDISH_RE.findall(t.lower())) for t in raw_pages]
        for ws in page_words:
            for w in ws:
                doc_freq[w] = doc_freq.get(w, 0) + 1

    def locate(needle: str) -> Optional[int]:
        hits = [p for p in range(lo, hi + 1) if needle in norm_pages[p - 1]]
        if not hits:
            hits = [
                p
                for p in range(wide_lo, wide_hi + 1)
                if needle in norm_pages[p - 1]
            ]
        return hits[0] if len(hits) == 1 else None

    for pc in pieces:
        key = match_key(pc.get("text") or "")
        if len(key) < _CHUNK_MATCH_MIN_CHARS:
            continue
        prefix = key[:_CHUNK_MATCH_CHARS]
        suffix = key[-_CHUNK_MATCH_CHARS:]
        ps = locate(prefix)
        pe = locate(suffix)
        if ps is None and pe is None and page_words is not None:
            span = _locate_by_words(
                pc.get("text") or "", page_words, doc_freq, wide_lo, wide_hi
            )
            if span is not None:
                ps, pe = span
        if ps is not None and pe is not None and ps > pe:
            continue  # both matched but out of order: trust neither
        old = (pc.get("page_start"), pc.get("page_end"))
        if ps is not None:
            pc["page_start"] = ps
        if pe is not None:
            pc["page_end"] = pe
        # Keep the invariant when only one edge was grounded and the model's
        # other edge contradicts it.
        cur_ps, cur_pe = pc.get("page_start"), pc.get("page_end")
        if cur_ps is not None and cur_pe is not None and cur_ps > cur_pe:
            if ps is not None:
                pc["page_end"] = ps
            else:
                pc["page_start"] = pe
        if (pc.get("page_start"), pc.get("page_end")) != old:
            logger.info(
                "Grounding: chunk pages %s -> (%s, %s) from the text layer",
                old,
                pc.get("page_start"),
                pc.get("page_end"),
            )
    return pieces


# --- table-of-contents noise -------------------------------------------------
#
# TOC pages come through the text layer as heading + dot-leader + page-number
# runs ("II. PARTICIPATION......... 4"). The dot runs tokenize horribly (the
# sample's TOC chunks recorded 751-1024 tokens for ~40 words of text), and the
# entries themselves are pure retrieval noise.

_DOT_LEADER_RE = re.compile(r"\.{4,}")  # 4+: never touches a real "..." ellipsis
# One TOC entry: a leader run, optional stray dots/space artifacts, page number.
_TOC_ENTRY_RE = re.compile(r"\.{4,}[\s.]*\d+")
_TOC_MIN_ENTRIES = 3
# TOC text is almost nothing but entries; real prose that happens to contain a
# few dotted rows has far more words per entry than a heading + page number.
_TOC_MAX_WORDS_PER_ENTRY = 12


def strip_dot_leaders(text: str) -> str:
    """Collapse dot-leader runs to a space and tidy the leftover spacing."""
    cleaned = _DOT_LEADER_RE.sub(" ", text)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def is_toc_text(text: str) -> bool:
    """Whether ``text`` is a table-of-contents fragment (mostly entries)."""
    entries = _TOC_ENTRY_RE.findall(text)
    if len(entries) < _TOC_MIN_ENTRIES:
        return False
    words = len(strip_dot_leaders(text).split())
    return words <= len(entries) * _TOC_MAX_WORDS_PER_ENTRY


def clean_raw_chunks(
    raw_chunks: List[Dict[str, Any]], section: Section
) -> List[Dict[str, Any]]:
    """Strip dot leaders everywhere; drop TOC chunks in unmapped sections.

    TOC pages land in synthetic "Unmapped pages" sections (Pass 1 assigns
    them to no section), which is where dropping is safe. In a mapped section
    a dotted list might be real content (a fund lineup, a rate table), so it
    is only cleaned, never dropped.
    """
    out: List[Dict[str, Any]] = []
    for rc in raw_chunks:
        text = rc.get("text") or ""
        if section.section_type == "unmapped" and is_toc_text(text):
            logger.info(
                "Section %r: dropping a table-of-contents chunk",
                section.title,
            )
            continue
        stripped = strip_dot_leaders(text)
        if stripped != text:
            rc = {**rc, "text": stripped}
        out.append(rc)
    return out


# Paragraphs shorter than this many words are exempt from deduplication:
# table headers and schedule rows ("Years of Service | Vesting Percentage",
# "less than 1 | 100.00") legitimately repeat within a section.
_DEDUP_MIN_PARAGRAPH_WORDS = 25


def _dedup_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def dedupe_raw_chunks(
    raw_chunks: List[Dict[str, Any]], section_title: str
) -> List[Dict[str, Any]]:
    """Drop text the model emitted more than once within a section.

    Observed failure modes despite the prompt rules: (a) a chunk re-emitted
    verbatim, and (b) a chunk padded with paragraphs copied from an earlier
    chunk (the model expanding a cross-reference by restating the referenced
    text). Whole-chunk duplicates are dropped outright; duplicated paragraphs
    of at least ``_DEDUP_MIN_PARAGRAPH_WORDS`` words are removed while the
    chunk's own paragraphs survive. First occurrence always wins, so section
    reading order is preserved.
    """
    seen_chunks: set = set()
    seen_paragraphs: set = set()
    out: List[Dict[str, Any]] = []
    for rc in raw_chunks:
        text = (rc.get("text") or "").strip()
        if not text:
            continue
        if _dedup_key(text) in seen_chunks:
            logger.warning(
                "Section %r: dropping a chunk re-emitted verbatim",
                section_title,
            )
            continue
        kept: List[str] = []
        dropped = 0
        for para in re.split(r"\n\s*\n", text):
            if len(para.split()) >= _DEDUP_MIN_PARAGRAPH_WORDS:
                key = _dedup_key(para)
                if key in seen_paragraphs:
                    dropped += 1
                    continue
                seen_paragraphs.add(key)
            kept.append(para)
        if dropped:
            logger.warning(
                "Section %r: removed %d paragraph(s) duplicated from an "
                "earlier chunk",
                section_title,
                dropped,
            )
        if not any(p.strip() for p in kept):
            logger.warning(
                "Section %r: dropping a chunk left empty after paragraph "
                "dedup",
                section_title,
            )
            continue
        seen_chunks.add(_dedup_key(text))
        if dropped:
            rc = {**rc, "text": "\n\n".join(kept)}
        out.append(rc)
    return out


def chunk_section(
    client: Any,
    config: ChunkerConfig,
    pdf_bytes: bytes,
    profile: DocumentProfile,
    section: Section,
    full_doc_block: Optional[Dict[str, Any]] = None,
    usage: Optional[UsageTracker] = None,
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
            usage=usage,
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
            usage=usage,
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
    usage: Optional[UsageTracker] = None,
    sections: Optional[List[Section]] = None,
    start_index: int = 0,
) -> List[Chunk]:
    """Run Pass 2 over every section and return density-packed chunks.

    Per section: get the model's chunks, drop text emitted more than once
    (see ``dedupe_raw_chunks``), split any chunk that exceeds the hard token
    ceiling, then greedily pack consecutive pieces up to the target size so
    chunks are substantial rather than tiny. Packing never crosses a section.

    Section calls are independent, so they fan out across
    ``config.pass2_concurrency`` threads; results are consumed strictly in
    section order, so chunk indexing and incremental persistence behave
    exactly as in a sequential run.

    ``on_section`` (if given) receives each section's finished chunks as soon
    as they exist, so callers can persist incrementally -- a failure partway
    through a long run then costs only the unfinished sections.

    ``sections``/``start_index`` support resuming: chunk only the given
    subset (default: all of ``profile.sections``) while keeping the full
    profile as model context, numbering chunks from ``start_index``.
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

    sections = list(profile.sections) if sections is None else list(sections)

    # Normalized text layer for per-chunk page grounding (empty for scanned
    # PDFs, in which case the model's self-reported pages stand).
    raw_pages = extract_page_texts(pdf_bytes)
    norm_pages = [match_key(t) for t in raw_pages]
    has_text_layer = any(norm_pages)

    # Shared across worker threads so one context overflow downgrades the
    # whole run: page count alone can't predict token cost (PDF pages bill
    # text + a per-page image), so a dense document can pass the 100-page
    # cap yet overflow the context window. The API's rejection is the only
    # reliable signal, and it would repeat on every full-document call.
    state = {"full_doc_block": full_doc_block}

    def fetch(section: Section) -> List[Dict[str, Any]]:
        try:
            block = state["full_doc_block"]
            if block is not None:
                try:
                    return chunk_section(
                        client, config, pdf_bytes, profile, section,
                        full_doc_block=block, usage=usage,
                    )
                except Exception as exc:
                    if "prompt is too long" not in str(exc):
                        raise
                    logger.warning(
                        "Pass 2: the full document overflows the model's "
                        "context window; falling back to per-section slices "
                        "(prompt caching disabled)"
                    )
                    state["full_doc_block"] = None
            return chunk_section(
                client, config, pdf_bytes, profile, section,
                full_doc_block=None, usage=usage,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Pass 2 failed on section {section.title!r} "
                f"(pages {section.page_start}-{section.page_end}): {exc}"
            ) from exc

    workers = min(max(1, config.pass2_concurrency), len(sections) or 1)
    if workers <= 1:
        results: Iterable[List[Dict[str, Any]]] = (fetch(s) for s in sections)
    elif full_doc_block is not None:
        # The first call writes the document into the prompt cache; only fan
        # out once it exists, so the remaining calls all read it instead of
        # each re-paying (and re-creating) the cache entry.
        def staggered() -> Iterator[List[Dict[str, Any]]]:
            yield fetch(sections[0])
            with ThreadPoolExecutor(max_workers=workers) as pool:
                yield from pool.map(fetch, sections[1:])

        results = staggered()
    else:

        def fanned_out() -> Iterator[List[Dict[str, Any]]]:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                yield from pool.map(fetch, sections)

        results = fanned_out()

    chunks: List[Chunk] = []
    index = start_index
    for pos, (section, raw_chunks) in enumerate(zip(sections, results), start=1):
        logger.info(
            "Pass 2: section %d/%d done: %s", pos, len(sections), section.title
        )
        if not raw_chunks:
            logger.warning(
                "Section %r (pages %d-%d) produced no chunks",
                section.title,
                section.page_start,
                section.page_end,
            )

        # 1) Clean TOC noise and duplicated text, then normalize + enforce
        #    the hard token ceiling, into ordered pieces.
        pieces: List[Dict[str, Any]] = []
        cleaned = clean_raw_chunks(raw_chunks, section)
        for rc in dedupe_raw_chunks(cleaned, section.title):
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

        # 2) Pack small pieces toward the target density, then verify the
        #    packed pages against where the text physically appears.
        packed = pack_chunks(pieces, pack_target, counter)
        if has_text_layer:
            packed = ground_chunk_pages(
                packed,
                norm_pages,
                max(1, min(section.page_start, profile.page_count)),
                max(section.page_start, min(section.page_end, profile.page_count)),
                raw_pages=raw_pages,
            )
        section_chunks: List[Chunk] = []
        for pc in packed:
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
