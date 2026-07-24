"""Pass 1 -- global document analysis (the "map").

Read the whole document (batched if long) and build a ``DocumentProfile``: the
section outline with page ranges, document-wide metadata, a glossary of defined
terms, and cross-references. Pass 2 consumes this so every chunk is produced
with full-document context instead of a blind linear read.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from .config import ChunkerConfig
from .fidelity import extract_page_texts, match_key
from .llm import UsageTracker, structured_call
from .models import DocumentProfile, GlossaryTerm, Section
from .pdf_io import PageBatch, document_block, iter_batches

logger = logging.getLogger(__name__)

PASS1_SYSTEM = (
    "You are an expert at analyzing benefits and insurance documents, "
    "especially Summary Plan Description (SPD) documents. You will be given a "
    "PDF (it may be digital text or scanned images). Read the entire document "
    "and produce a structured global map of it: document-wide metadata, the "
    "full ordered outline of sections with their page ranges, a glossary of "
    "terms the document explicitly defines, and notable cross-references "
    "between sections. Be thorough and faithful to the document -- do not "
    "invent sections or definitions.\n\n"
    "Cross-references must be actual references -- one section, statute, "
    "regulation, or attachment pointing to another -- written as complete "
    "standalone entries, not sentence fragments or general statements from "
    "the text. Keep the document's own wording for what is referenced: cite "
    "statute and section names exactly as printed (for example, never write "
    "'Internal Revenue Code Section 502(a)' when the document says 'Section "
    "502(a) of ERISA').\n\n"
    "CRITICAL -- how to number pages: number pages by their PHYSICAL position "
    "in this PDF. The first page you are given is page 1, the next is page 2, "
    "and so on. IGNORE any page numbers printed in the document's headers or "
    "footers -- those are often offset from the physical position (e.g. a "
    "cover and table of contents may push the printed numbering ahead). "
    "page_start and page_end must be physical page positions, counting from 1 "
    "at the first page provided."
)

# JSON Schema for one batch's partial profile.
_PROFILE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "doc_type": {"type": "string"},
        "title": {"type": "string"},
        "plan_name": {"type": "string"},
        "sponsor": {"type": "string"},
        "effective_dates": {"type": "array", "items": {"type": "string"}},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "section_type": {"type": "string"},
                    "summary": {"type": "string"},
                    "page_start": {"type": "integer"},
                    "page_end": {"type": "integer"},
                },
                "required": [
                    "title",
                    "section_type",
                    "summary",
                    "page_start",
                    "page_end",
                ],
            },
        },
        "glossary": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "term": {"type": "string"},
                    "definition": {"type": "string"},
                },
                "required": ["term", "definition"],
            },
        },
        "cross_references": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": [
        "doc_type",
        "title",
        "plan_name",
        "sponsor",
        "effective_dates",
        "sections",
        "glossary",
        "cross_references",
        "notes",
    ],
}

_PASS1_INSTRUCTION = (
    "Analyze this PDF and return its global map as JSON matching the required "
    "schema. Include every distinct section in reading order. Set "
    "page_start/page_end to PHYSICAL page positions (first page provided = 1; "
    "ignore printed footer/header page numbers). Capture document-wide "
    "metadata, defined terms, and cross-references."
)


def analyze_batch(
    client: Any,
    config: ChunkerConfig,
    batch: PageBatch,
    usage: Optional[UsageTracker] = None,
) -> Dict[str, Any]:
    """Analyze one page-range batch; section pages are offset to absolute."""
    content = [
        document_block(batch.pdf_bytes),  # citations off (structured outputs)
        {"type": "text", "text": _PASS1_INSTRUCTION},
    ]
    raw = structured_call(
        client,
        model=config.pass1_model,
        system=PASS1_SYSTEM,
        content=content,
        schema=_PROFILE_SCHEMA,
        max_tokens=config.max_output_tokens,
        repair_attempts=config.max_repair_attempts,
        transient_retries=config.max_transient_retries,
        usage=usage,
    )
    # Shift batch-relative page numbers to absolute document pages.
    for sec in raw.get("sections", []):
        sec["page_start"] = int(sec.get("page_start", 1)) + batch.page_offset
        sec["page_end"] = int(sec.get("page_end", 1)) + batch.page_offset
    return raw


def analyze_document(
    client: Any,
    config: ChunkerConfig,
    pdf_bytes: bytes,
    source_file: str,
    usage: Optional[UsageTracker] = None,
) -> DocumentProfile:
    """Run Pass 1 over the whole document and reconcile into one profile."""
    batches = iter_batches(
        pdf_bytes,
        config.max_pages_per_batch,
        config.batch_overlap_pages,
        max_encoded_bytes=config.max_encoded_request_bytes,
    )
    workers = min(max(1, config.pass1_concurrency), len(batches) or 1)
    if workers <= 1:
        partials = [analyze_batch(client, config, b, usage=usage) for b in batches]
    else:
        # Batches are independent; fan out. pool.map preserves batch order,
        # which reconcile relies on (first non-empty metadata wins, and batch
        # 1 holds the title page). The Anthropic client is thread-safe.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            partials = list(
                pool.map(
                    lambda b: analyze_batch(client, config, b, usage=usage),
                    batches,
                )
            )
    page_count = batches[-1].page_end if batches else 0
    return reconcile(
        partials,
        source_file=source_file,
        page_count=page_count,
        page_texts=extract_page_texts(pdf_bytes),
    )


# --- reconciliation (deterministic merge of per-batch partials) -------------


def _normalize_title(title: str) -> str:
    t = re.sub(r"\s+", " ", title.strip().lower())
    return t.strip(" .:-–—")


def _first_nonempty(values: List[str]) -> str:
    for v in values:
        if v and v.strip():
            return v.strip()
    return ""


def _dedupe_strings(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        key = re.sub(r"\s+", " ", v.strip().lower())
        if key and key not in seen:
            seen.add(key)
            out.append(v.strip())
    return out


def _sanitize_references(values: List[str]) -> List[str]:
    """Clean model-emitted cross-references before deduping.

    Observed defects: entries arriving with leading punctuation from a
    truncated sentence (", Section II describes ...") and entries that are
    empty once stripped. Content errors (a paraphrase or a misnamed statute)
    can't be fixed deterministically -- the Pass 1 prompt addresses those.
    """
    cleaned: List[str] = []
    for v in values:
        v = v.strip().lstrip(",;:.-–—— ").strip()
        if v:
            cleaned.append(v)
    return _dedupe_strings(cleaned)


def _merge_sections(raw_sections: List[Dict[str, Any]]) -> List[Section]:
    """Fold duplicates created by batch overlap, then order by page.

    Two sections fold together when their normalized titles match and their
    page ranges overlap or touch -- the case where the same heading is seen in
    two overlapping batches. The folded section spans the union of the ranges.
    Grouping by title (rather than only comparing page-adjacent neighbors)
    means a duplicate still folds even when a different section sorts between
    its two sightings.
    """
    sections = [Section.from_dict(s) for s in raw_sections]
    sections.sort(key=lambda s: (s.page_start, s.page_end))

    merged: List[Section] = []
    by_title: Dict[str, List[Section]] = {}
    for sec in sections:
        key = _normalize_title(sec.title)
        folded = False
        for prev in by_title.get(key, []):
            overlaps = (
                sec.page_start <= prev.page_end + 1
                and prev.page_start <= sec.page_end + 1
            )
            if overlaps:
                prev.page_start = min(prev.page_start, sec.page_start)
                prev.page_end = max(prev.page_end, sec.page_end)
                if len(sec.summary) > len(prev.summary):
                    prev.summary = sec.summary
                folded = True
                break
        if not folded:
            by_title.setdefault(key, []).append(sec)
            merged.append(sec)
    merged.sort(key=lambda s: (s.page_start, s.page_end))
    return merged


def _merge_glossary(raw_glossaries: List[List[Dict[str, Any]]]) -> List[GlossaryTerm]:
    seen = set()
    out: List[GlossaryTerm] = []
    for gl in raw_glossaries:
        for item in gl:
            term = GlossaryTerm.from_dict(item)
            key = term.term.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(term)
    return out


# --- text-layer grounding of section page ranges ----------------------------
#
# Despite the prompt, Pass 1 sometimes reports the document's *printed*
# footer page numbers instead of physical PDF pages (a cover page and TOC
# offset the two), which corrupts chunk citations, reorders sections, and
# makes per-section fidelity compare against the wrong pages. When the PDF
# has a text layer, we can fix this deterministically: find each section's
# heading in the per-page text and snap the outline to the pages where the
# headings physically appear.

# Headings shorter than this after normalization ("A.", "IV") are too likely
# to match by accident to be trusted.
_MATCH_MIN_KEY_CHARS = 6

# A page whose text matches this many distinct section headings is an index
# (table-of-contents) page: it matches *every* heading, so its hits would
# otherwise make every section ambiguous.
_INDEX_PAGE_MIN_HEADINGS = 3

# A heading whose match starts within this many normalized characters of the
# top of its page is treated as opening that page (so the previous section
# ends on the page before); deeper matches mean the page is shared with the
# previous section. The budget covers a running header line.
_TOP_OF_PAGE_CHARS = 120

# Word-fallback for headings whose extraction is scrambled: styled banners
# ("SCHEDULE OF BENEFITS -- HSA Plan") can extract with fused or reordered
# glyphs, which defeats contiguous matching while the words themselves remain
# in the layer. Only words appearing on few pages document-wide count, so
# ubiquitous vocabulary can't produce a false page.
_TITLE_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_TITLE_RARE_MAX_DOC_PAGES = 4  # a word on <= this many pages is distinctive
_TITLE_RARE_MIN_WORDS = 2  # need at least this many to trust the match


def _heading_hit(norm_page: str, key: str) -> Optional[int]:
    """Offset of ``key`` as a heading in a normalized page, or None.

    Occurrences immediately preceded by the word "section" are body
    cross-references ("... in Section III, Contributions."), not headings,
    and are skipped.
    """
    start = 0
    while True:
        off = norm_page.find(key, start)
        if off < 0:
            return None
        if not norm_page[:off].endswith("section"):
            return off
        start = off + 1


def ground_sections(
    sections: List[Section], page_texts: List[str]
) -> List[Section]:
    """Snap section page ranges to where their headings appear in the text layer.

    For every section whose heading text is found on exactly one non-index
    page, ``page_start`` snaps to that page. Ends are then derived from the
    following section's grounded start: a heading opening its page puts the
    previous section's end on the page before; a heading deeper in the page
    means the two sections share it. Ungrounded sections are only ever pulled
    back from a grounded neighbor's pages, never extended. Scanned PDFs (no
    text layer) and unmatched or ambiguous headings leave the model's ranges
    untouched, so grounding can only refine the outline, not degrade it.
    """
    norm_pages = [match_key(t) for t in page_texts]
    if not any(norm_pages):
        return sections  # scanned PDF: nothing to ground against

    # Every (section, page) heading hit, and how many sections hit each page.
    hits: Dict[int, List[Tuple[int, int]]] = {}  # sec idx -> [(page, offset)]
    page_hit_count: Dict[int, int] = {}
    for i, sec in enumerate(sections):
        key = match_key(sec.title)
        if len(key) < _MATCH_MIN_KEY_CHARS:
            continue
        for page, norm in enumerate(norm_pages, start=1):
            off = _heading_hit(norm, key)
            if off is not None:
                hits.setdefault(i, []).append((page, off))
                page_hit_count[page] = page_hit_count.get(page, 0) + 1

    index_pages = {
        page
        for page, count in page_hit_count.items()
        if count >= _INDEX_PAGE_MIN_HEADINGS
    }

    grounded: Dict[int, Tuple[int, int]] = {}  # sec idx -> (page, offset)
    for i, sec_hits in hits.items():
        content_hits = [h for h in sec_hits if h[0] not in index_pages]
        if len(content_hits) == 1:
            grounded[i] = content_hits[0]

    # Word-fallback for the still-ungrounded: the unique non-index page
    # holding every document-rare word of the title. Treated as opening its
    # page (a scrambled banner gives no usable offset).
    page_words = [set(_TITLE_WORD_RE.findall(t.lower())) for t in page_texts]
    doc_freq: Dict[str, int] = {}
    for ws in page_words:
        for w in ws:
            doc_freq[w] = doc_freq.get(w, 0) + 1
    for i, sec in enumerate(sections):
        if i in grounded or len(match_key(sec.title)) < _MATCH_MIN_KEY_CHARS:
            continue
        rare = {
            w
            for w in _TITLE_WORD_RE.findall(sec.title.lower())
            if 0 < doc_freq.get(w, 0) <= _TITLE_RARE_MAX_DOC_PAGES
        }
        if len(rare) < _TITLE_RARE_MIN_WORDS:
            continue
        pages = [
            p
            for p, ws in enumerate(page_words, start=1)
            if p not in index_pages and rare <= ws
        ]
        if len(pages) == 1:
            grounded[i] = (pages[0], 0)

    for i, (page, _off) in grounded.items():
        sec = sections[i]
        if sec.page_start != page:
            logger.warning(
                "Grounding: section %r page_start %d -> %d (heading found on "
                "physical page %d; Pass 1 likely reported printed page numbers)",
                sec.title,
                sec.page_start,
                page,
                page,
            )
            sec.page_start = page
        if sec.page_end < sec.page_start:
            sec.page_end = sec.page_start

    # Derive ends from each grounded section's start: the boundary it pins is
    # authoritative for whichever section precedes it in page order.
    order = sorted(
        range(len(sections)),
        key=lambda i: (sections[i].page_start, sections[i].page_end),
    )
    for a, b in zip(order, order[1:]):
        if b not in grounded:
            continue
        cur, nxt = sections[a], sections[b]
        _page, off = grounded[b]
        boundary = (
            nxt.page_start - 1 if off <= _TOP_OF_PAGE_CHARS else nxt.page_start
        )
        boundary = max(boundary, cur.page_start)
        # A grounded section's end is fully derived; an ungrounded one is only
        # pulled back off the neighbor's pages (its own claim may be printed
        # numbering, but extending it would be a guess).
        new_end = boundary if a in grounded else min(cur.page_end, boundary)
        if cur.page_end != new_end:
            logger.info(
                "Grounding: section %r page_end %d -> %d (next section %r "
                "starts on physical page %d)",
                cur.title,
                cur.page_end,
                new_end,
                nxt.title,
                nxt.page_start,
            )
            cur.page_end = new_end
    return [sections[i] for i in order]


def _fill_coverage_gaps(sections: List[Section], page_count: int) -> List[Section]:
    """Guarantee every physical page belongs to at least one section.

    Pass 2 only reads pages the outline names, so a page the model failed to
    assign to any section would otherwise be silently skipped. Section ranges
    are clamped into [1, page_count]; every remaining uncovered run of pages
    becomes a synthetic "Unmapped pages" section (and a warning), so its
    content still gets chunked.
    """
    if page_count <= 0:
        return sections

    kept: List[Section] = []
    for sec in sections:
        sec.page_start = max(1, min(sec.page_start, page_count))
        sec.page_end = max(sec.page_start, min(sec.page_end, page_count))
        kept.append(sec)

    covered = [False] * (page_count + 1)  # 1-indexed
    for sec in kept:
        for page in range(sec.page_start, sec.page_end + 1):
            covered[page] = True

    page = 1
    while page <= page_count:
        if covered[page]:
            page += 1
            continue
        gap_start = page
        while page <= page_count and not covered[page]:
            page += 1
        gap_end = page - 1
        logger.warning(
            "Pass 1 assigned no section to pages %d-%d; adding a synthetic "
            "'Unmapped pages' section so they still get chunked",
            gap_start,
            gap_end,
        )
        kept.append(
            Section(
                title=f"Unmapped pages {gap_start}-{gap_end}",
                section_type="unmapped",
                summary="Pages not assigned to any section by Pass 1.",
                page_start=gap_start,
                page_end=gap_end,
            )
        )
    kept.sort(key=lambda s: (s.page_start, s.page_end))
    return kept


def reconcile(
    partials: List[Dict[str, Any]],
    source_file: str,
    page_count: int,
    page_texts: Optional[List[str]] = None,
) -> DocumentProfile:
    """Merge per-batch partial profiles into one global ``DocumentProfile``.

    With ``page_texts`` (the per-page text layer), the merged outline is
    grounded against where section headings physically appear before gaps are
    filled, correcting printed-vs-physical page numbering from the model.
    """
    all_sections: List[Dict[str, Any]] = []
    for p in partials:
        all_sections.extend(p.get("sections", []))

    sections = _merge_sections(all_sections)
    if page_texts:
        sections = ground_sections(sections, page_texts)

    return DocumentProfile(
        source_file=source_file,
        page_count=page_count,
        doc_type=_first_nonempty([p.get("doc_type", "") for p in partials])
        or "unknown",
        title=_first_nonempty([p.get("title", "") for p in partials]),
        plan_name=_first_nonempty([p.get("plan_name", "") for p in partials]),
        sponsor=_first_nonempty([p.get("sponsor", "") for p in partials]),
        effective_dates=_dedupe_strings(
            [d for p in partials for d in p.get("effective_dates", [])]
        ),
        sections=_fill_coverage_gaps(sections, page_count),
        glossary=_merge_glossary([p.get("glossary", []) for p in partials]),
        cross_references=_sanitize_references(
            [c for p in partials for c in p.get("cross_references", [])]
        ),
        notes=_first_nonempty([p.get("notes", "") for p in partials]),
    )
