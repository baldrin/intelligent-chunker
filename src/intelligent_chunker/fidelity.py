"""Report-only fidelity check: chunk text vs the PDF's embedded text layer.

The model transcribes the document; nothing else in the pipeline verifies it
didn't drop or invent text. For digital PDFs this module compares, per
section, the word multiset of the pypdf-extracted text layer against the word
multiset of the produced chunks:

* ``coverage`` -- fraction of text-layer words present in the chunks
  (low coverage = content may have been missed);
* ``novelty``  -- fraction of chunk words absent from the text layer
  (high novelty = content may have been invented).

The document-level score compares ALL pages against ALL chunks, which is the
trustworthy number. Per-section scores carry a known bias: sections often
share a page, so a section's reference text includes its neighbors' words and
coverage reads low even for a perfect transcription. Sections with shared
pages are marked ``shared_pages: true`` and never warned about.

A per-chunk pass additionally scores every chunk against its own page range
and reports the offenders (``chunks`` in the report): high novelty there
localizes invented lines or text attributed to the wrong pages down to a
single chunk, which section-level scores can't do.

The comparison is deliberately rough: headers/footers repeat per page,
hyphenation and ligatures differ between extractors, and scanned PDFs have no
text layer at all (the report is then skipped). Scores are advisory signals,
never hard failures.
"""

from __future__ import annotations

import io
import logging
import re
from collections import Counter
from typing import Any, Dict, List

from pypdf import PdfReader

from .models import Chunk, DocumentProfile

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")

# Below this many characters across the whole text layer, treat the PDF as
# scanned (vision-only) and skip rather than reporting meaningless zeros.
_MIN_TEXT_LAYER_CHARS = 200

COVERAGE_WARN_BELOW = 0.85
NOVELTY_WARN_ABOVE = 0.15


def _words(text: str) -> Counter:
    return Counter(_WORD_RE.findall(text.lower()))


def match_key(text: str) -> str:
    """Normalize for text-layer matching: case, whitespace and punctuation are
    all extraction artifacts (the layer contains e.g. ``II.PARTICIPATION`` and
    mid-word splits like ``Defer ral``), so keep only [a-z0-9]. Used to locate
    section headings and chunk text on physical pages."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def score_texts(reference: str, candidate: str) -> Dict[str, Any]:
    """Multiset word overlap between a reference text and a candidate."""
    ref = _words(reference)
    cand = _words(candidate)
    ref_total = sum(ref.values())
    cand_total = sum(cand.values())
    matched = sum((ref & cand).values())
    return {
        "coverage": round(matched / ref_total, 4) if ref_total else 1.0,
        "novelty": round(1 - matched / cand_total, 4) if cand_total else 0.0,
        "reference_words": ref_total,
        "chunk_words": cand_total,
    }


def extract_page_texts(pdf_bytes: bytes) -> List[str]:
    """Per-page text layer via pypdf (empty strings for image-only pages)."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts: List[str] = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:  # malformed page content stream
            texts.append("")
    return texts


def sections_with_shared_pages(sections: List[Any]) -> List[bool]:
    """For each section, whether any of its pages is covered by another.

    Shared pages make that section's per-section scores unreliable (the
    reference includes neighbors' text), so warnings are suppressed there.
    """
    page_owners: Dict[int, int] = {}
    for sec in sections:
        for page in range(sec.page_start, sec.page_end + 1):
            page_owners[page] = page_owners.get(page, 0) + 1
    return [
        any(
            page_owners.get(page, 0) > 1
            for page in range(sec.page_start, sec.page_end + 1)
        )
        for sec in sections
    ]


# Chunks shorter than this many words give meaninglessly noisy per-chunk
# novelty ratios (a heading fragment can be 100% "novel" by accident).
CHUNK_NOVELTY_MIN_WORDS = 20

# For flagged chunks, report the specific lines that are mostly absent from
# the chunk's pages -- the actionable detail for a human reviewer.
_NOVEL_LINE_MIN_WORDS = 4
_NOVEL_LINE_ABSENT_FRACTION = 0.5
_NOVEL_LINES_MAX = 3


def chunk_novelty_flags(
    page_texts: List[str], profile: DocumentProfile, chunks: List[Chunk]
) -> List[Dict[str, Any]]:
    """Score each chunk against its own page range; return the offenders.

    A chunk is flagged when its novelty against its claimed pages exceeds
    ``NOVELTY_WARN_ABOVE`` (text broadly misattributed or misplaced) OR when
    any single line is mostly made of words absent from those pages (a
    localized invented line -- too small to move the whole-chunk ratio, which
    is exactly how a fabricated sentence hides). Chunks in synthetic
    "unmapped" sections are skipped: title/TOC pages produce junk ratios and
    their content is already known noise.
    """
    unmapped = {
        s.title for s in profile.sections if s.section_type == "unmapped"
    }
    flags: List[Dict[str, Any]] = []
    for chunk in chunks:
        if chunk.section_title in unmapped:
            continue
        if len(chunk.text.split()) < CHUNK_NOVELTY_MIN_WORDS:
            continue
        start = max(1, min(chunk.page_start, len(page_texts)))
        end = max(start, min(chunk.page_end, len(page_texts)))
        reference = "\n".join(page_texts[start - 1 : end])
        score = score_texts(reference, chunk.text)
        ref_words = set(_WORD_RE.findall(reference.lower()))
        novel_lines: List[str] = []
        for line in chunk.text.splitlines():
            words = _WORD_RE.findall(line.lower())
            if len(words) < _NOVEL_LINE_MIN_WORDS:
                continue
            absent = sum(1 for w in words if w not in ref_words)
            if absent / len(words) >= _NOVEL_LINE_ABSENT_FRACTION:
                novel_lines.append(line.strip())
        if score["novelty"] <= NOVELTY_WARN_ABOVE and not novel_lines:
            continue
        flags.append(
            {
                "chunk_index": chunk.chunk_index,
                "section": chunk.section_title,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "novelty": score["novelty"],
                "novel_lines": novel_lines[:_NOVEL_LINES_MAX],
            }
        )
        logger.warning(
            "Fidelity: chunk %d (pages %d-%d, section %r) novelty %.2f -- "
            "contains text not found on its pages%s",
            chunk.chunk_index,
            chunk.page_start,
            chunk.page_end,
            chunk.section_title,
            score["novelty"],
            "; e.g. %r" % novel_lines[0] if novel_lines else "",
        )
    return flags


def fidelity_report(
    pdf_bytes: bytes, profile: DocumentProfile, chunks: List[Chunk]
) -> Dict[str, Any]:
    """Build the per-section + document fidelity report (see module doc)."""
    page_texts = extract_page_texts(pdf_bytes)
    if sum(len(t.strip()) for t in page_texts) < _MIN_TEXT_LAYER_CHARS:
        return {
            "status": "skipped",
            "reason": "no text layer (scanned PDF?)",
        }

    by_section: Dict[str, List[str]] = {}
    for chunk in chunks:
        by_section.setdefault(chunk.section_title, []).append(chunk.text)

    shared = sections_with_shared_pages(profile.sections)
    sections = []
    for sec, has_shared in zip(profile.sections, shared):
        start = max(1, min(sec.page_start, len(page_texts)))
        end = max(start, min(sec.page_end, len(page_texts)))
        reference = "\n".join(page_texts[start - 1 : end])
        candidate = "\n".join(by_section.get(sec.title, []))
        score = score_texts(reference, candidate)
        sections.append({"title": sec.title, "shared_pages": has_shared, **score})

        if has_shared:
            continue  # scores biased by neighbors' text; report but don't warn
        if score["coverage"] < COVERAGE_WARN_BELOW:
            logger.warning(
                "Fidelity: section %r coverage %.2f -- text-layer content "
                "may be missing from its chunks",
                sec.title,
                score["coverage"],
            )
        if score["novelty"] > NOVELTY_WARN_ABOVE:
            logger.warning(
                "Fidelity: section %r novelty %.2f -- chunks contain text "
                "not found in the PDF text layer",
                sec.title,
                score["novelty"],
            )

    # The document score is a single global comparison (all pages vs all
    # chunks) -- immune to the shared-page bias, hence the number to trust.
    document = score_texts(
        "\n".join(page_texts), "\n".join(c.text for c in chunks)
    )
    if document["coverage"] < COVERAGE_WARN_BELOW:
        logger.warning(
            "Fidelity: document coverage %.2f -- text-layer content may be "
            "missing from the chunks",
            document["coverage"],
        )
    if document["novelty"] > NOVELTY_WARN_ABOVE:
        logger.warning(
            "Fidelity: document novelty %.2f -- chunks contain text not "
            "found in the PDF text layer",
            document["novelty"],
        )

    return {
        "status": "ok",
        "document": document,
        "sections": sections,
        "chunks": chunk_novelty_flags(page_texts, profile, chunks),
    }
