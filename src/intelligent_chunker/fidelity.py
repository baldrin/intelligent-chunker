"""Report-only fidelity check: chunk text vs the PDF's embedded text layer.

The model transcribes the document; nothing else in the pipeline verifies it
didn't drop or invent text. For digital PDFs this module compares, per
section, the word multiset of the pypdf-extracted text layer against the word
multiset of the produced chunks:

* ``coverage`` -- fraction of text-layer words present in the chunks
  (low coverage = content may have been missed);
* ``novelty``  -- fraction of chunk words absent from the text layer
  (high novelty = content may have been invented).

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

    sections = []
    ref_weighted_cov = 0.0
    cand_weighted_nov = 0.0
    ref_grand = 0
    cand_grand = 0
    for sec in profile.sections:
        start = max(1, min(sec.page_start, len(page_texts)))
        end = max(start, min(sec.page_end, len(page_texts)))
        reference = "\n".join(page_texts[start - 1 : end])
        candidate = "\n".join(by_section.get(sec.title, []))
        score = score_texts(reference, candidate)
        sections.append({"title": sec.title, **score})
        ref_weighted_cov += score["coverage"] * score["reference_words"]
        cand_weighted_nov += score["novelty"] * score["chunk_words"]
        ref_grand += score["reference_words"]
        cand_grand += score["chunk_words"]

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

    return {
        "status": "ok",
        "document": {
            "coverage": round(ref_weighted_cov / ref_grand, 4) if ref_grand else 1.0,
            "novelty": round(cand_weighted_nov / cand_grand, 4) if cand_grand else 0.0,
        },
        "sections": sections,
    }
