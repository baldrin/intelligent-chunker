"""Pass 1 -- global document analysis (the "map").

Read the whole document (batched if long) and build a ``DocumentProfile``: the
section outline with page ranges, document-wide metadata, a glossary of defined
terms, and cross-references. Pass 2 consumes this so every chunk is produced
with full-document context instead of a blind linear read.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from .config import ChunkerConfig
from .llm import structured_call
from .models import DocumentProfile, GlossaryTerm, Section
from .pdf_io import PageBatch, document_block, iter_batches

PASS1_SYSTEM = (
    "You are an expert at analyzing benefits and insurance documents, "
    "especially Summary Plan Description (SPD) documents. You will be given a "
    "PDF (it may be digital text or scanned images). Read the entire document "
    "and produce a structured global map of it: document-wide metadata, the "
    "full ordered outline of sections with their page ranges, a glossary of "
    "terms the document explicitly defines, and notable cross-references "
    "between sections. Be thorough and faithful to the document -- do not "
    "invent sections or definitions.\n\n"
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
    client: Any, config: ChunkerConfig, batch: PageBatch
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
    )
    # Shift batch-relative page numbers to absolute document pages.
    for sec in raw.get("sections", []):
        sec["page_start"] = int(sec.get("page_start", 1)) + batch.page_offset
        sec["page_end"] = int(sec.get("page_end", 1)) + batch.page_offset
    return raw


def analyze_document(
    client: Any, config: ChunkerConfig, pdf_bytes: bytes, source_file: str
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
        partials = [analyze_batch(client, config, b) for b in batches]
    else:
        # Batches are independent; fan out. pool.map preserves batch order,
        # which reconcile relies on (first non-empty metadata wins, and batch
        # 1 holds the title page). The Anthropic client is thread-safe.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            partials = list(
                pool.map(lambda b: analyze_batch(client, config, b), batches)
            )
    page_count = batches[-1].page_end if batches else 0
    return reconcile(partials, source_file=source_file, page_count=page_count)


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


def reconcile(
    partials: List[Dict[str, Any]], source_file: str, page_count: int
) -> DocumentProfile:
    """Merge per-batch partial profiles into one global ``DocumentProfile``."""
    all_sections: List[Dict[str, Any]] = []
    for p in partials:
        all_sections.extend(p.get("sections", []))

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
        sections=_merge_sections(all_sections),
        glossary=_merge_glossary([p.get("glossary", []) for p in partials]),
        cross_references=_dedupe_strings(
            [c for p in partials for c in p.get("cross_references", [])]
        ),
        notes=_first_nonempty([p.get("notes", "") for p in partials]),
    )
