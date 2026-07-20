"""Export chunker output as Databricks-ready Parquet tables.

Produces three typed, Delta-friendly tables from a ``chunks.jsonl`` /
``profile.json`` pair so they drop straight into Databricks AI Search
(formerly Vector Search) Delta Sync indexes:

* ``chunks``     -- one row per chunk; the retrieval unit. Carries a
                    contextual ``embedding_text`` (document + section context
                    folded in) plus the raw ``text`` and all filter metadata.
* ``documents``  -- one row per document; the global map (metadata, section
                    outline, glossary) for routing, filtering, and enrichment.
* ``glossary``   -- one row per defined term; a definitional index so "what
                    does X mean" queries hit definitions directly.

Embeddings are intentionally NOT included -- that is the later embedding phase.
The ``embedding_text`` column is what should be fed to GTE-large v1.5 (or a
Databricks-managed embedding model) to fill an ``embedding`` column.

The row-building functions are pure (no pyarrow) so they're easy to test; only
``write_parquet`` needs the ``databricks`` extra.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


def _sha1_16(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _doc_id(profile: Dict[str, Any]) -> str:
    # Stable per source document, so re-runs update rather than duplicate.
    key = profile.get("source_file", "") or profile.get("title", "")
    if not key:
        # Last resort: hash the whole profile, so two documents that both
        # lack a source_file/title can't silently collide on one doc_id.
        key = json.dumps(profile, sort_keys=True, ensure_ascii=False)
    return "doc_" + _sha1_16(key)


def _clean(s: Any) -> str:
    return (s or "").strip() if isinstance(s, str) else ("" if s is None else str(s))


def build_embedding_text(profile: Dict[str, Any], chunk: Dict[str, Any]) -> str:
    """Compose the contextual string to embed (Anthropic contextual-retrieval).

    Folds document and section context into the chunk so the vector encodes
    *where* the content lives, not just the raw passage.
    """
    title = _clean(profile.get("title")) or _clean(profile.get("plan_name"))
    plan = _clean(profile.get("plan_name"))

    doc_line = "Document: " + (title or _clean(profile.get("source_file")))
    # Append the plan only when the title doesn't already contain it. doc_type
    # is intentionally left out of the embedded text: it's redundant with the
    # title for these documents, low-signal for the vector (every chunk shares
    # it), and already available as a filter column.
    if plan and plan.lower() not in title.lower():
        doc_line += " — " + plan

    sec_line = "Section: " + _clean(chunk.get("section_title"))
    summary = _clean(chunk.get("section_summary"))
    if summary:
        sec_line += " — " + summary

    return "\n".join([doc_line, sec_line, _clean(chunk.get("text"))]).strip()


def build_chunk_rows(
    profile: Dict[str, Any], chunks: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int]:
    """Build chunk-table rows. Returns (rows, duplicates_dropped).

    IDs are a content hash of (doc_id + raw text), so exact-duplicate chunks
    (e.g. boundary bleed across two sections) collapse to one row. Order is
    preserved; the first occurrence wins.
    """
    doc_id = _doc_id(profile)
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    duplicates = 0

    for c in chunks:
        text = _clean(c.get("text"))
        if not text:
            continue
        cid = "chk_" + _sha1_16(doc_id + "::" + text)
        if cid in seen:
            duplicates += 1
            continue
        seen.add(cid)
        rows.append(
            {
                "id": cid,
                "doc_id": doc_id,
                "source_file": _clean(profile.get("source_file")),
                "text": text,
                "embedding_text": build_embedding_text(profile, c),
                "doc_type": _clean(profile.get("doc_type")),
                "title": _clean(profile.get("title")),
                "plan_name": _clean(profile.get("plan_name")),
                "sponsor": _clean(profile.get("sponsor")),
                "effective_dates": list(profile.get("effective_dates", []) or []),
                "section_title": _clean(c.get("section_title")),
                "section_type": _clean(c.get("section_type")),
                "section_summary": _clean(c.get("section_summary")),
                "page_start": int(c.get("page_start", 0) or 0),
                "page_end": int(c.get("page_end", 0) or 0),
                "keywords": list(c.get("keywords", []) or []),
                "cross_references": list(c.get("cross_references", []) or []),
                "token_count": (
                    int(c["token_count"]) if c.get("token_count") is not None else None
                ),
            }
        )
    return rows, duplicates


def build_document_row(
    profile: Dict[str, Any], chunk_count: int
) -> Dict[str, Any]:
    """Build the single document-table row (the global map)."""
    sections = profile.get("sections", []) or []
    glossary = profile.get("glossary", []) or []
    return {
        "doc_id": _doc_id(profile),
        "source_file": _clean(profile.get("source_file")),
        "title": _clean(profile.get("title")),
        "doc_type": _clean(profile.get("doc_type")),
        "plan_name": _clean(profile.get("plan_name")),
        "sponsor": _clean(profile.get("sponsor")),
        "effective_dates": list(profile.get("effective_dates", []) or []),
        "page_count": int(profile.get("page_count", 0) or 0),
        "section_count": len(sections),
        "chunk_count": int(chunk_count),
        # Nested structures kept as JSON strings for a flat, portable schema
        # (parse in Databricks with from_json when needed).
        "sections_json": json.dumps(sections, ensure_ascii=False),
        "glossary_json": json.dumps(glossary, ensure_ascii=False),
        "cross_references": list(profile.get("cross_references", []) or []),
        "notes": _clean(profile.get("notes")),
    }


def build_glossary_rows(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build glossary-table rows (one per defined term)."""
    doc_id = _doc_id(profile)
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for g in profile.get("glossary", []) or []:
        term = _clean(g.get("term"))
        if not term:
            continue
        gid = "gls_" + _sha1_16(doc_id + "::" + term.lower())
        if gid in seen:
            continue
        seen.add(gid)
        definition = _clean(g.get("definition"))
        rows.append(
            {
                "id": gid,
                "doc_id": doc_id,
                "term": term,
                "definition": definition,
                "embedding_text": f"{term}: {definition}" if definition else term,
            }
        )
    return rows


# --- Parquet writing (needs the `databricks` extra: pyarrow) ----------------


def _schemas():
    import pyarrow as pa

    string = pa.string()
    chunks = pa.schema(
        [
            ("id", string),
            ("doc_id", string),
            ("source_file", string),
            ("text", string),
            ("embedding_text", string),
            ("doc_type", string),
            ("title", string),
            ("plan_name", string),
            ("sponsor", string),
            ("effective_dates", pa.list_(string)),
            ("section_title", string),
            ("section_type", string),
            ("section_summary", string),
            ("page_start", pa.int32()),
            ("page_end", pa.int32()),
            ("keywords", pa.list_(string)),
            ("cross_references", pa.list_(string)),
            ("token_count", pa.int32()),
        ]
    )
    documents = pa.schema(
        [
            ("doc_id", string),
            ("source_file", string),
            ("title", string),
            ("doc_type", string),
            ("plan_name", string),
            ("sponsor", string),
            ("effective_dates", pa.list_(string)),
            ("page_count", pa.int32()),
            ("section_count", pa.int32()),
            ("chunk_count", pa.int32()),
            ("sections_json", string),
            ("glossary_json", string),
            ("cross_references", pa.list_(string)),
            ("notes", string),
        ]
    )
    glossary = pa.schema(
        [
            ("id", string),
            ("doc_id", string),
            ("term", string),
            ("definition", string),
            ("embedding_text", string),
        ]
    )
    return chunks, documents, glossary


def _write_table(rows: List[Dict[str, Any]], schema: Any, path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)


@dataclass
class ExportResult:
    chunks_path: str
    documents_path: str
    glossary_path: str
    chunk_rows: int
    glossary_rows: int
    duplicates_dropped: int


def export(
    profile: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    out_dir: str,
    include_glossary: bool = True,
) -> ExportResult:
    """Write chunks/documents/glossary Parquet tables to ``out_dir``."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    chunk_schema, doc_schema, gloss_schema = _schemas()

    chunk_rows, dups = build_chunk_rows(profile, chunks)
    doc_row = build_document_row(profile, chunk_count=len(chunk_rows))
    gloss_rows = build_glossary_rows(profile) if include_glossary else []

    chunks_path = os.path.join(out_dir, "chunks.parquet")
    documents_path = os.path.join(out_dir, "documents.parquet")
    glossary_path = (
        os.path.join(out_dir, "glossary.parquet") if include_glossary else ""
    )

    _write_table(chunk_rows, chunk_schema, chunks_path)
    _write_table([doc_row], doc_schema, documents_path)
    if include_glossary:
        _write_table(gloss_rows, gloss_schema, glossary_path)

    return ExportResult(
        chunks_path=chunks_path,
        documents_path=documents_path,
        glossary_path=glossary_path,
        chunk_rows=len(chunk_rows),
        glossary_rows=len(gloss_rows),
        duplicates_dropped=dups,
    )
