import json

import pytest

from intelligent_chunker import export_databricks as ex


PROFILE = {
    "source_file": "spd.pdf",
    "page_count": 20,
    "doc_type": "SPD",
    "title": "Acme 401(k) SPD",
    "plan_name": "Acme 401(k) Plan",
    "sponsor": "Acme",
    "effective_dates": ["2024-01-01"],
    "sections": [{"title": "Vesting", "section_type": "vesting", "summary": "your rights", "page_start": 5, "page_end": 6}],
    "glossary": [
        {"term": "Vesting", "definition": "your nonforfeitable right"},
        {"term": "vesting", "definition": "dup casing"},
    ],
    "cross_references": ["see Distributions"],
    "notes": "n",
}

CHUNKS = [
    {"text": "You are 100% vested after 3 years.", "section_title": "Vesting",
     "section_type": "vesting", "section_summary": "your rights",
     "page_start": 5, "page_end": 6, "keywords": ["vesting"],
     "cross_references": ["Distributions"], "token_count": 12, "chunk_index": 0},
    # exact duplicate text -> should collapse to one row
    {"text": "You are 100% vested after 3 years.", "section_title": "Distributions",
     "section_type": "distributions", "section_summary": "",
     "page_start": 6, "page_end": 6, "keywords": [], "cross_references": [],
     "token_count": 12, "chunk_index": 1},
    {"text": "Forfeitures are reallocated annually.", "section_title": "Vesting",
     "section_type": "vesting", "section_summary": "your rights",
     "page_start": 6, "page_end": 6, "keywords": ["forfeiture"],
     "cross_references": [], "token_count": 8, "chunk_index": 2},
]


def test_embedding_text_folds_in_context():
    et = ex.build_embedding_text(PROFILE, CHUNKS[0])
    assert "Acme 401(k) SPD" in et          # title
    assert "Acme 401(k) Plan" in et          # plan appended (not in title)
    assert "Section: Vesting — your rights" in et
    assert "You are 100% vested" in et
    # context comes before the chunk body
    assert et.index("Document:") < et.index("Section:") < et.index("You are 100%")


def test_embedding_text_dedupes_redundant_context():
    # Title already contains the plan name and doc type -> don't restate them.
    profile = {
        "title": "Summary Plan Description - Acme 401(k) Plan",
        "plan_name": "Acme 401(k) Plan",
        "doc_type": "Summary Plan Description (SPD)",
    }
    et = ex.build_embedding_text(profile, CHUNKS[0])
    doc_line = et.splitlines()[0]
    assert doc_line == "Document: Summary Plan Description - Acme 401(k) Plan"


def test_chunk_rows_fields_and_dedup():
    rows, dups = ex.build_chunk_rows(PROFILE, CHUNKS)
    assert dups == 1            # the exact-duplicate text collapsed
    assert len(rows) == 2
    r = rows[0]
    # required identity + provenance
    assert r["id"].startswith("chk_")
    assert r["doc_id"].startswith("doc_")
    assert r["source_file"] == "spd.pdf"
    # document metadata denormalized onto the chunk for filtering
    assert r["plan_name"] == "Acme 401(k) Plan"
    assert r["effective_dates"] == ["2024-01-01"]
    # section + retrieval signals
    assert r["section_title"] == "Vesting"
    assert r["keywords"] == ["vesting"]
    assert r["embedding_text"].startswith("Document:")
    assert r["token_count"] == 12


def test_chunk_ids_are_stable():
    rows1, _ = ex.build_chunk_rows(PROFILE, CHUNKS)
    rows2, _ = ex.build_chunk_rows(PROFILE, CHUNKS)
    assert [r["id"] for r in rows1] == [r["id"] for r in rows2]


def test_document_row():
    rows, _ = ex.build_chunk_rows(PROFILE, CHUNKS)
    doc = ex.build_document_row(PROFILE, chunk_count=len(rows))
    assert doc["doc_id"].startswith("doc_")
    assert doc["chunk_count"] == 2
    assert doc["section_count"] == 1
    assert json.loads(doc["sections_json"])[0]["title"] == "Vesting"
    assert json.loads(doc["glossary_json"])  # valid JSON array
    assert doc["cross_references"] == ["see Distributions"]


def test_glossary_rows_dedup_case_insensitive():
    rows = ex.build_glossary_rows(PROFILE)
    assert len(rows) == 1  # "Vesting" / "vesting" collapse
    assert rows[0]["id"].startswith("gls_")
    assert rows[0]["embedding_text"].startswith("Vesting:")


def test_parquet_roundtrip(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    result = ex.export(PROFILE, CHUNKS, out_dir=str(tmp_path))
    assert result.chunk_rows == 2
    assert result.duplicates_dropped == 1
    tbl = pq.read_table(result.chunks_path)
    assert tbl.num_rows == 2
    import pyarrow as pa

    names = set(tbl.schema.names)
    assert {"id", "text", "embedding_text", "keywords", "effective_dates"} <= names
    # array columns typed as a list (DVS array<string>)
    assert pa.types.is_list(tbl.schema.field("keywords").type)
    assert pa.types.is_list(tbl.schema.field("effective_dates").type)
