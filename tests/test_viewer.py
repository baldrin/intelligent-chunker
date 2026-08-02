"""Viewer HTML generation, curation affordances, and embedding safety."""

import json

from intelligent_chunker import viewer


def _sample():
    profile = {
        "source_file": "spd.pdf",
        "title": "Sample SPD",
        "page_count": 2,
        "sections": [
            {
                "title": "Eligibility",
                "section_type": "eligibility",
                "summary": "",
                "page_start": 1,
                "page_end": 2,
            }
        ],
    }
    chunks = [
        {
            "chunk_index": 0,
            "section_title": "Eligibility",
            "section_type": "eligibility",
            "text": "Employees are eligible </script><!-- sneaky",
            "token_count": 10,
            "page_start": 1,
            "page_end": 1,
            "keywords": ["k"],
            "cross_references": [],
        }
    ]
    return profile, chunks


def test_embed_escapes_script_breakers():
    out = viewer._embed({"t": "</script> <!-- x"})
    # Once the escaped forms are removed, no raw breaker sequences remain.
    assert "</" not in out.replace("<\\/", "")
    assert "<!--" not in out.replace("<\\!--", "")


def test_build_html_embeds_data_and_curation_ui():
    profile, chunks = _sample()
    html = viewer.build_html(profile, chunks)
    assert "const DATA = " in html
    # Document text containing script-breaking sequences never survives raw.
    assert "</script><!--" not in html
    for marker in (
        "dlBtn",
        "resetBtn",
        "seccb",
        "Download curated JSONL",
        "localStorage",
        "curated.jsonl",
    ):
        assert marker in html


def test_load_chunks_roundtrip(tmp_path):
    p = tmp_path / "c.jsonl"
    rows = [{"chunk_index": i, "text": f"t{i}"} for i in range(3)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert viewer.load_chunks(str(p)) == rows
