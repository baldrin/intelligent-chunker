import json

from conftest import FakeClient, make_pdf

from intelligent_chunker import pipeline
from intelligent_chunker.config import ChunkerConfig
from intelligent_chunker.models import Chunk, DocumentProfile, Section


def _chunk(section_title, index=0, text="t"):
    return Chunk(
        text=text,
        source_file="x.pdf",
        chunk_index=index,
        section_title=section_title,
        section_type="general",
        section_summary="",
        page_start=1,
        page_end=1,
    )


SECTIONS = [Section(f"S{i}", "general", "", i, i) for i in range(1, 4)]


def test_prefix_all_sections_done():
    chunks = [_chunk("S1"), _chunk("S2"), _chunk("S3")]
    assert pipeline.completed_section_prefix(SECTIONS, chunks) == 3


def test_prefix_partial():
    chunks = [_chunk("S1"), _chunk("S2")]
    assert pipeline.completed_section_prefix(SECTIONS, chunks) == 2


def test_prefix_empty_file():
    assert pipeline.completed_section_prefix(SECTIONS, []) == 0


def test_prefix_stops_at_first_gap_even_if_later_titles_present():
    # S2 missing: the run died mid-S2, so S3's chunks (impossible in a real
    # file, but be safe) must not extend the prefix.
    chunks = [_chunk("S1"), _chunk("S3")]
    assert pipeline.completed_section_prefix(SECTIONS, chunks) == 1


def _config():
    # Bogus tokenizer id -> heuristic counter, no network dependency.
    return ChunkerConfig(
        max_tokens=1000, target_tokens=500, tokenizer_id="invalid/none"
    )


def test_run_resume_skips_completed_sections(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(make_pdf(3))
    out = tmp_path / "chunks.jsonl"
    profile_path = tmp_path / "profile.json"

    profile = DocumentProfile(
        source_file="doc.pdf", page_count=3, sections=list(SECTIONS)
    )
    pipeline.write_profile(profile, str(profile_path))

    # Simulate an interrupted run: S1 finished, S2/S3 did not.
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(_chunk("S1", 0, "kept text").to_dict()) + "\n")

    payload = {"chunks": [{"text": "new text", "keywords": [], "cross_references": []}]}
    client = FakeClient([payload])
    result = pipeline.run(
        str(pdf),
        config=_config(),
        out_path=str(out),
        profile_path=str(profile_path),
        client=client,
        resume=True,
    )

    # Pass 1 skipped, only S2 and S3 called.
    assert len(client.messages.calls) == 2
    assert [c.section_title for c in result.chunks] == ["S1", "S2", "S3"]
    assert [c.chunk_index for c in result.chunks] == [0, 1, 2]
    assert result.chunks[0].text == "kept text"

    # The file on disk matches the returned chunks.
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [c["section_title"] for c in lines] == ["S1", "S2", "S3"]


def test_run_resume_without_existing_files_runs_fresh(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(make_pdf(1))
    pass1_payload = {
        "doc_type": "SPD",
        "title": "T",
        "plan_name": "",
        "sponsor": "",
        "effective_dates": [],
        "sections": [
            {"title": "Only", "section_type": "general", "summary": "",
             "page_start": 1, "page_end": 1}
        ],
        "glossary": [],
        "cross_references": [],
        "notes": "",
    }
    pass2_payload = {
        "chunks": [{"text": "a", "keywords": [], "cross_references": []}]
    }
    client = FakeClient([pass1_payload, pass2_payload])
    result = pipeline.run(
        str(pdf),
        config=_config(),
        out_path=str(tmp_path / "chunks.jsonl"),
        profile_path=str(tmp_path / "profile.json"),
        client=client,
        resume=True,
    )
    assert len(client.messages.calls) == 2  # Pass 1 + one section
    assert len(result.chunks) == 1
