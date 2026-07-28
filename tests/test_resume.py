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


# --- truncated chunks file (interrupted mid-write) ---------------------------


def test_read_chunks_truncated_line_discards_partial_section(tmp_path):
    p = tmp_path / "chunks.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps(_chunk("S1", 0).to_dict()) + "\n")
        f.write(json.dumps(_chunk("S2", 1).to_dict()) + "\n")
        f.write(json.dumps(_chunk("S2", 2).to_dict()) + "\n")
        f.write('{"text": "cut off mid-wri')  # torn final line
    chunks = pipeline._read_chunks(str(p))
    # S2 cannot be proven complete (the torn line may be its tail), so all
    # of S2 is dropped and only S1 survives to be counted as done.
    assert [c.section_title for c in chunks] == ["S1"]


def test_read_chunks_truncated_first_line_reruns_everything(tmp_path):
    p = tmp_path / "chunks.jsonl"
    p.write_text('{"text": "cut', encoding="utf-8")
    assert pipeline._read_chunks(str(p)) == []


def test_run_resume_recovers_from_truncated_chunks_file(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(make_pdf(3))
    out = tmp_path / "chunks.jsonl"
    profile_path = tmp_path / "profile.json"
    profile = DocumentProfile(
        source_file="doc.pdf", page_count=3, sections=list(SECTIONS)
    )
    pipeline.write_profile(profile, str(profile_path))
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(_chunk("S1", 0, "kept").to_dict()) + "\n")
        f.write(json.dumps(_chunk("S2", 1, "suspect").to_dict()) + "\n")
        f.write('{"text": "cut off mid-wri')
    payload = {
        "chunks": [{"text": "new text", "keywords": [], "cross_references": []}]
    }
    client = FakeClient([payload])
    result = pipeline.run(
        str(pdf),
        config=_config(),
        out_path=str(out),
        profile_path=str(profile_path),
        client=client,
        resume=True,
        fidelity=False,
    )
    # S2 (possibly incomplete) and S3 re-chunk; S1 is kept as-is.
    assert len(client.messages.calls) == 2
    assert [c.section_title for c in result.chunks] == ["S1", "S2", "S3"]
    assert result.chunks[0].text == "kept"
    assert result.chunks[1].text == "new text"


# --- resumed profile sanity checks -------------------------------------------


def test_run_resume_rejects_page_count_mismatch(tmp_path):
    import pytest

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(make_pdf(2))
    profile_path = tmp_path / "profile.json"
    profile = DocumentProfile(
        source_file="doc.pdf", page_count=5, sections=list(SECTIONS)
    )
    pipeline.write_profile(profile, str(profile_path))
    client = FakeClient([{"chunks": []}])
    with pytest.raises(ValueError, match="does not match this PDF"):
        pipeline.run(
            str(pdf),
            config=_config(),
            profile_path=str(profile_path),
            client=client,
            resume=True,
        )
    assert len(client.messages.calls) == 0  # failed before any API spend


def test_run_resume_warns_on_source_file_mismatch(tmp_path, caplog):
    import logging

    pdf = tmp_path / "renamed.pdf"
    pdf.write_bytes(make_pdf(3))
    profile_path = tmp_path / "profile.json"
    profile = DocumentProfile(
        source_file="original.pdf", page_count=3, sections=list(SECTIONS)
    )
    pipeline.write_profile(profile, str(profile_path))
    payload = {"chunks": [{"text": "t", "keywords": [], "cross_references": []}]}
    client = FakeClient([payload])
    with caplog.at_level(logging.WARNING, logger="intelligent_chunker.pipeline"):
        result = pipeline.run(
            str(pdf),
            config=_config(),
            profile_path=str(profile_path),
            client=client,
            resume=True,
            fidelity=False,
        )
    assert any("original.pdf" in r.getMessage() for r in caplog.records)
    assert len(result.chunks) == 3  # page counts match, so the run proceeds


def test_run_resume_uniquifies_old_duplicate_title_profile(tmp_path):
    # A profile written before titles were uniquified: two sections named
    # "Definitions". The old bug counted the second one as already chunked
    # (same title as the finished first one) and silently skipped its pages.
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(make_pdf(4))
    out = tmp_path / "chunks.jsonl"
    profile_path = tmp_path / "profile.json"
    sections = [
        Section("Definitions", "general", "", 1, 1),
        Section("Benefits", "general", "", 2, 2),
        Section("Definitions", "general", "", 3, 3),
        Section("Claims", "general", "", 4, 4),
    ]
    profile = DocumentProfile(
        source_file="doc.pdf", page_count=4, sections=sections
    )
    pipeline.write_profile(profile, str(profile_path))
    # Interrupted run: the first two sections finished.
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(_chunk("Definitions", 0).to_dict()) + "\n")
        f.write(json.dumps(_chunk("Benefits", 1).to_dict()) + "\n")
    payload = {"chunks": [{"text": "t", "keywords": [], "cross_references": []}]}
    client = FakeClient([payload])
    result = pipeline.run(
        str(pdf),
        config=_config(),
        out_path=str(out),
        profile_path=str(profile_path),
        client=client,
        resume=True,
        fidelity=False,
    )
    # The renamed second Definitions and Claims both chunk (2 calls).
    assert len(client.messages.calls) == 2
    assert [c.section_title for c in result.chunks] == [
        "Definitions",
        "Benefits",
        "Definitions (2)",
        "Claims",
    ]


def test_run_resume_with_fidelity_disabled_drops_stale_fidelity_block(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(make_pdf(3))
    profile_path = tmp_path / "profile.json"
    profile = DocumentProfile(
        source_file="doc.pdf", page_count=3, sections=list(SECTIONS)
    )
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(
            {**profile.to_dict(), "fidelity": {"status": "ok", "stale": True}},
            f,
        )
    payload = {"chunks": [{"text": "t", "keywords": [], "cross_references": []}]}
    client = FakeClient([payload])
    pipeline.run(
        str(pdf),
        config=_config(),
        out_path=str(tmp_path / "chunks.jsonl"),
        profile_path=str(profile_path),
        client=client,
        resume=True,
        fidelity=False,
    )
    with open(profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "fidelity" not in data  # the stale block no longer lingers
