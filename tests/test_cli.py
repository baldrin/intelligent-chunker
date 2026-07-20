import json

from conftest import make_pdf

from intelligent_chunker import cli, fidelity
from intelligent_chunker.models import Chunk, DocumentProfile, Section

# Long enough to clear the fidelity module's scanned-PDF floor (200 chars)
# and the per-chunk novelty word floor (20 words).
_TEXT = "The plan covers eligible employees after one year of service. " * 5


def _write_fixtures(tmp_path, chunk_text=_TEXT):
    profile = DocumentProfile(
        source_file="x.pdf",
        page_count=1,
        sections=[Section("S", "general", "", 1, 1)],
    )
    chunk = Chunk(
        text=chunk_text,
        source_file="x.pdf",
        chunk_index=0,
        section_title="S",
        section_type="general",
        section_summary="",
        page_start=1,
        page_end=1,
    )
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(make_pdf(1))
    prof = tmp_path / "profile.json"
    prof.write_text(json.dumps(profile.to_dict()))
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps(chunk.to_dict()) + "\n")
    return pdf, prof, chunks


def _run(pdf, prof, chunks, *extra):
    return cli.main(
        ["fidelity", str(pdf), "--profile", str(prof), "--chunks", str(chunks)]
        + list(extra)
    )


def test_fidelity_command_recomputes_and_writes_block(tmp_path, monkeypatch, capsys):
    pdf, prof, chunks = _write_fixtures(tmp_path)
    # Blank test PDFs have no text layer; fake one matching the chunk.
    monkeypatch.setattr(fidelity, "extract_page_texts", lambda b: [_TEXT])
    assert _run(pdf, prof, chunks) == 0
    out = capsys.readouterr().out
    assert "coverage 1.0000" in out
    assert "Flagged chunks: none." in out
    written = json.loads(prof.read_text())
    assert written["fidelity"]["status"] == "ok"
    assert written["fidelity"]["document"]["coverage"] == 1.0
    assert written["fidelity"]["chunks"] == []


def test_fidelity_command_reports_flagged_chunks(tmp_path, monkeypatch, capsys):
    invented = "Call the hotline at 555-0199 to claim your wellness voucher."
    pdf, prof, chunks = _write_fixtures(tmp_path, chunk_text=_TEXT + "\n" + invented)
    monkeypatch.setattr(fidelity, "extract_page_texts", lambda b: [_TEXT])
    assert _run(pdf, prof, chunks) == 0
    out = capsys.readouterr().out
    assert "Flagged chunks: 1" in out
    assert "555" in out  # the offending line is quoted
    written = json.loads(prof.read_text())
    assert written["fidelity"]["chunks"][0]["chunk_index"] == 0


def test_fidelity_command_skips_scanned_pdfs(tmp_path, capsys):
    pdf, prof, chunks = _write_fixtures(tmp_path)  # blank PDF: no text layer
    assert _run(pdf, prof, chunks) == 0
    assert "skipped" in capsys.readouterr().out
    assert json.loads(prof.read_text())["fidelity"]["status"] == "skipped"


def test_fidelity_command_no_write_leaves_profile_untouched(tmp_path, monkeypatch):
    pdf, prof, chunks = _write_fixtures(tmp_path)
    monkeypatch.setattr(fidelity, "extract_page_texts", lambda b: [_TEXT])
    before = prof.read_text()
    assert _run(pdf, prof, chunks, "--no-write") == 0
    assert prof.read_text() == before


def test_fidelity_command_missing_input_fails(tmp_path, capsys):
    pdf, prof, chunks = _write_fixtures(tmp_path)
    assert _run(pdf, prof, tmp_path / "nope.jsonl") == 1
    assert "Input not found" in capsys.readouterr().err
