from conftest import make_pdf

from intelligent_chunker import fidelity
from intelligent_chunker.models import Chunk, DocumentProfile, Section


def test_identical_text_scores_perfectly():
    text = "The plan covers eligible employees after one year of service."
    score = fidelity.score_texts(text, text)
    assert score["coverage"] == 1.0
    assert score["novelty"] == 0.0


def test_disjoint_text_scores_zero():
    score = fidelity.score_texts("alpha beta gamma", "delta epsilon")
    assert score["coverage"] == 0.0
    assert score["novelty"] == 1.0


def test_partial_overlap():
    # Reference has 4 words, candidate reproduces 2 of them + 2 new ones.
    score = fidelity.score_texts("one two three four", "one two five six")
    assert score["coverage"] == 0.5
    assert score["novelty"] == 0.5


def test_scores_ignore_case_and_punctuation():
    score = fidelity.score_texts(
        "Vesting: 100% after 3 years.", "vesting 100 after 3 years"
    )
    assert score["coverage"] == 1.0
    assert score["novelty"] == 0.0


def test_repeated_words_count_as_multiset():
    # Candidate repeats "pay" more than the reference contains.
    score = fidelity.score_texts("pay period", "pay pay period")
    assert score["coverage"] == 1.0
    assert round(score["novelty"], 2) == 0.33


def test_empty_candidate_is_full_miss():
    score = fidelity.score_texts("some reference words", "")
    assert score["coverage"] == 0.0
    assert score["novelty"] == 0.0  # nothing invented either


def test_shared_page_detection():
    sections = [
        Section("A", "general", "", 1, 3),
        Section("B", "general", "", 3, 5),   # shares page 3 with A
        Section("C", "general", "", 6, 7),   # exclusive
        Section("D", "general", "", 6, 6),   # nested inside C's range
    ]
    assert fidelity.sections_with_shared_pages(sections) == [
        True,
        True,
        True,
        True,
    ]
    exclusive = [
        Section("A", "general", "", 1, 2),
        Section("B", "general", "", 3, 4),
    ]
    assert fidelity.sections_with_shared_pages(exclusive) == [False, False]


# --- per-chunk novelty -------------------------------------------------------

_PAGE = (
    "The trustee is responsible for trusteeing the plan assets and holds "
    "them in possession under the trust agreement for participants."
)


def _chunk(idx, text, ps=1, pe=1, section="S"):
    return Chunk(
        text=text,
        source_file="x.pdf",
        chunk_index=idx,
        section_title=section,
        section_type="general",
        section_summary="",
        page_start=ps,
        page_end=pe,
    )


def _profile(sections):
    return DocumentProfile(source_file="x.pdf", page_count=2, sections=sections)


def test_chunk_novelty_clean_chunk_not_flagged():
    profile = _profile([Section("S", "general", "", 1, 1)])
    flags = fidelity.chunk_novelty_flags([_PAGE], profile, [_chunk(0, _PAGE)])
    assert flags == []


def test_chunk_novelty_flags_invented_line_and_reports_it():
    invented = "Call the hotline at 555-0199 to claim your wellness voucher."
    text = _PAGE + "\n" + invented
    profile = _profile([Section("S", "general", "", 1, 1)])
    flags = fidelity.chunk_novelty_flags([_PAGE], profile, [_chunk(0, text)])
    assert len(flags) == 1
    assert flags[0]["chunk_index"] == 0
    assert invented in flags[0]["novel_lines"]


def test_chunk_novelty_flags_wrong_page_attribution():
    other_page = "Completely different content about claims and appeals here."
    profile = _profile([Section("S", "general", "", 1, 2)])
    # Chunk text lives on page 1 but claims page 2.
    flags = fidelity.chunk_novelty_flags(
        [_PAGE, other_page], profile, [_chunk(0, _PAGE, ps=2, pe=2)]
    )
    assert len(flags) == 1
    assert flags[0]["novelty"] > fidelity.NOVELTY_WARN_ABOVE


def test_chunk_novelty_skips_unmapped_sections_and_tiny_chunks():
    profile = _profile(
        [
            Section("Unmapped pages 1-2", "unmapped", "", 1, 1),
            Section("S", "general", "", 2, 2),
        ]
    )
    chunks = [
        # Junk in an unmapped section: skipped even though fully novel.
        _chunk(0, _PAGE, section="Unmapped pages 1-2"),
        # Tiny chunk: below the word floor, skipped.
        _chunk(1, "short novel fragment here", ps=2, pe=2),
    ]
    flags = fidelity.chunk_novelty_flags(["different words"] * 2, profile, chunks)
    assert flags == []


def test_unmapped_section_low_coverage_does_not_warn(monkeypatch, caplog):
    import logging

    # Two exclusive-page sections whose chunks barely cover their pages:
    # coverage is low for both, but only the mapped one should warn --
    # proving the new gate keys on section_type, not the score.
    # Long enough combined to clear the scanned-PDF floor (200 chars).
    page1 = "reference words the chunk will mostly fail to cover on page one " * 3
    page2 = "distinct reference words the chunk also fails to cover on page two " * 3
    monkeypatch.setattr(fidelity, "extract_page_texts", lambda b: [page1, page2])
    profile = DocumentProfile(
        source_file="x.pdf",
        page_count=2,
        sections=[
            Section("Unmapped pages 1-1", "unmapped", "", 1, 1),
            Section("Mapped", "general", "", 2, 2),
        ],
    )
    tiny = "barely overlap"
    chunks = [
        _chunk(0, tiny, ps=1, pe=1, section="Unmapped pages 1-1"),
        _chunk(1, tiny, ps=2, pe=2, section="Mapped"),
    ]
    with caplog.at_level(logging.WARNING, logger="intelligent_chunker.fidelity"):
        report = fidelity.fidelity_report(make_pdf(2), profile, chunks)

    coverage_warnings = [
        r.getMessage() for r in caplog.records if "coverage" in r.getMessage()
    ]
    assert any("Mapped" in m for m in coverage_warnings)  # mapped still warns
    assert not any("Unmapped" in m for m in coverage_warnings)
    # Both scores are still reported regardless of warning suppression.
    titles = {s["title"] for s in report["sections"]}
    assert titles == {"Unmapped pages 1-1", "Mapped"}


def test_report_skips_pdfs_without_text_layer():
    # conftest's blank-page PDFs have no text layer -> scanned-PDF path.
    profile = DocumentProfile(
        source_file="x.pdf",
        page_count=2,
        sections=[Section("S", "general", "", 1, 2)],
    )
    chunk = Chunk(
        text="anything",
        source_file="x.pdf",
        chunk_index=0,
        section_title="S",
        section_type="general",
        section_summary="",
        page_start=1,
        page_end=2,
    )
    report = fidelity.fidelity_report(make_pdf(2), profile, [chunk])
    assert report["status"] == "skipped"
    assert "text layer" in report["reason"]
