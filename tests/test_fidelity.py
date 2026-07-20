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
