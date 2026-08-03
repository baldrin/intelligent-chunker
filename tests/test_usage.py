from conftest import FakeClient, FakeUsage, WordCounter, make_pdf

from intelligent_chunker import chunker
from intelligent_chunker.config import ChunkerConfig
from intelligent_chunker.llm import UsageTracker
from intelligent_chunker.models import DocumentProfile, Section


def test_tracker_accumulates_per_model():
    tracker = UsageTracker()
    tracker.record("claude-haiku-4-5", FakeUsage())
    tracker.record("claude-haiku-4-5", FakeUsage())
    assert tracker.calls == 2
    totals = tracker.totals()
    assert totals["input_tokens"] == 200
    assert totals["output_tokens"] == 20
    assert totals["cache_creation_input_tokens"] == 10
    assert totals["cache_read_input_tokens"] == 100


def test_tracker_cost_estimate_known_model():
    tracker = UsageTracker()
    tracker.record("claude-haiku-4-5", FakeUsage())
    # 100 in @ $1 + 10 out @ $5 + 5 write @ $1.25 + 50 read @ $0.10, per MTok.
    expected = (100 * 1.00 + 10 * 5.00 + 5 * 1.25 + 50 * 0.10) / 1_000_000
    assert abs(tracker.estimated_cost_usd() - expected) < 1e-12
    assert "$" in tracker.summary()


def test_tracker_cost_estimate_databricks_models():
    # Databricks-served endpoints bill the same per-token rates; the prefix
    # is version-less so any served revision (e.g. sonnet-4-6) matches.
    tracker = UsageTracker()
    tracker.record("databricks-claude-haiku-4-5", FakeUsage())
    expected = (100 * 1.00 + 10 * 5.00 + 5 * 1.25 + 50 * 0.10) / 1_000_000
    assert abs(tracker.estimated_cost_usd() - expected) < 1e-12

    sonnet = UsageTracker()
    sonnet.record("databricks-claude-sonnet-4-6", FakeUsage())
    expected = (100 * 3.00 + 10 * 15.00 + 5 * 3.75 + 50 * 0.30) / 1_000_000
    assert abs(sonnet.estimated_cost_usd() - expected) < 1e-12

    opus = UsageTracker()
    opus.record("databricks-claude-opus-4-1", FakeUsage())
    expected = (100 * 5.00 + 10 * 25.00 + 5 * 6.25 + 50 * 0.50) / 1_000_000
    assert abs(opus.estimated_cost_usd() - expected) < 1e-12


def test_tracker_unknown_model_reports_tokens_without_cost():
    tracker = UsageTracker()
    tracker.record("some-future-model", FakeUsage())
    assert tracker.estimated_cost_usd() is None
    summary = tracker.summary()
    assert "$" not in summary and "100" in summary


def test_tracker_tolerates_usage_without_cache_fields():
    class Bare:
        input_tokens = 7
        output_tokens = 3
        cache_creation_input_tokens = None
        cache_read_input_tokens = None

    tracker = UsageTracker()
    tracker.record("claude-haiku-4-5", Bare())
    tracker.record("claude-haiku-4-5", None)  # ignored
    assert tracker.calls == 1
    assert tracker.totals()["cache_creation_input_tokens"] == 0


def test_chunk_document_records_usage():
    payload = {"chunks": [{"text": "a b c", "keywords": [], "cross_references": []}]}
    client = FakeClient([payload])
    config = ChunkerConfig(max_tokens=100, target_tokens=50)
    profile = DocumentProfile(
        source_file="x.pdf",
        page_count=2,
        sections=[
            Section("S1", "general", "", 1, 1),
            Section("S2", "general", "", 2, 2),
        ],
    )
    tracker = UsageTracker()
    chunker.chunk_document(
        client, config, make_pdf(2), profile, WordCounter(), usage=tracker
    )
    assert tracker.calls == 2
    assert tracker.totals()["input_tokens"] == 200
