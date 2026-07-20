from conftest import FakeClient, WordCounter, make_pdf

from intelligent_chunker import chunker
from intelligent_chunker.config import ChunkerConfig
from intelligent_chunker.models import DocumentProfile, Section

COUNTER = WordCounter()


def test_short_text_returned_unchanged():
    text = "One small chunk that fits."
    assert chunker.enforce_token_limit(text, 50, COUNTER) == [text]


def test_splits_only_at_sentence_boundaries():
    text = (
        "Alpha beta gamma delta epsilon. "
        "Zeta eta theta iota kappa. "
        "Lambda mu nu xi omicron."
    )  # three 5-word sentences = 15 tokens
    pieces = chunker.enforce_token_limit(text, 10, COUNTER)
    assert len(pieces) >= 2
    for p in pieces:
        assert COUNTER.count(p) <= 10
        assert p.strip().endswith(".")  # each piece ends on a sentence boundary


def test_long_sentence_splits_at_whitespace_without_breaking_words():
    words = [f"word{i}" for i in range(20)]
    sentence = " ".join(words)  # one 20-word "sentence", no internal period
    pieces = chunker.enforce_token_limit(sentence, 7, COUNTER)
    for p in pieces:
        assert COUNTER.count(p) <= 7
    # No word was broken: concatenating the pieces recovers the exact words.
    assert " ".join(pieces).split() == words


def test_chunk_document_applies_token_guard_and_metadata():
    # Model returns one oversized chunk; the guard must split it.
    long_text = " ".join(f"w{i}" for i in range(30))
    payload = {
        "chunks": [
            {"text": long_text, "keywords": ["kw"], "cross_references": ["xref"]}
        ]
    }
    client = FakeClient([payload])
    config = ChunkerConfig(max_tokens=10, target_tokens=8)
    profile = DocumentProfile(
        source_file="x.pdf",
        page_count=2,
        sections=[
            Section(
                title="Eligibility",
                section_type="eligibility",
                summary="who is covered",
                page_start=1,
                page_end=2,
            )
        ],
    )
    chunks = chunker.chunk_document(client, config, make_pdf(2), profile, COUNTER)

    assert len(chunks) > 1  # 30 tokens / 10 max => multiple chunks
    for i, c in enumerate(chunks):
        assert c.token_count <= 10
        assert c.chunk_index == i  # contiguous, document-wide index
        assert c.section_title == "Eligibility"
        assert c.section_type == "eligibility"
        assert c.page_start == 1 and c.page_end == 2
        assert c.keywords == ["kw"]
        assert c.cross_references == ["xref"]


def test_pack_chunks_merges_small_pieces_up_to_target():
    pieces = [
        {"text": "a a a", "keywords": ["k1"], "cross_references": ["x1"]},
        {"text": "b b b", "keywords": ["k2"], "cross_references": []},
        {"text": "c c c", "keywords": ["k1"], "cross_references": ["x2"]},
        {"text": "d d d d d d d d", "keywords": [], "cross_references": []},
    ]
    # target 7 words: "a a a" + "b b b" = 6 ok; + "c c c" = 9 > 7 -> new chunk.
    packed = chunker.pack_chunks(pieces, target_tokens=7, counter=COUNTER)
    assert len(packed) == 3
    assert COUNTER.count(packed[0]["text"]) == 6  # a+b merged
    assert packed[0]["keywords"] == ["k1", "k2"]  # unioned, order-preserved
    assert packed[0]["cross_references"] == ["x1"]
    assert packed[1]["text"] == "c c c"  # c alone (adding it would exceed 7)
    assert packed[2]["text"] == "d d d d d d d d"  # 8 > target, stands alone
    for p in packed:
        # nothing was split; packing only ever combines
        assert p["text"]


def test_pack_chunks_never_exceeds_target_when_pieces_fit():
    pieces = [
        {"text": f"w{i}", "keywords": [], "cross_references": []} for i in range(10)
    ]
    packed = chunker.pack_chunks(pieces, target_tokens=4, counter=COUNTER)
    assert all(COUNTER.count(p["text"]) <= 4 for p in packed)
    # 10 single-word pieces packed 4 per chunk -> 3 chunks (4,4,2)
    assert len(packed) == 3


def test_chunk_document_packs_to_denser_chunks():
    # Model returns many tiny chunks; packing should consolidate them.
    payload = {
        "chunks": [
            {"text": "s s s", "keywords": ["a"], "cross_references": []}
            for _ in range(9)
        ]
    }
    client = FakeClient([payload])
    config = ChunkerConfig(max_tokens=100, target_tokens=12)
    profile = DocumentProfile(
        source_file="x.pdf",
        page_count=1,
        sections=[Section("S", "general", "", 1, 1)],
    )
    chunks = chunker.chunk_document(client, config, make_pdf(1), profile, COUNTER)
    # 9 pieces of 3 tokens = 27 tokens; packed to ~12 -> far fewer than 9 chunks.
    assert len(chunks) < 9
    for c in chunks:
        assert c.token_count <= 12


def test_global_context_includes_section_and_map():
    profile = DocumentProfile(
        source_file="x.pdf",
        page_count=3,
        title="Acme Plan",
        doc_type="SPD",
        sections=[
            Section("Intro", "intro", "", 1, 1),
            Section("Claims", "claims", "how to file", 2, 3),
        ],
    )
    ctx = chunker._global_context(profile, profile.sections[1])
    assert "Acme Plan" in ctx
    assert "Section outline" in ctx
    assert "CURRENT SECTION" in ctx
    assert "Claims" in ctx


def test_split_long_sentence_emits_oversized_token_alone():
    # One unbreakable 1-word "sentence" over the limit is emitted as-is.
    class CharCounter:
        def count(self, text):
            return len(text)

    pieces = chunker._split_long_sentence("aaa bbbbbbbbbb cc", 5, CharCounter())
    assert "bbbbbbbbbb" in pieces  # emitted alone, not word-broken
    assert all(" " not in p or CharCounter().count(p) <= 5 for p in pieces)


def test_chunk_document_reports_sections_incrementally():
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
    seen = []
    chunks = chunker.chunk_document(
        client, config, make_pdf(2), profile, COUNTER,
        on_section=lambda cs: seen.append(len(cs)),
    )
    assert len(seen) == 2  # called once per section
    assert sum(seen) == len(chunks)
