import pytest
from conftest import FakeClient, WordCounter, make_pdf

from intelligent_chunker import chunker
from intelligent_chunker.config import ChunkerConfig
from intelligent_chunker.fidelity import match_key
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


def test_chunk_document_parallel_preserves_section_order():
    # One repeated payload keeps the racy call-counting fake deterministic.
    payload = {"chunks": [{"text": "a b c", "keywords": [], "cross_references": []}]}
    client = FakeClient([payload])
    config = ChunkerConfig(max_tokens=100, target_tokens=50, pass2_concurrency=3)
    profile = DocumentProfile(
        source_file="x.pdf",
        page_count=4,
        sections=[Section(f"S{i}", "general", "", i, i) for i in range(1, 5)],
    )
    seen_sections = []
    chunks = chunker.chunk_document(
        client, config, make_pdf(4), profile, COUNTER,
        on_section=lambda cs: seen_sections.append(cs[0].section_title),
    )
    assert len(client.messages.calls) == 4
    # Results are consumed in section order regardless of completion order.
    assert seen_sections == ["S1", "S2", "S3", "S4"]
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]
    assert [c.section_title for c in chunks] == ["S1", "S2", "S3", "S4"]


def test_normalize_chunk_pages_offsets_and_clamps():
    raw = [
        # Slice-relative 1-2 with offset 4 -> absolute 5-6.
        {"text": "a", "page_start": 1, "page_end": 2},
        # End past the section: clamped down to sec_end.
        {"text": "b", "page_start": 3, "page_end": 9},
        # Missing pages -> falls back to the section range.
        {"text": "c"},
        # Inverted range -> falls back to the section range.
        {"text": "d", "page_start": 4, "page_end": 1},
    ]
    out = chunker._normalize_chunk_pages(raw, offset=4, sec_start=5, sec_end=8)
    assert [(c["page_start"], c["page_end"]) for c in out] == [
        (5, 6),
        (7, 8),
        (5, 8),
        (5, 8),
    ]


def test_pack_chunks_merges_page_ranges():
    pieces = [
        {"text": "a b", "keywords": [], "cross_references": [],
         "page_start": 3, "page_end": 3},
        {"text": "c d", "keywords": [], "cross_references": [],
         "page_start": 4, "page_end": 5},
    ]
    packed = chunker.pack_chunks(pieces, target_tokens=10, counter=COUNTER)
    assert len(packed) == 1
    assert packed[0]["page_start"] == 3 and packed[0]["page_end"] == 5


def test_pack_chunks_without_pages_yields_none():
    pieces = [{"text": "a b", "keywords": [], "cross_references": []}]
    packed = chunker.pack_chunks(pieces, target_tokens=10, counter=COUNTER)
    assert packed[0]["page_start"] is None and packed[0]["page_end"] is None


def test_chunk_document_uses_model_pages_and_falls_back_to_section():
    payload = {
        "chunks": [
            {"text": "a b c", "keywords": [], "cross_references": [],
             "page_start": 2, "page_end": 2},
            {"text": "d e f", "keywords": [], "cross_references": []},
        ]
    }
    client = FakeClient([payload])
    config = ChunkerConfig(max_tokens=100, target_tokens=3)
    profile = DocumentProfile(
        source_file="x.pdf",
        page_count=3,
        sections=[Section("S", "general", "", 1, 3)],
    )
    chunks = chunker.chunk_document(client, config, make_pdf(3), profile, COUNTER)
    assert (chunks[0].page_start, chunks[0].page_end) == (2, 2)
    # No model pages -> normalize fell back to the whole section range.
    assert (chunks[1].page_start, chunks[1].page_end) == (1, 3)


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


# --- per-chunk page grounding ------------------------------------------------

_P2 = "The quick brown fox paragraph starts here and continues onward"
_P3 = "eventually that same paragraph finishes with distinct closing words"

_NORM_PAGES = [match_key(t) for t in [
    "page one has entirely different filler content about eligibility",
    _P2,
    _P3,
    "page four is unrelated trailing material",
]]


def test_ground_chunk_pages_corrects_both_edges():
    # Text physically spans pages 2-3 but the model claimed (1, 1).
    pieces = [{"text": _P2 + " " + _P3, "page_start": 1, "page_end": 1}]
    out = chunker.ground_chunk_pages(pieces, _NORM_PAGES, 1, 4)
    assert (out[0]["page_start"], out[0]["page_end"]) == (2, 3)


def test_ground_chunk_pages_fills_missing_pages():
    pieces = [{"text": _P2, "page_start": None, "page_end": None}]
    out = chunker.ground_chunk_pages(pieces, _NORM_PAGES, 1, 4)
    assert (out[0]["page_start"], out[0]["page_end"]) == (2, 2)


def test_ground_chunk_pages_keeps_ambiguous_and_unmatched_edges():
    dup = match_key("repeated boilerplate appears on more than one page")
    norm = [dup, dup, match_key(_P3)]
    pieces = [
        # Prefix ambiguous (pages 1 and 2): start kept; suffix unique: end set.
        {"text": "repeated boilerplate appears on more than one page " + _P3,
         "page_start": 1, "page_end": 1},
        # Nothing matches anywhere: untouched.
        {"text": "words that exist nowhere in the layer at all honestly",
         "page_start": 2, "page_end": 2},
        # Too short to trust: untouched.
        {"text": "tiny", "page_start": 2, "page_end": 2},
    ]
    out = chunker.ground_chunk_pages(pieces, norm, 1, 3)
    assert (out[0]["page_start"], out[0]["page_end"]) == (1, 3)
    assert (out[1]["page_start"], out[1]["page_end"]) == (2, 2)
    assert (out[2]["page_start"], out[2]["page_end"]) == (2, 2)


def test_ground_chunk_pages_repairs_contradicted_model_end():
    # Start grounds to page 3; the model's end of 1 can't be right.
    pieces = [{"text": _P3, "page_start": 1, "page_end": 1}]
    out = chunker.ground_chunk_pages(pieces, _NORM_PAGES, 1, 4)
    assert (out[0]["page_start"], out[0]["page_end"]) == (3, 3)


def test_ground_chunk_pages_searches_only_the_section_range():
    # Same text exists on pages 2 and 4; a section spanning only 1-3 sees a
    # unique hit, so the section range disambiguates.
    norm = [match_key("filler"), match_key(_P2), match_key("x"), match_key(_P2)]
    pieces = [{"text": _P2, "page_start": 1, "page_end": 1}]
    out = chunker.ground_chunk_pages(pieces, norm, 1, 3)
    assert (out[0]["page_start"], out[0]["page_end"]) == (2, 2)


# --- TOC noise cleaning ------------------------------------------------------

# Real shape from the sample SPD: a single line of heading + leader + number
# entries (the text layer keeps no line breaks), including pypdf's stray
# ". 8" artifact after a leader run.
_TOC_TEXT = (
    "SUMMARY PLAN DESCRIPTION OVERVIEW .................... 1 "
    "I. BASIC PLAN INFORMATION .................... 2 "
    "II. PARTICIPATION .................... 4 "
    "V. VESTING .................... . 8"
)


def test_strip_dot_leaders_cleans_runs_but_keeps_ellipses():
    assert chunker.strip_dot_leaders("A .......... 4 B") == "A 4 B"
    assert chunker.strip_dot_leaders("wait... what") == "wait... what"
    assert chunker.strip_dot_leaders("End of sentence.") == "End of sentence."


def test_is_toc_text_detects_single_line_toc():
    assert chunker.is_toc_text(_TOC_TEXT)
    # Prose with a couple of dotted rows is not a TOC.
    prose = ("The plan invests in options described below. " * 8
             + "Fund A .......... 12 Fund B .......... 34")
    assert not chunker.is_toc_text(prose)
    assert not chunker.is_toc_text("Alpha .......... 1 Beta .......... 2")


def test_clean_raw_chunks_drops_toc_only_in_unmapped_sections():
    unmapped = Section("Unmapped pages 1-2", "unmapped", "", 1, 2)
    mapped = Section("Overview", "overview", "", 1, 2)
    raw = [
        {"text": "SUMMARY PLAN DESCRIPTION\nAHS Plan"},  # title page: kept
        {"text": _TOC_TEXT},
    ]
    out = chunker.clean_raw_chunks(list(raw), unmapped)
    assert [c["text"] for c in out] == ["SUMMARY PLAN DESCRIPTION\nAHS Plan"]
    # Same TOC text in a mapped section survives, but with leaders stripped.
    out = chunker.clean_raw_chunks(list(raw), mapped)
    assert len(out) == 2
    assert "...." not in out[1]["text"]
    assert "II. PARTICIPATION 4" in out[1]["text"]


def test_chunk_document_drops_toc_and_counts_clean_tokens():
    payload = {
        "chunks": [
            {"text": "Title page words", "keywords": [], "cross_references": []},
            {"text": _TOC_TEXT, "keywords": [], "cross_references": []},
        ]
    }
    client = FakeClient([payload])
    config = ChunkerConfig(max_tokens=100, target_tokens=5)
    profile = DocumentProfile(
        source_file="x.pdf",
        page_count=2,
        sections=[Section("Unmapped pages 1-2", "unmapped", "", 1, 2)],
    )
    chunks = chunker.chunk_document(client, config, make_pdf(2), profile, COUNTER)
    assert len(chunks) == 1
    assert chunks[0].text == "Title page words"


# --- section-level dedup -----------------------------------------------------

# A paragraph long enough to be subject to dedup (>= 25 words).
_LONG_PARA = " ".join(f"tax withholding rule {i}" for i in range(10))


def test_dedupe_drops_chunk_reemitted_verbatim():
    raw = [
        {"text": _LONG_PARA, "keywords": ["a"], "cross_references": []},
        # Same text with different wrapping/case still counts as a re-emit.
        {"text": _LONG_PARA.upper().replace(" ", "  "), "keywords": ["b"],
         "cross_references": []},
    ]
    out = chunker.dedupe_raw_chunks(raw, "S")
    assert len(out) == 1
    assert out[0]["keywords"] == ["a"]  # first occurrence wins


def test_dedupe_removes_paragraphs_copied_from_earlier_chunk():
    own = "Installment distributions are paid in substantially equal amounts."
    raw = [
        {"text": "intro\n\n" + _LONG_PARA, "keywords": [], "cross_references": []},
        # Model padded a later chunk by restating the earlier paragraph.
        {"text": own + "\n\n" + _LONG_PARA, "keywords": [], "cross_references": []},
    ]
    out = chunker.dedupe_raw_chunks(raw, "S")
    assert len(out) == 2
    assert out[1]["text"] == own
    # A chunk that was *only* copied paragraphs disappears entirely.
    raw.append({"text": _LONG_PARA, "keywords": [], "cross_references": []})
    assert len(chunker.dedupe_raw_chunks(raw, "S")) == 2


def test_dedupe_keeps_short_repeating_blocks():
    # Table headers / schedule rows repeat legitimately and must survive.
    row = "Years of Service | Vesting Percentage\nless than 1 | 100.00"
    raw = [
        {"text": "Schedule A:\n\n" + row, "keywords": [], "cross_references": []},
        {"text": "Schedule B:\n\n" + row, "keywords": [], "cross_references": []},
    ]
    out = chunker.dedupe_raw_chunks(raw, "S")
    assert len(out) == 2
    assert row in out[0]["text"] and row in out[1]["text"]


def test_chunk_document_dedupes_within_section():
    payload = {
        "chunks": [
            {"text": _LONG_PARA, "keywords": [], "cross_references": []},
            {"text": _LONG_PARA, "keywords": [], "cross_references": []},
        ]
    }
    client = FakeClient([payload])
    config = ChunkerConfig(max_tokens=200, target_tokens=100)
    profile = DocumentProfile(
        source_file="x.pdf",
        page_count=1,
        sections=[Section("S", "general", "", 1, 1)],
    )
    chunks = chunker.chunk_document(client, config, make_pdf(1), profile, COUNTER)
    combined = "\n".join(c.text for c in chunks)
    assert combined.count(_LONG_PARA) == 1


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


def _two_section_profile() -> DocumentProfile:
    return DocumentProfile(
        source_file="x.pdf",
        page_count=2,
        sections=[
            Section(
                title="Eligibility",
                section_type="eligibility",
                summary="who is covered",
                page_start=1,
                page_end=1,
            ),
            Section(
                title="Benefits",
                section_type="benefits",
                summary="what is covered",
                page_start=2,
                page_end=2,
            ),
        ],
    )


def test_context_overflow_falls_back_to_sliced_mode():
    # A document can pass the 100-page cap for cached full-document mode yet
    # overflow the context window (pages bill text + image tokens). The API's
    # 400 must downgrade the run to sliced sections, not kill it.
    overflow = Exception(
        "Error code: 400 - prompt is too long: 200275 tokens > 200000 maximum"
    )
    payload = {"chunks": [{"text": "hello world"}]}
    client = FakeClient([overflow, payload, payload])
    config = ChunkerConfig(pass2_concurrency=1)

    chunks = chunker.chunk_document(
        client, config, make_pdf(2), _two_section_profile(), COUNTER
    )

    assert len(chunks) == 2  # both sections still chunked
    calls = client.messages.calls
    assert len(calls) == 3  # failed full-doc call + one sliced call per section
    doc_blocks = [c["messages"][0]["content"][0] for c in calls]
    assert "cache_control" in doc_blocks[0]  # the full-document attempt
    assert "cache_control" not in doc_blocks[1]  # retried sliced
    assert "cache_control" not in doc_blocks[2]  # later sections never retry


def test_non_overflow_errors_still_fail_the_section():
    client = FakeClient([Exception("boom")])
    config = ChunkerConfig(pass2_concurrency=1)
    with pytest.raises(RuntimeError, match="Pass 2 failed on section"):
        list(
            chunker.chunk_document(
                client, config, make_pdf(2), _two_section_profile(), COUNTER
            )
        )
def test_ground_chunk_pages_finds_text_drifted_outside_the_section():
    # Field case: the model labeled a chunk one page late (printed page
    # numbers offset from physical position), so its text sits just before
    # the section's claimed range. The in-range search finds nothing; the
    # widened retry must locate and correct it.
    norm = [
        match_key(_P2),  # the chunk's text physically lives on page 1
        match_key("the section's claimed pages hold different material"),
        match_key("closing filler that matches nothing in the chunk"),
    ]
    pieces = [{"text": _P2, "page_start": 2, "page_end": 2}]
    out = chunker.ground_chunk_pages(pieces, norm, 2, 3)
    assert (out[0]["page_start"], out[0]["page_end"]) == (1, 1)


def test_ground_chunk_pages_widened_retry_still_requires_uniqueness():
    # The same drifted text appears on both sides of the section: the wide
    # pass sees two hits, so the model's value is kept rather than guessed.
    norm = [match_key(_P2), match_key("mid filler"), match_key(_P2)]
    pieces = [{"text": _P2, "page_start": 2, "page_end": 2}]
    out = chunker.ground_chunk_pages(pieces, norm, 2, 2)
    assert (out[0]["page_start"], out[0]["page_end"]) == (2, 2)


# --- word-overlap grounding fallback -----------------------------------------

_SENTENCE = (
    "(ii) Orthopedic shoes or custom-molded foot orthotics are covered "
    "when prescribed to prevent complications associated with diabetes"
)
# The text layer holds the same words SCRAMBLED (hanging-indent extraction),
# so contiguous substring matching cannot find them.
_SCRAMBLED = (
    "orthotics diabetes prescribed molded custom orthopedic "
    "shoes complications covered foot"
)
_FALLBACK_RAW = [
    "welcome cover page filler content here",
    "claimed page holds unrelated eligibility words",
    "middle page other unrelated material words",
    _SCRAMBLED,
    "trailing page content words again",
]
_FALLBACK_NORM = [match_key(t) for t in _FALLBACK_RAW]


def test_word_fallback_locates_scrambled_extraction():
    # Field case: printed page numbers put the chunk 2 pages early, and the
    # layer's word order is scrambled so substring grounding finds nothing.
    pieces = [{"text": _SENTENCE, "page_start": 2, "page_end": 2}]
    out = chunker.ground_chunk_pages(
        pieces, _FALLBACK_NORM, 1, 2, raw_pages=_FALLBACK_RAW
    )
    assert (out[0]["page_start"], out[0]["page_end"]) == (4, 4)


def test_word_fallback_requires_raw_pages():
    pieces = [{"text": _SENTENCE, "page_start": 2, "page_end": 2}]
    out = chunker.ground_chunk_pages(pieces, _FALLBACK_NORM, 1, 2)
    assert (out[0]["page_start"], out[0]["page_end"]) == (2, 2)


def test_word_fallback_spans_adjacent_pages():
    raw = list(_FALLBACK_RAW)
    raw[3] = "orthotics diabetes prescribed molded custom orthopedic"
    raw[4] = "shoes complications covered foot orthopedic extras"
    norm = [match_key(t) for t in raw]
    pieces = [{"text": _SENTENCE, "page_start": 2, "page_end": 2}]
    out = chunker.ground_chunk_pages(pieces, norm, 1, 2, raw_pages=raw)
    assert (out[0]["page_start"], out[0]["page_end"]) == (4, 5)


def test_word_fallback_ignores_generic_text():
    # A piece with no distinctive vocabulary must not be relocated on the
    # strength of boilerplate words alone.
    generic = "the plan will provide coverage for services under this plan"
    raw = ["plan coverage services provide under this the will for words"] * 5
    norm = [match_key(t) for t in raw]
    pieces = [{"text": generic, "page_start": 2, "page_end": 2}]
    out = chunker.ground_chunk_pages(pieces, norm, 1, 2, raw_pages=raw)
    assert (out[0]["page_start"], out[0]["page_end"]) == (2, 2)


def test_chunk_document_reports_section_progress():
    payload = {"chunks": [{"text": "a b c", "keywords": [], "cross_references": []}]}
    client = FakeClient([payload])
    config = ChunkerConfig(max_tokens=100, target_tokens=50, pass2_concurrency=2)
    profile = DocumentProfile(
        source_file="x.pdf",
        page_count=2,
        sections=[Section(f"S{i}", "general", "", i, i) for i in (1, 2)],
    )
    events = []
    chunker.chunk_document(
        client, config, make_pdf(2), profile, COUNTER,
        on_progress=lambda d, t: events.append((d, t)),
    )
    # Results are consumed in section order, so progress is deterministic.
    assert events == [(0, 2), (1, 2), (2, 2)]
