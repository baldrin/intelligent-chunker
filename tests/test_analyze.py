from intelligent_chunker import analyze


def _section(title, ps, pe, summary="", stype="general"):
    return {
        "title": title,
        "section_type": stype,
        "summary": summary,
        "page_start": ps,
        "page_end": pe,
    }


def test_merge_sections_folds_overlap_duplicates():
    # Same section seen in two overlapping batches (page 5 overlap).
    raw = [
        _section("Eligibility", 1, 5, summary="short"),
        _section("ELIGIBILITY ", 5, 8, summary="a longer summary"),
        _section("Claims", 9, 12),
    ]
    merged = analyze._merge_sections(raw)
    titles = [s.title for s in merged]
    assert len(merged) == 2
    elig = merged[0]
    assert elig.page_start == 1 and elig.page_end == 8  # ranges unioned
    assert elig.summary == "a longer summary"  # longer summary kept
    assert "Claims" in titles


def test_merge_sections_keeps_distinct_titles():
    raw = [_section("A", 1, 2), _section("B", 3, 4)]
    assert len(analyze._merge_sections(raw)) == 2


def test_merge_sections_orders_by_page():
    raw = [_section("Later", 10, 12), _section("Earlier", 1, 3)]
    merged = analyze._merge_sections(raw)
    assert [s.title for s in merged] == ["Earlier", "Later"]


def test_reconcile_unifies_metadata():
    partials = [
        {
            "doc_type": "SPD",
            "title": "Acme Health Plan",
            "plan_name": "",
            "sponsor": "Acme",
            "effective_dates": ["2024-01-01"],
            "sections": [_section("Intro", 1, 2)],
            "glossary": [{"term": "Deductible", "definition": "amount"}],
            "cross_references": ["see Claims"],
            "notes": "",
        },
        {
            "doc_type": "",
            "title": "",
            "plan_name": "Acme Health",
            "sponsor": "",
            "effective_dates": ["2024-01-01", "2025-01-01"],
            "sections": [_section("Claims", 3, 4)],
            "glossary": [{"term": "deductible", "definition": "dup"}],
            "cross_references": ["See Claims"],
            "notes": "batch 2",
        },
    ]
    profile = analyze.reconcile(partials, source_file="x.pdf", page_count=4)
    assert profile.doc_type == "SPD"
    assert profile.title == "Acme Health Plan"  # first non-empty
    assert profile.plan_name == "Acme Health"
    assert profile.effective_dates == ["2024-01-01", "2025-01-01"]  # deduped
    assert len(profile.glossary) == 1  # case-insensitive term dedupe
    assert len(profile.cross_references) == 1  # case-insensitive string dedupe
    assert [s.title for s in profile.sections] == ["Intro", "Claims"]


def test_analyze_batch_offsets_pages():
    from conftest import FakeClient, make_pdf

    from intelligent_chunker.config import ChunkerConfig
    from intelligent_chunker.pdf_io import PageBatch

    payload = {
        "doc_type": "SPD",
        "title": "T",
        "plan_name": "",
        "sponsor": "",
        "effective_dates": [],
        "sections": [_section("S", 2, 3)],  # batch-relative pages
        "glossary": [],
        "cross_references": [],
        "notes": "",
    }
    client = FakeClient([payload])
    batch = PageBatch(page_start=51, page_end=80, pdf_bytes=make_pdf(2))
    out = analyze.analyze_batch(client, ChunkerConfig(), batch)
    # page_offset = 50, so batch-relative 2..3 -> absolute 52..53
    assert out["sections"][0]["page_start"] == 52
    assert out["sections"][0]["page_end"] == 53


def test_merge_sections_folds_interleaved_duplicates():
    # Duplicate sightings of "X" separated by "Y" in page order must still fold.
    raw = [
        _section("X", 45, 50),
        _section("Y", 46, 48),
        _section("X", 46, 55),
    ]
    merged = analyze._merge_sections(raw)
    assert len(merged) == 2
    x = next(s for s in merged if s.title == "X")
    assert x.page_start == 45 and x.page_end == 55


def _reconcile_titles(sections, page_count):
    partials = [
        {
            "doc_type": "SPD",
            "title": "T",
            "plan_name": "",
            "sponsor": "",
            "effective_dates": [],
            "sections": sections,
            "glossary": [],
            "cross_references": [],
            "notes": "",
        }
    ]
    profile = analyze.reconcile(partials, source_file="x.pdf", page_count=page_count)
    return profile.sections


def test_coverage_guard_fills_gaps_at_start_middle_end():
    # Outline covers only 3-4 and 7-8 of a 10-page document.
    sections = [_section("A", 3, 4), _section("B", 7, 8)]
    out = _reconcile_titles(sections, page_count=10)
    assert [s.title for s in out] == [
        "Unmapped pages 1-2",
        "A",
        "Unmapped pages 5-6",
        "B",
        "Unmapped pages 9-10",
    ]
    unmapped = [s for s in out if s.section_type == "unmapped"]
    assert [(s.page_start, s.page_end) for s in unmapped] == [(1, 2), (5, 6), (9, 10)]


def test_coverage_guard_no_gaps_is_a_no_op():
    sections = [_section("A", 1, 5), _section("B", 6, 10)]
    out = _reconcile_titles(sections, page_count=10)
    assert [s.title for s in out] == ["A", "B"]


def test_coverage_guard_clamps_out_of_bounds_ranges():
    # Model reported a section past the end of the document.
    sections = [_section("A", 1, 8), _section("B", 9, 99)]
    out = _reconcile_titles(sections, page_count=10)
    assert [s.title for s in out] == ["A", "B"]
    assert out[1].page_start == 9 and out[1].page_end == 10


def test_coverage_guard_overlapping_sections_leave_no_false_gaps():
    sections = [_section("A", 1, 6), _section("B", 4, 10)]
    out = _reconcile_titles(sections, page_count=10)
    assert all(s.section_type != "unmapped" for s in out)


def test_sanitize_references_strips_leading_punctuation_and_empties():
    values = [
        ", Section II describes eligibility and entry dates",
        "See Section III, Contributions",
        " ,,  ",
        "see section iii, contributions",  # case-insensitive duplicate
        "— Refer to DOL Regulation §2550.404a-5",
    ]
    out = analyze._sanitize_references(values)
    assert out == [
        "Section II describes eligibility and entry dates",
        "See Section III, Contributions",
        "Refer to DOL Regulation §2550.404a-5",
    ]


def test_reconcile_sanitizes_cross_references():
    partials = [
        {
            "doc_type": "SPD",
            "title": "T",
            "plan_name": "",
            "sponsor": "",
            "effective_dates": [],
            "sections": [_section("A", 1, 2)],
            "glossary": [],
            "cross_references": [",  See Section V, Vesting", ""],
            "notes": "",
        }
    ]
    profile = analyze.reconcile(partials, source_file="x.pdf", page_count=2)
    assert profile.cross_references == ["See Section V, Vesting"]


# --- text-layer grounding ----------------------------------------------------

# Physical 6-page layout: title page, TOC (printed numbering starts at 1
# *after* it, so printed pages run two behind physical), then content. Beta's
# heading sits mid-page 4 (shares the page with Alpha's tail); Gamma's opens
# page 6. Page 4 also body-references Gamma ("in Section III. Gamma") -- a
# non-heading occurrence that must not make Gamma ambiguous.
_PAD = "lorem ipsum dolor sit amet " * 8  # >120 normalized chars

_GROUND_PAGES = [
    "Employee Handbook",
    "Contents: I. Alpha .......... 1  II. Beta .......... 2  "
    "III. Gamma .......... 4",
    "HDR 1\nI.  ALPHA\nalpha body text",
    "HDR 2\n" + _PAD + " as described in Section III. Gamma. "
    "II.Beta\nbeta body text",
    "HDR 3\nmore beta body text",
    "HDR 4\nIII. GAMMA\ngamma body text",
]


def _ground_input():
    from intelligent_chunker.models import Section

    # Beta and Gamma claim printed page numbers (two behind physical), which
    # also missorts Beta ahead of Alpha.
    return [
        Section.from_dict(_section("I. Alpha", 3, 3)),
        Section.from_dict(_section("II. Beta", 2, 3)),
        Section.from_dict(_section("III. Gamma", 4, 4)),
    ]


def test_ground_sections_corrects_printed_page_numbers():
    out = analyze.ground_sections(_ground_input(), _GROUND_PAGES)
    assert [s.title for s in out] == ["I. Alpha", "II. Beta", "III. Gamma"]
    # Alpha ends on 4 (Beta's heading is mid-page: shared); Beta ends on 5
    # (Gamma's heading opens page 6); Gamma's end can't precede its start.
    assert [(s.page_start, s.page_end) for s in out] == [(3, 4), (4, 5), (6, 6)]


def test_ground_sections_skips_ambiguous_and_unmatched_headings():
    from intelligent_chunker.models import Section

    pages = [
        "Renewal Notice for the plan",
        "some other content",
        "Renewal Notice appears again",
    ]
    sections = [
        Section.from_dict(_section("Renewal Notice", 2, 2)),  # two hits
        Section.from_dict(_section("Nowhere Heading", 3, 3)),  # zero hits
    ]
    out = analyze.ground_sections(sections, pages)
    assert [(s.page_start, s.page_end) for s in out] == [(2, 2), (3, 3)]


def test_ground_sections_no_text_layer_is_a_no_op():
    out = analyze.ground_sections(_ground_input(), ["", "", "", "", "", ""])
    # Scanned PDF: everything untouched (callers pre-sort via _merge_sections).
    assert [s.title for s in out] == ["I. Alpha", "II. Beta", "III. Gamma"]
    assert [(s.page_start, s.page_end) for s in out] == [(3, 3), (2, 3), (4, 4)]


def test_reconcile_grounds_before_filling_gaps():
    partials = [
        {
            "doc_type": "SPD",
            "title": "T",
            "plan_name": "",
            "sponsor": "",
            "effective_dates": [],
            "sections": [
                _section("I. Alpha", 3, 3),
                _section("II. Beta", 2, 3),
                _section("III. Gamma", 4, 4),
            ],
            "glossary": [],
            "cross_references": [],
            "notes": "",
        }
    ]
    profile = analyze.reconcile(
        partials, source_file="x.pdf", page_count=6, page_texts=_GROUND_PAGES
    )
    # Grounded outline first, then the title/TOC pages fall out as unmapped.
    assert [s.title for s in profile.sections] == [
        "Unmapped pages 1-2",
        "I. Alpha",
        "II. Beta",
        "III. Gamma",
    ]
    assert [(s.page_start, s.page_end) for s in profile.sections] == [
        (1, 2),
        (3, 4),
        (4, 5),
        (6, 6),
    ]


def test_analyze_document_parallel_matches_sequential():
    from conftest import FakeClient, make_pdf

    from intelligent_chunker.config import ChunkerConfig

    payload = {
        "doc_type": "SPD",
        "title": "T",
        "plan_name": "",
        "sponsor": "",
        "effective_dates": [],
        "sections": [_section("S", 1, 2)],  # batch-relative
        "glossary": [],
        "cross_references": [],
        "notes": "",
    }
    pdf = make_pdf(4)

    def run_with(concurrency):
        client = FakeClient([payload])
        config = ChunkerConfig(
            max_pages_per_batch=2,
            batch_overlap_pages=0,
            pass1_concurrency=concurrency,
        )
        profile = analyze.analyze_document(client, config, pdf, "x.pdf")
        return client, profile

    seq_client, seq = run_with(1)
    par_client, par = run_with(3)

    # Both modes made one call per batch and reconciled identically:
    # "S" at abs 1-2 (batch 1) and abs 3-4 (batch 2) touch -> one section 1-4.
    assert len(seq_client.messages.calls) == len(par_client.messages.calls) == 2
    assert seq.to_dict() == par.to_dict()
    assert par.page_count == 4
    assert len(par.sections) == 1
    assert par.sections[0].page_start == 1 and par.sections[0].page_end == 4


def test_ground_sections_word_fallback_locates_scrambled_heading():
    from intelligent_chunker.models import Section

    # Field case: a styled schedule banner extracts scrambled/fused, so the
    # contiguous heading never matches -- but its distinctive words are all on
    # the one physical page. Pass 1 claimed printed numbers (two behind).
    pages = [
        "Introduction to the plan overview text",
        "eligibility filler content for the middle",
        "BENEFITS OF SCHEDULE hsa plan deductible rows here",
        "continuation of the schedule table rows",
        "closing filler page",
        "more closing filler",
    ]
    sections = [
        Section.from_dict(_section("Introduction", 1, 2)),
        Section.from_dict(_section("Schedule of Benefits - HSA Plan", 5, 6)),
    ]
    out = analyze.ground_sections(sections, pages)
    schedule = next(s for s in out if "HSA" in s.title)
    assert schedule.page_start == 3  # snapped by the word fallback
    intro = next(s for s in out if s.title == "Introduction")
    assert intro.page_end == 2  # boundary derived: schedule opens page 3


def test_ground_sections_word_fallback_needs_unique_and_rare_words():
    from intelligent_chunker.models import Section

    # Title words spread over many pages (nothing rare) or present on two
    # candidate pages: both cases must leave the model's range untouched.
    pages = ["plan benefits words"] * 5 + [
        "special rider addendum text",
        "special rider addendum text again",
    ]
    sections = [
        Section.from_dict(_section("Plan Benefits", 2, 2)),  # nothing rare
        Section.from_dict(_section("Special Rider Addendum", 1, 1)),  # 2 hits
    ]
    out = analyze.ground_sections(sections, pages)
    assert [(s.page_start, s.page_end) for s in out] == [(1, 1), (2, 2)]
