from intelligent_chunker import pdf_io
from conftest import make_pdf


def test_count_pages():
    assert pdf_io.count_pages(make_pdf(7)) == 7


def test_plan_batches_single_when_small():
    assert pdf_io.plan_batches(40, max_pages_per_batch=50, overlap_pages=1) == [
        (1, 40)
    ]


def test_plan_batches_overlap_and_coverage():
    ranges = pdf_io.plan_batches(120, max_pages_per_batch=50, overlap_pages=1)
    # Every batch is at most 50 pages.
    assert all(end - start + 1 <= 50 for start, end in ranges)
    # Coverage: first page 1, last page reaches the end.
    assert ranges[0][0] == 1
    assert ranges[-1][1] == 120
    # Consecutive batches overlap by exactly one page.
    for (s1, e1), (s2, e2) in zip(ranges, ranges[1:]):
        assert s2 == e1  # 1-page overlap => next start == prev end


def test_slice_pdf_page_counts():
    pdf = make_pdf(10)
    sub = pdf_io.slice_pdf(pdf, 3, 6)
    assert pdf_io.count_pages(sub) == 4  # pages 3,4,5,6


def test_iter_batches_offsets():
    pdf = make_pdf(120)
    batches = pdf_io.iter_batches(pdf, max_pages_per_batch=50, overlap_pages=1)
    assert batches[0].page_start == 1
    assert batches[0].page_offset == 0
    # Second batch starts where the first ended (overlap), offset reflects it.
    assert batches[1].page_offset == batches[1].page_start - 1
    assert pdf_io.count_pages(batches[1].pdf_bytes) == batches[1].num_pages


def test_document_block_shape():
    block = pdf_io.document_block(make_pdf(1))
    assert block["type"] == "document"
    assert block["source"]["media_type"] == "application/pdf"
    # Citations default OFF: the API rejects them alongside structured outputs.
    assert "citations" not in block
    assert "\n" not in block["source"]["data"]  # base64 must be newline-free


def test_document_block_citations_optional():
    block = pdf_io.document_block(make_pdf(1), citations=True)
    assert block["citations"] == {"enabled": True}


def test_slice_to_fit_splits_oversized_ranges():
    from intelligent_chunker.pdf_io import encoded_size, slice_pdf, slice_to_fit

    pdf = make_pdf(8)
    single = encoded_size(slice_pdf(pdf, 1, 1))
    # Budget fits ~2 pages of overhead-heavy blank PDF, forcing splits.
    budget = encoded_size(slice_pdf(pdf, 1, 2))
    batches = slice_to_fit(pdf, 1, 8, budget)
    assert [b.page_start for b in batches][0] == 1
    assert batches[-1].page_end == 8
    # Contiguous, non-overlapping coverage.
    for prev, cur in zip(batches, batches[1:]):
        assert cur.page_start == prev.page_end + 1
    for b in batches:
        assert encoded_size(b.pdf_bytes) <= budget

    # A budget below a single page is an explicit error, not a hang.
    import pytest

    with pytest.raises(ValueError):
        slice_to_fit(pdf, 1, 2, single - 1)
