"""PDF input handling -- batching only, never text extraction.

The model reads the PDF; this module just gets bytes to it. For long documents
we slice page-range batches (so each API call stays within the native PDF
block limits) using ``pypdf``, which copies page objects without interpreting
their text. Page numbers are 1-indexed and inclusive throughout.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pypdf import PdfReader, PdfWriter


@dataclass
class PageBatch:
    """A contiguous page range and the sub-PDF bytes that contain it.

    ``page_start`` is the absolute (document-wide) 1-indexed page of the first
    page in this batch. Inside ``pdf_bytes`` that same page appears as page 1,
    so add ``page_start - 1`` to convert a batch-relative page number back to
    an absolute one.
    """

    page_start: int  # absolute, 1-indexed, inclusive
    page_end: int    # absolute, 1-indexed, inclusive
    pdf_bytes: bytes

    @property
    def num_pages(self) -> int:
        return self.page_end - self.page_start + 1

    @property
    def page_offset(self) -> int:
        """Add to a batch-relative page number to get an absolute page."""
        return self.page_start - 1


def read_pdf(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def count_pages(pdf_bytes: bytes) -> int:
    return len(PdfReader(io.BytesIO(pdf_bytes)).pages)


def plan_batches(
    page_count: int, max_pages_per_batch: int, overlap_pages: int
) -> List[Tuple[int, int]]:
    """Plan (start, end) 1-indexed inclusive page ranges covering the document.

    Consecutive batches overlap by ``overlap_pages`` so a section straddling a
    boundary is fully visible to at least one batch.
    """
    if page_count <= 0:
        return []
    if max_pages_per_batch < 1:
        raise ValueError("max_pages_per_batch must be >= 1")
    overlap = max(0, min(overlap_pages, max_pages_per_batch - 1))
    stride = max_pages_per_batch - overlap

    batches: List[Tuple[int, int]] = []
    start = 1
    while start <= page_count:
        end = min(start + max_pages_per_batch - 1, page_count)
        batches.append((start, end))
        if end >= page_count:
            break
        start += stride
    return batches


def encoded_size(pdf_bytes: bytes) -> int:
    """Size of ``pdf_bytes`` after base64 encoding (what the API request pays)."""
    return 4 * ((len(pdf_bytes) + 2) // 3)


def _slice(reader: PdfReader, start: int, end: int) -> bytes:
    """Slice pages [start, end] (1-indexed) from an already-parsed reader."""
    writer = PdfWriter()
    for i in range(start - 1, end):  # pypdf pages are 0-indexed
        writer.add_page(reader.pages[i])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def slice_pdf(pdf_bytes: bytes, start: int, end: int) -> bytes:
    """Return a new PDF containing only pages [start, end] (1-indexed)."""
    return _slice(PdfReader(io.BytesIO(pdf_bytes)), start, end)


def slice_to_fit(
    pdf_bytes: bytes, start: int, end: int, max_encoded_bytes: int
) -> List[PageBatch]:
    """Slice [start, end], splitting further until every piece fits the budget.

    Scanned-image pages can be large enough that a page-count-based batch
    blows past the platform's request-size limit (Anthropic API / Azure AI
    Foundry: 32 MB; Databricks model serving: ~4 MB — see config). Ranges are
    halved recursively until each slice's *encoded* size fits.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return _slice_to_fit(reader, start, end, max_encoded_bytes)


def _slice_to_fit(
    reader: PdfReader, start: int, end: int, max_encoded_bytes: int
) -> List[PageBatch]:
    sub = _slice(reader, start, end)
    if encoded_size(sub) <= max_encoded_bytes:
        return [PageBatch(page_start=start, page_end=end, pdf_bytes=sub)]
    if start == end:
        raise ValueError(
            f"Page {start} alone encodes to {encoded_size(sub)} bytes, over the "
            f"{max_encoded_bytes}-byte request budget. Raise "
            "ChunkerConfig.max_request_mb (if the serving platform allows) or "
            "re-scan/downsample the PDF."
        )
    mid = (start + end) // 2
    return _slice_to_fit(reader, start, mid, max_encoded_bytes) + _slice_to_fit(
        reader, mid + 1, end, max_encoded_bytes
    )


def iter_batches(
    pdf_bytes: bytes,
    max_pages_per_batch: int,
    overlap_pages: int,
    max_encoded_bytes: Optional[int] = None,
) -> List[PageBatch]:
    """Split a document into ``PageBatch`` objects ready to send to the model.

    When ``max_encoded_bytes`` is given, any planned batch whose encoded size
    exceeds it is split into smaller page ranges (losing the overlap between
    the split pieces, but staying under the platform request limit).
    """
    page_count = count_pages(pdf_bytes)
    ranges = plan_batches(page_count, max_pages_per_batch, overlap_pages)
    reader = PdfReader(io.BytesIO(pdf_bytes))
    batches: List[PageBatch] = []
    for start, end in ranges:
        if start == 1 and end == page_count:
            sub = pdf_bytes  # whole document; no need to re-encode
        else:
            sub = _slice(reader, start, end)
        if max_encoded_bytes is not None and encoded_size(sub) > max_encoded_bytes:
            batches.extend(_slice_to_fit(reader, start, end, max_encoded_bytes))
        else:
            batches.append(
                PageBatch(page_start=start, page_end=end, pdf_bytes=sub)
            )
    return batches


def document_block(pdf_bytes: bytes, citations: bool = False) -> Dict[str, Any]:
    """Build a native PDF ``document`` content block for the Messages API.

    Works for both digital-text and scanned-image PDFs -- Claude reads scanned
    pages via vision. Citations are OFF by default: the API rejects citations
    when structured outputs (``output_config.format``) are used, and both
    passes rely on structured outputs. Page provenance instead comes from the
    model's structured ``page_start``/``page_end`` fields.
    """
    data = base64.standard_b64encode(pdf_bytes).decode("ascii")
    block: Dict[str, Any] = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": data,
        },
    }
    if citations:
        block["citations"] = {"enabled": True}
    return block
