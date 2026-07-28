"""Orchestration: load -> Pass 1 -> Pass 2 -> write JSONL + profile.json."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .analyze import analyze_document, uniquify_titles
from .chunker import chunk_document
from .config import ChunkerConfig
from .fidelity import fidelity_report
from .llm import UsageTracker, make_client
from .models import Chunk, DocumentProfile
from .pdf_io import count_pages, read_pdf
from .tokenizer import get_token_counter

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    profile: DocumentProfile
    chunks: List[Chunk]
    usage: Optional[UsageTracker] = None
    fidelity: Optional[Dict[str, Any]] = None


def completed_section_prefix(
    sections: List[object], chunks: List[Chunk]
) -> int:
    """Longest prefix of ``sections`` that already has chunks on disk.

    The chunks file is written strictly in section order, so the first
    section with no chunks marks where an interrupted run stopped. A section
    that legitimately produced zero chunks re-runs on resume -- harmless,
    just a little repeated work.
    """
    titles_with_chunks = {c.section_title for c in chunks}
    done = 0
    for sec in sections:
        if sec.title not in titles_with_chunks:
            break
        done += 1
    return done


def _read_chunks(path: str) -> List[Chunk]:
    """Read a chunks JSONL file written by an interrupted run, safely.

    A run killed mid-write leaves a truncated final line, and -- because the
    OS flushes on buffer boundaries, not section boundaries -- possibly some
    complete lines of the section that was being written. Reading stops at
    the first unparseable line, and every trailing chunk belonging to the
    same section as the last good line is dropped too: that section cannot
    be proven complete, so it re-chunks on resume (cheap) rather than
    resuming with a silently missing tail (content loss).
    """
    chunks: List[Chunk] = []
    truncated = False
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(Chunk.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                logger.warning(
                    "%s line %d is not a valid chunk record (interrupted "
                    "write?); ignoring it and everything after",
                    path,
                    lineno,
                )
                truncated = True
                break
    if truncated and chunks:
        torn_title = chunks[-1].section_title
        dropped = 0
        while chunks and chunks[-1].section_title == torn_title:
            chunks.pop()
            dropped += 1
        logger.warning(
            "Dropping %d chunk(s) of section %r: the file was cut off "
            "mid-write, so that section may be incomplete and will re-chunk",
            dropped,
            torn_title,
        )
    return chunks


def run(
    pdf_path: str,
    config: Optional[ChunkerConfig] = None,
    out_path: Optional[str] = None,
    profile_path: Optional[str] = None,
    client: Optional[object] = None,
    resume: bool = False,
    fidelity: bool = True,
) -> ChunkResult:
    """Chunk one PDF end to end.

    Returns the profile + chunks, and optionally writes ``chunks.jsonl`` and
    ``profile.json``. ``client`` can be injected (tests pass a fake).

    With ``resume=True``, an existing ``profile_path`` skips Pass 1 and an
    existing ``out_path`` skips every section that already finished, so an
    interrupted run only re-pays for the unfinished tail.
    """
    config = config or ChunkerConfig()
    client = client or make_client()
    counter = get_token_counter(config.tokenizer_id)
    source_file = os.path.basename(pdf_path)
    usage = UsageTracker()

    pdf_bytes = read_pdf(pdf_path)

    profile = None
    if resume and profile_path and os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = DocumentProfile.from_dict(json.load(f))
        # A profile for a different PDF corrupts everything downstream (page
        # clamps, slicing, grounding), so mismatches are a hard error; a
        # differing filename alone may just be a rename, so it only warns.
        actual_pages = count_pages(pdf_bytes)
        if profile.page_count != actual_pages:
            raise ValueError(
                f"Resume: {profile_path} says {profile.page_count} pages but "
                f"{source_file} has {actual_pages} -- the profile does not "
                "match this PDF. Delete it or re-run without --resume."
            )
        if profile.source_file and profile.source_file != source_file:
            logger.warning(
                "Resume: profile was written for %r but chunking %r; "
                "continuing since the page counts match",
                profile.source_file,
                source_file,
            )
        # Profiles written before titles were uniquified can carry duplicate
        # titles, which corrupt the resume prefix; normalize and persist so
        # disk titles match the chunks this run writes.
        uniquify_titles(profile.sections)
        write_profile(profile, profile_path)
        logger.info(
            "Resume: loaded profile from %s (%d sections); skipping Pass 1",
            profile_path,
            len(profile.sections),
        )

    if profile is None:
        logger.info("Pass 1: analyzing %s", source_file)
        profile = analyze_document(
            client, config, pdf_bytes, source_file, usage=usage
        )
        logger.info("Pass 1: found %d sections", len(profile.sections))
        # Persist the profile before Pass 2 so a failure partway through the
        # (many-call) chunking pass never costs the completed analysis.
        if profile_path:
            write_profile(profile, profile_path)
            logger.info("Wrote profile to %s", profile_path)

    # On resume, keep chunks from every section that fully finished; the
    # first section without chunks (and everything after) re-runs.
    kept: List[Chunk] = []
    remaining = list(profile.sections)
    if resume and out_path and os.path.exists(out_path):
        existing = _read_chunks(out_path)
        done = completed_section_prefix(profile.sections, existing)
        done_titles = {s.title for s in profile.sections[:done]}
        kept = [c for c in existing if c.section_title in done_titles]
        for i, chunk in enumerate(kept):  # renumber the kept prefix
            chunk.chunk_index = i
        remaining = list(profile.sections[done:])
        logger.info(
            "Resume: %d/%d sections already chunked (%d chunks kept)",
            done,
            len(profile.sections),
            len(kept),
        )

    logger.info("Pass 2: chunking %d sections", len(remaining))
    out_file = None
    on_section = None
    if out_path:
        # Stream chunks to disk as each section finishes; on failure the file
        # holds every completed section instead of nothing.
        out_file = open(out_path, "w", encoding="utf-8")
        for chunk in kept:
            out_file.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        out_file.flush()

        def on_section(section_chunks: List[Chunk]) -> None:
            for chunk in section_chunks:
                out_file.write(
                    json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n"
                )
            out_file.flush()

    try:
        new_chunks = chunk_document(
            client, config, pdf_bytes, profile, counter,
            on_section=on_section, usage=usage,
            sections=remaining, start_index=len(kept),
        )
    finally:
        if out_file is not None:
            out_file.close()
    chunks = kept + new_chunks
    logger.info("Pass 2: produced %d chunks (%d new)", len(chunks), len(new_chunks))

    # Report-only fidelity check against the PDF text layer (no API calls);
    # persisted alongside the profile so the scores travel with the map.
    report = None
    if fidelity:
        report = fidelity_report(pdf_bytes, profile, chunks)
        if report["status"] == "ok":
            doc = report["document"]
            logger.info(
                "Fidelity: coverage %.2f, novelty %.2f (see profile.json)",
                doc["coverage"],
                doc["novelty"],
            )
        else:
            logger.info("Fidelity: %s", report["reason"])
    if profile_path:
        # Rewritten even when fidelity is skipped, so a stale block from an
        # earlier run never describes chunks that no longer exist.
        payload = profile.to_dict()
        if report is not None:
            payload["fidelity"] = report
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Usage: %s", usage.summary())

    return ChunkResult(
        profile=profile, chunks=chunks, usage=usage, fidelity=report
    )


def write_profile(profile: DocumentProfile, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)


def write_chunks(chunks: List[Chunk], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
