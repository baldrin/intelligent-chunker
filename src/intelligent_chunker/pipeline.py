"""Orchestration: load -> Pass 1 -> Pass 2 -> write JSONL + profile.json."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from .analyze import analyze_document
from .chunker import chunk_document
from .config import ChunkerConfig
from .llm import UsageTracker, make_client
from .models import Chunk, DocumentProfile
from .pdf_io import read_pdf
from .tokenizer import get_token_counter

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    profile: DocumentProfile
    chunks: List[Chunk]
    usage: Optional[UsageTracker] = None


def run(
    pdf_path: str,
    config: Optional[ChunkerConfig] = None,
    out_path: Optional[str] = None,
    profile_path: Optional[str] = None,
    client: Optional[object] = None,
) -> ChunkResult:
    """Chunk one PDF end to end.

    Returns the profile + chunks, and optionally writes ``chunks.jsonl`` and
    ``profile.json``. ``client`` can be injected (tests pass a fake).
    """
    config = config or ChunkerConfig()
    client = client or make_client()
    counter = get_token_counter(config.tokenizer_id)
    source_file = os.path.basename(pdf_path)
    usage = UsageTracker()

    pdf_bytes = read_pdf(pdf_path)

    logger.info("Pass 1: analyzing %s", source_file)
    profile = analyze_document(client, config, pdf_bytes, source_file, usage=usage)
    logger.info("Pass 1: found %d sections", len(profile.sections))

    # Persist the profile before Pass 2 so a failure partway through the
    # (many-call) chunking pass never costs the completed analysis.
    if profile_path:
        write_profile(profile, profile_path)
        logger.info("Wrote profile to %s", profile_path)

    logger.info("Pass 2: chunking %d sections", len(profile.sections))
    out_file = None
    on_section = None
    if out_path:
        # Stream chunks to disk as each section finishes; on failure the file
        # holds every completed section instead of nothing.
        out_file = open(out_path, "w", encoding="utf-8")

        def on_section(section_chunks: List[Chunk]) -> None:
            for chunk in section_chunks:
                out_file.write(
                    json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n"
                )
            out_file.flush()

    try:
        chunks = chunk_document(
            client, config, pdf_bytes, profile, counter,
            on_section=on_section, usage=usage,
        )
    finally:
        if out_file is not None:
            out_file.close()
    logger.info("Pass 2: produced %d chunks", len(chunks))
    logger.info("Usage: %s", usage.summary())

    return ChunkResult(profile=profile, chunks=chunks, usage=usage)


def write_profile(profile: DocumentProfile, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)


def write_chunks(chunks: List[Chunk], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
