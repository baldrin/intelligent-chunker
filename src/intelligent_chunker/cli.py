"""Command-line entrypoint: chunk a PDF into JSONL + a profile.json map."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from dotenv import load_dotenv

from .config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TARGET_TOKENS,
    ChunkerConfig,
)
from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intelligent-chunker",
        description="Two-pass, model-native intelligent chunker for PDFs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    chunk = sub.add_parser("chunk", help="Chunk a PDF into JSONL.")
    chunk.add_argument("pdf", help="Path to the input PDF.")
    chunk.add_argument(
        "--out", default="chunks.jsonl", help="Output JSONL path."
    )
    chunk.add_argument(
        "--profile",
        default="profile.json",
        help="Where to write the Pass 1 global map.",
    )
    chunk.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Hard per-chunk token ceiling (enforced against the embedder).",
    )
    chunk.add_argument(
        "--target-tokens",
        type=int,
        default=DEFAULT_TARGET_TOKENS,
        help="Approximate per-chunk token target the model aims for.",
    )
    chunk.add_argument("--pass1-model", default=DEFAULT_MODEL)
    chunk.add_argument("--pass2-model", default=DEFAULT_MODEL)
    chunk.add_argument(
        "--max-pages-per-batch",
        type=int,
        default=50,
        help="Pages per Pass 1 batch for long documents.",
    )
    chunk.add_argument(
        "--pass1-concurrency",
        type=int,
        default=4,
        help="Parallel Pass 1 batch requests (1 = sequential).",
    )
    chunk.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging."
    )

    view = sub.add_parser(
        "view", help="Build a self-contained HTML viewer for the output."
    )
    view.add_argument("--chunks", default="chunks.jsonl", help="Chunks JSONL path.")
    view.add_argument("--profile", default="profile.json", help="Profile JSON path.")
    view.add_argument("--out", default="viewer.html", help="Output HTML path.")
    view.add_argument(
        "--open", action="store_true", help="Open the viewer in a browser."
    )

    export = sub.add_parser(
        "export",
        help="Export Databricks-ready Parquet tables (chunks/documents/glossary).",
    )
    export.add_argument("--chunks", default="chunks.jsonl", help="Chunks JSONL path.")
    export.add_argument(
        "--profile", default="profile.json", help="Profile JSON path."
    )
    export.add_argument(
        "--out-dir",
        default="databricks_export",
        help="Directory to write the Parquet tables into.",
    )
    export.add_argument(
        "--no-glossary",
        action="store_true",
        help="Skip the glossary table.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "chunk":
        config = ChunkerConfig(
            pass1_model=args.pass1_model,
            pass2_model=args.pass2_model,
            max_tokens=args.max_tokens,
            target_tokens=args.target_tokens,
            max_pages_per_batch=args.max_pages_per_batch,
            pass1_concurrency=args.pass1_concurrency,
        )
        try:
            result = run(
                args.pdf,
                config=config,
                out_path=args.out,
                profile_path=args.profile,
            )
        except FileNotFoundError as exc:
            print(f"Input not found: {exc.filename or exc}", file=sys.stderr)
            return 1
        print(
            f"Wrote {len(result.chunks)} chunks to {args.out} "
            f"and the document map to {args.profile} "
            f"({len(result.profile.sections)} sections)."
        )
        return 0

    if args.command == "view":
        from .viewer import build_html, load_chunks, load_profile

        try:
            html = build_html(load_profile(args.profile), load_chunks(args.chunks))
        except FileNotFoundError as exc:
            print(
                f"Input not found: {exc.filename or exc}. "
                "Run `intelligent-chunker chunk` first.",
                file=sys.stderr,
            )
            return 1
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Wrote viewer to {args.out}")
        if args.open:
            import webbrowser
            from pathlib import Path

            webbrowser.open(Path(args.out).resolve().as_uri())
        return 0

    if args.command == "export":
        # Probe pyarrow explicitly: a bare `except ImportError` around the
        # whole export would misattribute any ImportError raised inside it.
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            print(
                "Parquet export needs pyarrow. Install it with:\n"
                '  pip install -e ".[databricks]"',
                file=sys.stderr,
            )
            return 1

        from .export_databricks import export
        from .viewer import load_chunks, load_profile

        try:
            result = export(
                load_profile(args.profile),
                load_chunks(args.chunks),
                out_dir=args.out_dir,
                include_glossary=not args.no_glossary,
            )
        except FileNotFoundError as exc:
            print(
                f"Input not found: {exc.filename or exc}. "
                "Run `intelligent-chunker chunk` first.",
                file=sys.stderr,
            )
            return 1
        print(
            f"Exported to {args.out_dir}/:\n"
            f"  chunks.parquet     {result.chunk_rows} rows"
            f" ({result.duplicates_dropped} duplicates dropped)\n"
            f"  documents.parquet  1 row\n"
            + (
                f"  glossary.parquet   {result.glossary_rows} rows"
                if not args.no_glossary
                else "  (glossary skipped)"
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
