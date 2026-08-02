"""Command-line entrypoint: chunk a PDF into JSONL + a profile.json map."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv

from .config import (
    DEFAULT_MAX_PAGES_PER_BATCH,
    DEFAULT_MAX_REQUEST_MB,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TARGET_TOKENS,
    GTE_TOKENIZER_ID,
    ChunkerConfig,
)
from .pipeline import run


def _add_pass1_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared by `chunk` and `analyze` (Pass 1 + request shaping)."""
    parser.add_argument("pdf", help="Path to the input PDF.")
    parser.add_argument(
        "--profile",
        default="profile.json",
        help="Where to write the Pass 1 global map.",
    )
    parser.add_argument("--pass1-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-pages-per-batch",
        type=int,
        default=DEFAULT_MAX_PAGES_PER_BATCH,
        help="Pages per Pass 1 batch for long documents.",
    )
    parser.add_argument(
        "--pass1-concurrency",
        type=int,
        default=4,
        help="Parallel Pass 1 batch requests (1 = sequential).",
    )
    parser.add_argument(
        "--max-request-mb",
        type=float,
        default=DEFAULT_MAX_REQUEST_MB,
        help="Per-request payload budget in MB (base64-encoded PDF). Default "
        "fits the Anthropic API / Azure AI Foundry (32 MB); use ~3 when "
        "calls route through Databricks model serving (~4 MB limit).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intelligent-chunker",
        description="Two-pass, model-native intelligent chunker for PDFs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    chunk = sub.add_parser("chunk", help="Chunk a PDF into JSONL.")
    _add_pass1_args(chunk)
    chunk.add_argument(
        "--out", default="chunks.jsonl", help="Output JSONL path."
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
    chunk.add_argument("--pass2-model", default=DEFAULT_MODEL)
    chunk.add_argument(
        "--pass2-concurrency",
        type=int,
        default=4,
        help="Parallel Pass 2 section requests (1 = sequential).",
    )
    chunk.add_argument(
        "--tokenizer",
        default=os.environ.get("CHUNKER_TOKENIZER", GTE_TOKENIZER_ID),
        help="Hugging Face model id, or a path to a local tokenizer.json for "
        "offline use (also via CHUNKER_TOKENIZER).",
    )
    chunk.add_argument(
        "--no-fidelity",
        action="store_true",
        help="Skip the report-only fidelity check against the PDF text layer.",
    )
    chunk.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing profile/chunks file from an interrupted run: "
        "skip Pass 1 if the profile exists and re-run only unfinished "
        "sections.",
    )

    analyze = sub.add_parser(
        "analyze",
        help="Run Pass 1 only: write the document map (profile.json) so it "
        "can be inspected before paying for Pass 2.",
    )
    _add_pass1_args(analyze)

    fidelity = sub.add_parser(
        "fidelity",
        help="Recompute the report-only fidelity check from existing outputs "
        "(no API calls) and refresh the profile's fidelity block.",
    )
    fidelity.add_argument("pdf", help="Path to the source PDF.")
    fidelity.add_argument(
        "--chunks", default="chunks.jsonl", help="Chunks JSONL path."
    )
    fidelity.add_argument(
        "--profile", default="profile.json", help="Profile JSON path."
    )
    fidelity.add_argument(
        "--no-write",
        action="store_true",
        help="Print the summary without updating the profile file.",
    )
    fidelity.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging."
    )

    view = sub.add_parser(
        "view",
        help="Build a self-contained HTML viewer/curation tool: review "
        "chunks, edit text, include/exclude, download a curated JSONL.",
    )
    view.add_argument("--chunks", default="chunks.jsonl", help="Chunks JSONL path.")
    view.add_argument("--profile", default="profile.json", help="Profile JSON path.")
    view.add_argument("--out", default="viewer.html", help="Output HTML path.")
    view.add_argument(
        "--open", action="store_true", help="Open the viewer in a browser."
    )

    serve = sub.add_parser(
        "serve",
        help="Run the web app locally: upload an SPD, process with a "
        "progress meter, review/curate the results.",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--data-dir",
        default=os.environ.get("CHUNKER_APP_DATA", "app_data"),
        help="Root for uploads and job outputs (also via CHUNKER_APP_DATA). "
        "On Databricks this must be a UC Volume path.",
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
            pass2_concurrency=args.pass2_concurrency,
            max_request_mb=args.max_request_mb,
            tokenizer_id=args.tokenizer,
        )
        try:
            result = run(
                args.pdf,
                config=config,
                out_path=args.out,
                profile_path=args.profile,
                resume=args.resume,
                fidelity=not args.no_fidelity,
            )
        except FileNotFoundError as exc:
            print(f"Input not found: {exc.filename or exc}", file=sys.stderr)
            return 1
        print(
            f"Wrote {len(result.chunks)} chunks to {args.out} "
            f"and the document map to {args.profile} "
            f"({len(result.profile.sections)} sections)."
        )
        if result.usage is not None and result.usage.calls:
            print(f"Usage: {result.usage.summary()}")
        return 0

    if args.command == "analyze":
        from .analyze import analyze_document
        from .llm import UsageTracker, make_client
        from .pdf_io import read_pdf
        from .pipeline import write_profile

        config = ChunkerConfig(
            pass1_model=args.pass1_model,
            max_pages_per_batch=args.max_pages_per_batch,
            pass1_concurrency=args.pass1_concurrency,
            max_request_mb=args.max_request_mb,
        )
        try:
            pdf_bytes = read_pdf(args.pdf)
        except FileNotFoundError as exc:
            print(f"Input not found: {exc.filename or exc}", file=sys.stderr)
            return 1
        usage = UsageTracker()
        profile = analyze_document(
            make_client(),
            config,
            pdf_bytes,
            os.path.basename(args.pdf),
            usage=usage,
        )
        write_profile(profile, args.profile)
        print(
            f"Wrote the document map to {args.profile}: "
            f"{len(profile.sections)} sections, "
            f"{len(profile.glossary)} glossary terms, "
            f"{profile.page_count} pages."
        )
        print(f"Usage: {usage.summary()}")
        return 0

    if args.command == "fidelity":
        import json

        from .fidelity import fidelity_report
        from .models import Chunk, DocumentProfile
        from .pdf_io import read_pdf

        try:
            pdf_bytes = read_pdf(args.pdf)
            with open(args.profile, "r", encoding="utf-8") as f:
                profile_dict = json.load(f)
            with open(args.chunks, "r", encoding="utf-8") as f:
                chunks = [
                    Chunk.from_dict(json.loads(line))
                    for line in f
                    if line.strip()
                ]
        except FileNotFoundError as exc:
            print(
                f"Input not found: {exc.filename or exc}. "
                "Run `intelligent-chunker chunk` first.",
                file=sys.stderr,
            )
            return 1
        profile = DocumentProfile.from_dict(profile_dict)
        report = fidelity_report(pdf_bytes, profile, chunks)

        if report["status"] == "ok":
            doc = report["document"]
            print(
                f"Document: coverage {doc['coverage']:.4f}, "
                f"novelty {doc['novelty']:.4f} "
                f"({doc['reference_words']} text-layer words, "
                f"{doc['chunk_words']} chunk words)."
            )
            flagged = report.get("chunks", [])
            if flagged:
                print(f"Flagged chunks: {len(flagged)}")
                for f_ in flagged:
                    hints = f_.get("novel_line_hints") or []
                    line = (
                        f"; e.g. {f_['novel_lines'][0]!r}"
                        + (f" [{hints[0]}]" if hints else "")
                        if f_["novel_lines"]
                        else ""
                    )
                    print(
                        f"  chunk {f_['chunk_index']} "
                        f"(pages {f_['page_start']}-{f_['page_end']}, "
                        f"{f_['section']!r}): novelty {f_['novelty']:.2f}"
                        f"{line}"
                    )
            else:
                print("Flagged chunks: none.")
        else:
            print(f"Fidelity skipped: {report['reason']}")

        if not args.no_write:
            with open(args.profile, "w", encoding="utf-8") as f:
                json.dump(
                    {**profile_dict, "fidelity": report},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"Updated the fidelity block in {args.profile}.")
        return 0

    if args.command == "serve":
        try:
            import uvicorn

            from .webapp.app import create_app
        except ImportError:
            print(
                "The web app needs the 'app' extra: pip install -e '.[app]'",
                file=sys.stderr,
            )
            return 1
        uvicorn.run(
            create_app(args.data_dir), host=args.host, port=args.port
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
