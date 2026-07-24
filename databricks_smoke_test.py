"""Smoke-test a Databricks Anthropic Messages endpoint for this pipeline's needs.

Run from inside the tenant (or paste into a Databricks notebook). Requires the
``anthropic`` and ``pypdf`` packages. Configure via env or flags:

    export DATABRICKS_HOST=https://adb-xxxx.azuredatabricks.net
    export DATABRICKS_TOKEN=<pat>
    python databricks_smoke_test.py [--model databricks-claude-haiku-4-5] [--pdf some.pdf]

Each check exercises one feature the chunker depends on, in order of
increasing exoticness, so the first FAIL tells you where Databricks support
stops:

    1. plain text call        -- routing + auth work at all
    2. streaming              -- llm.py streams every request
    3. structured output      -- output_config json_schema (both passes)
    4. PDF document block     -- native PDF input (the whole pipeline)
    5. prompt caching         -- cache_control on the document block (Pass 2)

Total cost is a fraction of a cent. Checks that depend on a failed check are
skipped rather than reported as their own failures.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

import anthropic

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def tiny_pdf(path: str | None) -> bytes:
    """First page of ``path``, or a generated blank page when no PDF is given."""
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    if path:
        writer.add_page(PdfReader(path).pages[0])
    else:
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def doc_block(pdf_bytes: bytes, cached: bool = False) -> dict:
    import base64

    block = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.standard_b64encode(pdf_bytes).decode("ascii"),
        },
    }
    if cached:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="databricks-claude-haiku-4-5")
    parser.add_argument("--host", default=os.environ.get("DATABRICKS_HOST"))
    parser.add_argument("--pdf", help="Optional real PDF; its first page is used.")
    args = parser.parse_args()

    token = os.environ.get("DATABRICKS_TOKEN")
    if not args.host or not token:
        sys.exit("Set DATABRICKS_HOST (or --host) and DATABRICKS_TOKEN.")

    client = anthropic.Anthropic(
        api_key="unused",
        base_url=args.host.rstrip("/") + "/serving-endpoints/anthropic",
        default_headers={"Authorization": f"Bearer {token}"},
    )
    pdf = tiny_pdf(args.pdf)
    results: dict[str, str] = {}

    def run(name: str, fn) -> bool:
        try:
            fn()
        except Exception as exc:  # report the API's own error verbatim
            results[name] = f"FAIL  {type(exc).__name__}: {exc}"
            print(f"[FAIL] {name}\n       {type(exc).__name__}: {exc}")
            return False
        results[name] = "PASS"
        print(f"[ok]   {name}")
        return True

    def text_call():
        client.messages.create(
            model=args.model,
            max_tokens=32,
            messages=[{"role": "user", "content": "Reply with the word: pong"}],
        )

    def streaming():
        with client.messages.stream(
            model=args.model,
            max_tokens=32,
            messages=[{"role": "user", "content": "Reply with the word: pong"}],
        ) as s:
            s.get_final_message()

    def structured():
        with client.messages.stream(
            model=args.model,
            max_tokens=64,
            messages=[{"role": "user", "content": 'Answer: what color is the sky?'}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        ) as s:
            msg = s.get_final_message()
        json.loads(msg.content[0].text)  # must be conforming JSON

    def pdf_block():
        client.messages.create(
            model=args.model,
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": [
                        doc_block(pdf),
                        {"type": "text", "text": "Describe this document in one sentence."},
                    ],
                }
            ],
        )

    def caching():
        for _ in range(2):  # second call should be a cache read
            resp = client.messages.create(
                model=args.model,
                max_tokens=64,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            doc_block(pdf, cached=True),
                            {"type": "text", "text": "Describe this document in one sentence."},
                        ],
                    }
                ],
            )
        usage = getattr(resp, "usage", None)
        read = getattr(usage, "cache_read_input_tokens", 0) or 0
        if read <= 0:
            raise RuntimeError(
                "cache_control accepted but second call reported no "
                f"cache_read_input_tokens (usage={usage!r})"
            )

    if not run("1. plain text call", text_call):
        print("\nRouting/auth failed; nothing else can be tested.")
        return 1
    run("2. streaming", streaming)
    run("3. structured output (json_schema)", structured)
    if run("4. PDF document block", pdf_block):
        run("5. prompt caching on PDF block", caching)
    else:
        print("[skip] 5. prompt caching (PDF block not accepted)")
        results["5. prompt caching on PDF block"] = "SKIP"

    print("\nSummary:")
    for name, outcome in results.items():
        print(f"  {name}: {outcome.splitlines()[0]}")
    print(
        "\nVerdict: the chunker needs ALL five to run unmodified against "
        "Databricks.\n  1-3 only: possible with a text-extraction fallback "
        "(feature work).\n  1-4: works; without 5 every Pass 2 call pays "
        "full input price."
    )
    return 0 if all(v == "PASS" for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
