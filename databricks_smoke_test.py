"""Smoke-test a Databricks Anthropic Messages endpoint for this pipeline's needs.

Run from inside the tenant (or paste into a Databricks notebook). Requires the
``anthropic`` and ``pypdf`` packages. Configure via env or flags:

    export DATABRICKS_HOST=https://adb-xxxx.azuredatabricks.net
    export DATABRICKS_TOKEN=<pat>
    python databricks_smoke_test.py [--model databricks-claude-...] [--pdf some.pdf]

Each check exercises one feature the chunker depends on, in order of
increasing exoticness, so the first FAIL tells you where Databricks support
stops:

    1. plain text call        -- routing + auth work at all
    2. streaming              -- llm.py streams every request
    3. structured output      -- output_config json_schema (both passes)
    4. PDF document block     -- native PDF input (the whole pipeline)
    5. prompt caching         -- cache_control on a prefix padded above the
                                 minimum cacheable length (Pass 2 economics)

Total cost is a fraction of a cent. Checks that depend on a failed check are
skipped rather than reported as their own failures.

Private-tenant TLS: if the endpoint presents a certificate from a private CA
(SSL: CERTIFICATE_VERIFY_FAILED), point verification at the corporate bundle
with --ca-bundle /path/to/corp-ca.pem (or export SSL_CERT_FILE=...), or
``pip install truststore`` to trust the OS certificate store, which corporate
machines usually have provisioned. --insecure disables verification entirely
and exists only to isolate TLS from other failures; never use it for real runs.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

import anthropic

DESCRIBE = {"type": "text", "text": "Describe this document in one sentence."}

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


def doc_block(pdf_bytes: bytes) -> dict:
    import base64

    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.standard_b64encode(pdf_bytes).decode("ascii"),
        },
    }


def build_http_client(ca_bundle: str | None, insecure: bool):
    """TLS setup for private tenants. None means use the SDK's default client."""
    import httpx

    if insecure:
        print("[warn] TLS verification DISABLED -- diagnostic use only")
        return httpx.Client(verify=False)
    if ca_bundle:
        print(f"[info] TLS: verifying against CA bundle {ca_bundle}")
        return httpx.Client(verify=ca_bundle)
    try:
        import ssl

        import truststore
    except ImportError:
        return None
    print("[info] TLS: using the OS trust store (truststore)")
    return httpx.Client(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="databricks-claude-haiku-4-5")
    parser.add_argument("--host", default=os.environ.get("DATABRICKS_HOST"))
    parser.add_argument("--pdf", help="Optional real PDF; its first page is used.")
    parser.add_argument(
        "--ca-bundle",
        default=os.environ.get("CHUNKER_CA_BUNDLE"),
        help="PEM bundle for a private CA (also via CHUNKER_CA_BUNDLE).",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification. Diagnostic use only.",
    )
    args = parser.parse_args()

    token = os.environ.get("DATABRICKS_TOKEN")
    if not args.host or not token:
        sys.exit("Set DATABRICKS_HOST (or --host) and DATABRICKS_TOKEN.")

    http_client = build_http_client(args.ca_bundle, args.insecure)
    client = anthropic.Anthropic(
        api_key="unused",
        base_url=args.host.rstrip("/") + "/serving-endpoints/anthropic",
        default_headers={"Authorization": f"Bearer {token}"},
        **({"http_client": http_client} if http_client else {}),
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
        # Roomy max_tokens: a truncated answer is an unterminated JSON string,
        # which would fail the check for the wrong reason.
        with client.messages.stream(
            model=args.model,
            max_tokens=256,
            messages=[{"role": "user", "content": 'Answer: what color is the sky?'}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        ) as s:
            msg = s.get_final_message()
        # The first text block holds the JSON (other block types may precede).
        text = next(
            (b.text for b in msg.content if getattr(b, "type", None) == "text"),
            None,
        )
        if text is None:
            raise RuntimeError("no text block in structured-output response")
        json.loads(text)  # must be conforming JSON

    def pdf_block():
        client.messages.create(
            model=args.model,
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": [doc_block(pdf), DESCRIBE],
                }
            ],
        )

    def caching():
        # Pad the cached prefix well past every model's minimum cacheable
        # length; below the minimum, cache_control is silently ignored and
        # the check would fail for the wrong reason.
        pad = {
            "type": "text",
            "text": "Neutral cache-padding sentence for the smoke test. " * 600,
            "cache_control": {"type": "ephemeral"},
        }
        stats = []
        for _ in range(2):  # call 1 writes the cache, call 2 should read it
            resp = client.messages.create(
                model=args.model,
                max_tokens=64,
                messages=[
                    {"role": "user", "content": [doc_block(pdf), pad, DESCRIBE]}
                ],
            )
            u = getattr(resp, "usage", None)
            stats.append(
                (
                    getattr(u, "cache_creation_input_tokens", 0) or 0,
                    getattr(u, "cache_read_input_tokens", 0) or 0,
                )
            )
        print(f"       cache (created, read): call1={stats[0]} call2={stats[1]}")
        if stats[1][1] > 0:
            return
        if stats[0][0] <= 0:
            raise RuntimeError(
                f"cache never created {stats}: the endpoint likely strips "
                "cache_control / caching unsupported"
            )
        raise RuntimeError(
            f"cache written on call 1 but not read on call 2 {stats}: "
            "cache not shared across requests (routing/TTL?)"
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
