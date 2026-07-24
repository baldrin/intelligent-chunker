# intelligent-chunker

Two-pass, **model-native** intelligent chunker for unstructured PDFs — built for
Summary Plan Description (SPD) benefits documents, but general-purpose.

No PDF text-extraction library: **Claude Haiku reads the PDF directly** (digital
text *and* scanned pages, via vision). Chunking happens in two passes so chunks
keep full-document context instead of being cut blindly:

1. **Pass 1 — Global analysis** (`analyze.py`): read the whole document and
   build a *map* — section outline with page ranges, document-wide metadata,
   glossary, cross-references. A deterministic **coverage guard** then fills
   any pages the outline missed with synthetic "Unmapped pages" sections, so
   Pass 2 never silently skips content.
2. **Pass 2 — Context-aware chunking** (`chunker.py`): re-read each section
   *with the map as context* (sections fan out across threads) and emit
   coherent chunks — each carrying its own physical page range — that never
   break mid-word/mid-sentence. A deterministic **token guard** then enforces
   the embedder's hard limit so nothing is silently truncated.

Output is `chunks.jsonl` (one embedding-ready chunk per line, with metadata)
plus `profile.json` (the global map, including a **fidelity report** — see
below). Each run ends with a token-usage line and an estimated cost.

> **Scope:** the chunker only. Embedding (against an existing **GTE-large v1.5**
> model over HTTP) is interface-only/stubbed in `embed.py` and wired in a later
> phase.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[tokenizer,dev]"
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

The `tokenizer` extra installs the GTE tokenizer for exact chunk sizing; without
it the chunker falls back to a heuristic counter and warns.

## Use

```bash
intelligent-chunker chunk path/to/spd.pdf --out chunks.jsonl --profile profile.json
# options: --max-tokens 1024 --target-tokens 512 --max-pages-per-batch 50
#          --pass1-concurrency 4 --pass2-concurrency 4 --max-request-mb 25
#          --pass1-model claude-haiku-4-5 --pass2-model claude-haiku-4-5
#          --resume --no-fidelity -v

# Interrupted mid-run? Re-run with --resume: an existing profile.json skips
# Pass 1, and only sections missing from chunks.jsonl are re-chunked.

# Or run Pass 1 alone and inspect the map before paying for Pass 2:
intelligent-chunker analyze path/to/spd.pdf --profile profile.json

# Browse the output in a self-contained HTML viewer (no server needed):
intelligent-chunker view --chunks chunks.jsonl --profile profile.json --open

# Export Databricks-ready Parquet tables (needs the `databricks` extra: pyarrow):
intelligent-chunker export --chunks chunks.jsonl --profile profile.json \
    --out-dir databricks_export   # then load with databricks/build_vector_index.py
```

> **Request-size limits:** each API call carries a base64-encoded PDF slice.
> `--max-request-mb` / `ChunkerConfig.max_request_mb` (default 25) caps that
> payload and splits batches that exceed it. The default fits the Anthropic
> API and Azure AI Foundry (32 MB/request); drop it to ~3 if calls route
> through Databricks model serving (~4 MB/request).

> **Databricks routing:** the pipeline can call Claude through a Databricks
> workspace's native Anthropic endpoint instead of the direct API. Set
> `ANTHROPIC_BASE_URL=https://<workspace-host>/serving-endpoints/anthropic`
> and `ANTHROPIC_AUTH_TOKEN=<token>` (see `.env.example`), use
> `databricks-claude-*` model names via `--pass1-model`/`--pass2-model`, and
> pass `--max-request-mb 3`. For endpoints behind a private/corporate CA, set
> `CHUNKER_CA_BUNDLE=/path/to/ca.pem` or `pip install truststore`. Verify an
> endpoint end-to-end with `databricks_smoke_test.py` (run inside the
> tenant); all five checks passing means the pipeline runs unmodified. Cost
> estimates are omitted for `databricks-*` model names (Databricks bills in
> DBUs), so run summaries report tokens only.

### Fidelity report

After chunking, the pipeline compares each section's chunks against the PDF's
embedded text layer (word-multiset overlap, no API calls) and writes the
scores into `profile.json` under `fidelity`: **coverage** (fraction of
text-layer words present in the chunks — low means content may have been
missed) and **novelty** (fraction of chunk words absent from the text layer —
high means content may have been invented). The **document-level score is the
number to trust** — it compares all pages against all chunks. Per-section
scores are biased low whenever sections share a page (the reference then
includes neighbors' text); such sections are marked `shared_pages: true` and
excluded from warnings. Scores outside the advisory thresholds (coverage
< 0.85, novelty > 0.15) log warnings. The comparison is deliberately rough —
repeated headers/footers and hyphenation add noise, and scanned PDFs (no text
layer) skip the report — so treat scores as signals to inspect in the viewer,
not hard pass/fail. `--no-fidelity` skips it.

Or from Python:

```python
from intelligent_chunker import ChunkerConfig
from intelligent_chunker.pipeline import run

result = run("spd.pdf", ChunkerConfig(), out_path="chunks.jsonl", profile_path="profile.json")
print(len(result.chunks), "chunks across", len(result.profile.sections), "sections")
```

## Tests

```bash
pytest
```

Unit tests cover the deterministic pieces (page batching, profile reconciliation,
the token guard) with the model API mocked. An end-to-end run needs a real
`ANTHROPIC_API_KEY`.
