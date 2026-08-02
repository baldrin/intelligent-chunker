# Running the chunker web app as a Databricks App

The same FastAPI app that runs locally (`intelligent-chunker serve`) deploys
as a Databricks App. Inside the workspace it gets two things for free: calls
to `/serving-endpoints/anthropic` are internal (no corporate-proxy TLS
problem, so no `CHUNKER_CA_BUNDLE` needed), and auth can come from the app's
identity instead of a personal token.

## Non-negotiable: storage goes to a UC Volume

The app instance's local disk is tiny; filling it crashes and reboots the
app. Everything the app writes -- uploads, chunks, profiles, curated files --
goes under `CHUNKER_APP_DATA`, which **must** point at a Unity Catalog Volume
path. Uploads are streamed as raw request bodies directly into that root (the
app deliberately avoids multipart parsing, which spools to system temp).

## Setup

1. **Create a UC Volume** (or pick an existing one) and choose a directory,
   e.g. `/Volumes/<catalog>/<schema>/<volume>/spd_chunker`.
2. **Put the GTE tokenizer there**: download
   `https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5/resolve/main/tokenizer.json`
   once (browser works) and upload it to the volume. Without it the app falls
   back to heuristic token sizing and the UI warns loudly on every job.
3. **Sync this repo as the app source** and copy `databricks_app/app.yaml`
   and `databricks_app/requirements.txt` to the repo root in the workspace
   copy. Fill in the placeholders in `app.yaml` (volume path, workspace
   host).
4. **Create the app** pointing at that source, with a secret resource named
   `anthropic-token` holding the Bearer token for the serving endpoint (PAT
   for bring-up; the app's service principal token once permissions are set).
5. **Grant the app's service principal**:
   - `CAN QUERY` on the Claude serving endpoints (pay-per-token).
   - `READ VOLUME` + `WRITE VOLUME` on the volume.

## In-tenant verification checklist (first deploy)

Work through these in order -- each is a small, isolated check, and the app's
own error reporting (job status shows pipeline errors verbatim) makes
failures diagnosable:

1. **Volume file I/O**: open the app URL, upload a tiny PDF with the Haiku
   model. If job creation fails immediately, plain `open()` on the
   `/Volumes/...` path isn't available in the Apps runtime -- report back,
   and a storage shim via the Databricks SDK Files API goes behind the same
   data-root abstraction (`webapp/jobs.py` keeps all paths in one place).
2. **Serving auth**: a 403 on the first processing call means the secret /
   service-principal permission on the serving endpoint isn't right. The
   error text in the job status will say exactly what the endpoint returned.
3. **Upload size**: try the largest real SPD (the ~55 MB scanned one is the
   stress case). If the Apps proxy rejects it, recompress first
   (ghostscript ~150 DPI grayscale) -- already the plan for that document.
4. **Restart behavior**: restart the app and confirm finished jobs still
   list (state is journaled to `job.json` in the volume; only in-flight
   progress lives in memory).

## Local development

```bash
pip install -e '.[dev,app]'
intelligent-chunker serve --port 8000 --data-dir app_data
```

The local app reads the same `.env` as the CLI (`ANTHROPIC_BASE_URL`,
`ANTHROPIC_AUTH_TOKEN`, `CHUNKER_CA_BUNDLE`, `CHUNKER_TOKENIZER`).
