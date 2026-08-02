"""FastAPI app: upload an SPD, process it, review/curate the results.

A thin shell over ``pipeline.run`` -- see ``jobs.JobManager`` for the job
model (one at a time, per-job directory under the data root). The results
page is the existing self-contained curation viewer, served with a save-back
URL so curated JSONL lands next to the job's other files.

The upload is the raw request body (``application/pdf``), NOT multipart:
multipart parsers spool large bodies to the system temp directory, and on
Databricks the instance disk is tiny -- streaming the body straight into the
job directory keeps every byte on the data root.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..config import GTE_TOKENIZER_ID
from ..fidelity import fidelity_report
from ..models import Chunk, DocumentProfile
from ..viewer import build_html, load_chunks, load_profile
from .jobs import JobManager
from .templates import INDEX_HTML

_UPLOAD_CHUNK = 1 << 20  # 1 MiB


def model_choices() -> List[Dict[str, str]]:
    """Model picker options, derived from how the environment routes calls.

    Production quality is the default (Sonnet); Haiku is the explicit
    cheap-test opt-in. Databricks-routed environments get databricks-* names.
    """
    if os.environ.get("ANTHROPIC_BASE_URL"):
        return [
            {"id": "databricks-claude-sonnet-4-6", "label": "Sonnet 4.6 — production"},
            {"id": "databricks-claude-haiku-4-5", "label": "Haiku 4.5 — test (cheap)"},
        ]
    return [
        {"id": "claude-sonnet-4-6", "label": "Sonnet 4.6 — production"},
        {"id": "claude-haiku-4-5", "label": "Haiku 4.5 — test (cheap)"},
    ]


def default_max_request_mb() -> float:
    """3 MB when routed through Databricks serving (~4 MB cap), else 25."""
    return 3.0 if os.environ.get("ANTHROPIC_BASE_URL") else 25.0


def create_app(
    data_root: Optional[str] = None,
    client_factory: Optional[Callable[[], Any]] = None,
) -> FastAPI:
    root = data_root or os.environ.get("CHUNKER_APP_DATA", "app_data")
    manager = JobManager(
        root,
        client_factory=client_factory,
        tokenizer_id=os.environ.get("CHUNKER_TOKENIZER", GTE_TOKENIZER_ID),
    )
    app = FastAPI(title="Intelligent Chunker")
    app.state.manager = manager

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        options = "".join(
            f'<option value="{m["id"]}">{m["label"]}</option>'
            for m in model_choices()
        )
        return (
            INDEX_HTML
            .replace("/*__MODEL_OPTIONS__*/", options)
            .replace("/*__DEFAULT_MB__*/", str(default_max_request_mb()))
        )

    @app.get("/jobs")
    def list_jobs() -> List[Dict[str, Any]]:
        return manager.list_jobs()

    @app.post("/jobs")
    async def create_job(
        request: Request,
        filename: str = "upload.pdf",
        model: str = "",
        max_request_mb: float = 0.0,
    ) -> JSONResponse:
        model = model or model_choices()[0]["id"]
        mb = max_request_mb or default_max_request_mb()
        try:
            job_id = manager.create(filename, model, mb)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        dest = os.path.join(manager.jobs_dir, job_id, "upload.pdf")
        received = 0
        try:
            with open(dest, "wb") as f:
                async for part in request.stream():
                    received += len(part)
                    f.write(part)
            if received == 0:
                raise ValueError("empty upload body")
        except Exception as exc:
            manager.release(job_id)
            manager._update(job_id, state="error", error=f"upload failed: {exc}")
            raise HTTPException(status_code=400, detail=f"upload failed: {exc}")
        manager.start(job_id)
        return JSONResponse({"job_id": job_id}, status_code=201)

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str) -> Dict[str, Any]:
        status = manager.get(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return status

    @app.get("/jobs/{job_id}/results", response_class=HTMLResponse)
    def job_results(job_id: str) -> str:
        chunks_path = manager.file_path(job_id, "chunks.jsonl")
        profile_path = manager.file_path(job_id, "profile.json")
        if not chunks_path or not profile_path:
            raise HTTPException(status_code=404, detail="results not ready")
        return build_html(
            load_profile(profile_path),
            load_chunks(chunks_path),
            save_url=f"/jobs/{job_id}/curated",
        )

    @app.post("/jobs/{job_id}/curated")
    async def save_curated(job_id: str, request: Request) -> Dict[str, Any]:
        d = manager.job_dir(job_id)
        profile_path = manager.file_path(job_id, "profile.json")
        pdf_path = manager.file_path(job_id, "upload.pdf")
        if d is None or profile_path is None:
            raise HTTPException(status_code=404, detail="unknown job")
        body = (await request.body()).decode("utf-8", errors="replace")
        rows: List[Dict[str, Any]] = []
        try:
            for i, line in enumerate(body.splitlines(), start=1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict) or "text" not in row:
                        raise ValueError(f"line {i}: not a chunk record")
                    rows.append(row)
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSONL: {exc}")
        if not rows:
            raise HTTPException(status_code=400, detail="no chunks in body")

        # Quality gate on save: replace the browser's heuristic estimates
        # with exact GTE counts, then re-score fidelity against the curated
        # set so the reviewer signs off on real numbers.
        config = manager.config_for(job_id)
        counter, exact = manager.exact_counter(config)
        over_limit: List[int] = []
        for row in rows:
            if row.get("edited"):
                row["token_count"] = counter.count(row.get("text") or "")
            if (row.get("token_count") or 0) > config.max_tokens:
                over_limit.append(int(row.get("chunk_index", -1)))

        result: Dict[str, Any] = {
            "saved": len(rows),
            "exact_token_counts": exact,
            "over_max_tokens": over_limit,
        }
        if pdf_path:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            profile = DocumentProfile.from_dict(load_profile(profile_path))
            chunks = [Chunk.from_dict(r) for r in rows]
            report = fidelity_report(pdf_bytes, profile, chunks)
            if report.get("status") == "ok":
                result["fidelity"] = {
                    "coverage": report["document"]["coverage"],
                    "novelty": report["document"]["novelty"],
                }
            else:
                result["fidelity"] = {"status": report.get("reason", "skipped")}

        out = os.path.join(d, "curated.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return result

    @app.get("/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str) -> FileResponse:
        path = manager.file_path(job_id, name)
        if path is None:
            raise HTTPException(status_code=404, detail="no such file")
        return FileResponse(path, filename=name)

    return app


# uvicorn target: `uvicorn intelligent_chunker.webapp.app:app`
app = create_app()
