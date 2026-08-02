"""Background job runner: one pipeline run at a time over a data root.

All files live under ``<data_root>/jobs/<job_id>/`` -- on Databricks the data
root MUST be a UC Volume path (the app instance's local disk is tiny and
filling it crashes the app), so nothing here ever writes outside the root.

State is kept in memory (lock-guarded, updated by the worker thread's
progress callbacks) and journaled to ``job.json`` on every change, so a
restarted app can still list and serve finished jobs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from ..config import ChunkerConfig
from ..fidelity import COVERAGE_WARN_BELOW, NOVELTY_WARN_ABOVE
from ..llm import make_client
from ..pipeline import run
from ..tokenizer import HFTokenCounter, get_token_counter

logger = logging.getLogger(__name__)

_JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# job.json / status fields the API returns verbatim.
_PUBLIC_FIELDS = (
    "job_id",
    "filename",
    "model",
    "max_request_mb",
    "state",
    "phase",
    "done",
    "total",
    "created_at",
    "finished_at",
    "error",
    "usage",
    "quality",
    "has_curated",
)


def quality_summary(
    fidelity: Optional[Dict[str, Any]], exact_tokens: bool
) -> Dict[str, Any]:
    """Reviewer-facing quality gate: scores vs thresholds + flag triage counts.

    ``exact_tokens`` is whether the real GTE tokenizer sized the chunks --
    heuristic sizing is a quality degradation that must be surfaced loudly,
    never buried in a log line.
    """
    out: Dict[str, Any] = {
        "exact_tokens": exact_tokens,
        "fidelity_status": (fidelity or {}).get("status", "missing"),
    }
    if fidelity and fidelity.get("status") == "ok":
        doc = fidelity["document"]
        flags = fidelity.get("chunks", [])
        hints = [h for f in flags for h in f.get("novel_line_hints", [])]
        out.update(
            {
                "coverage": doc["coverage"],
                "novelty": doc["novelty"],
                "coverage_ok": doc["coverage"] >= COVERAGE_WARN_BELOW,
                "novelty_ok": doc["novelty"] <= NOVELTY_WARN_ABOVE,
                "flagged_chunks": len(flags),
                "flags_misattributed": sum(
                    1 for h in hints if "misattributed" in h
                ),
                "flags_not_in_layer": sum(
                    1 for h in hints if "not in the text layer" in h
                ),
            }
        )
    return out


class JobManager:
    """Owns the jobs directory and the single background worker."""

    def __init__(
        self,
        data_root: str,
        client_factory: Optional[Callable[[], Any]] = None,
        tokenizer_id: Optional[str] = None,
    ):
        self.data_root = data_root
        self.jobs_dir = os.path.join(data_root, "jobs")
        os.makedirs(self.jobs_dir, exist_ok=True)
        self._client_factory = client_factory or make_client
        self._tokenizer_id = tokenizer_id
        self._lock = threading.Lock()
        self._active: Optional[str] = None
        self._live: Dict[str, Dict[str, Any]] = {}

    # --- paths ---------------------------------------------------------

    def job_dir(self, job_id: str) -> Optional[str]:
        """Directory for ``job_id``, or None for bad ids / unknown jobs."""
        if not _JOB_ID_RE.match(job_id):
            return None
        path = os.path.join(self.jobs_dir, job_id)
        return path if os.path.isdir(path) else None

    def file_path(self, job_id: str, name: str) -> Optional[str]:
        allowed = {"upload.pdf", "chunks.jsonl", "profile.json", "curated.jsonl"}
        d = self.job_dir(job_id)
        if d is None or name not in allowed:
            return None
        path = os.path.join(d, name)
        return path if os.path.isfile(path) else None

    # --- status --------------------------------------------------------

    def _write_status(self, job_id: str, status: Dict[str, Any]) -> None:
        d = os.path.join(self.jobs_dir, job_id)
        with open(os.path.join(d, "job.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            status = self._live.setdefault(job_id, {"job_id": job_id})
            status.update(fields)
            snapshot = dict(status)
        self._write_status(job_id, snapshot)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            live = self._live.get(job_id)
            if live is not None:
                status = dict(live)
                return self._public(job_id, status)
        d = self.job_dir(job_id)
        if d is None:
            return None
        try:
            with open(os.path.join(d, "job.json"), "r", encoding="utf-8") as f:
                return self._public(job_id, json.load(f))
        except (OSError, json.JSONDecodeError):
            return None

    def _public(self, job_id: str, status: Dict[str, Any]) -> Dict[str, Any]:
        status["has_curated"] = self.file_path(job_id, "curated.jsonl") is not None
        return {k: status.get(k) for k in _PUBLIC_FIELDS}

    def list_jobs(self) -> List[Dict[str, Any]]:
        jobs = []
        try:
            ids = os.listdir(self.jobs_dir)
        except OSError:
            return []
        for job_id in ids:
            if _JOB_ID_RE.match(job_id):
                status = self.get(job_id)
                if status:
                    jobs.append(status)
        jobs.sort(key=lambda s: s.get("created_at") or 0, reverse=True)
        return jobs

    # --- lifecycle -----------------------------------------------------

    def create(self, filename: str, model: str, max_request_mb: float) -> str:
        """Reserve a job id + directory; caller streams the upload into it.

        Raises RuntimeError when a job is already running (one at a time).
        """
        with self._lock:
            if self._active is not None:
                raise RuntimeError("a job is already running")
            job_id = uuid.uuid4().hex[:12]
            self._active = job_id
        os.makedirs(os.path.join(self.jobs_dir, job_id), exist_ok=True)
        self._update(
            job_id,
            filename=os.path.basename(filename) or "upload.pdf",
            model=model,
            max_request_mb=max_request_mb,
            state="uploading",
            created_at=time.time(),
        )
        return job_id

    def release(self, job_id: str) -> None:
        """Free the single-job slot (upload failed before start)."""
        with self._lock:
            if self._active == job_id:
                self._active = None

    def start(self, job_id: str) -> None:
        """Kick off the pipeline thread for an uploaded job."""
        self._update(job_id, state="queued", phase=None, done=0, total=0)
        thread = threading.Thread(
            target=self._run, args=(job_id,), daemon=True
        )
        thread.start()

    def config_for(self, job_id: str) -> ChunkerConfig:
        status = self.get(job_id) or {}
        kwargs: Dict[str, Any] = {}
        model = status.get("model")
        if model:
            kwargs["pass1_model"] = model
            kwargs["pass2_model"] = model
        if status.get("max_request_mb"):
            kwargs["max_request_mb"] = float(status["max_request_mb"])
        if self._tokenizer_id:
            kwargs["tokenizer_id"] = self._tokenizer_id
        return ChunkerConfig(**kwargs)

    def exact_counter(self, config: ChunkerConfig):
        """(counter, is_exact) -- heuristic sizing must be surfaced, not hidden."""
        counter = get_token_counter(config.tokenizer_id)
        return counter, isinstance(counter, HFTokenCounter)

    def _run(self, job_id: str) -> None:
        d = os.path.join(self.jobs_dir, job_id)
        config = self.config_for(job_id)

        def on_progress(phase: str, done: int, total: int) -> None:
            self._update(
                job_id, state="running", phase=phase, done=done, total=total
            )

        try:
            _, exact = self.exact_counter(config)
            result = run(
                os.path.join(d, "upload.pdf"),
                config=config,
                out_path=os.path.join(d, "chunks.jsonl"),
                profile_path=os.path.join(d, "profile.json"),
                client=self._client_factory(),
                on_progress=on_progress,
            )
            self._update(
                job_id,
                state="done",
                finished_at=time.time(),
                usage=result.usage.summary() if result.usage else None,
                quality=quality_summary(result.fidelity, exact),
            )
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            self._update(
                job_id, state="error", finished_at=time.time(), error=str(exc)
            )
        finally:
            with self._lock:
                if self._active == job_id:
                    self._active = None
            with self._lock:
                self._live.pop(job_id, None)
