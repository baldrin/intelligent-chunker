"""End-to-end web app flow over HTTP with a fake LLM client."""

import json
import threading
import time

import pytest
from conftest import FakeClient, make_pdf

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from intelligent_chunker.webapp.app import create_app  # noqa: E402

PASS1 = {
    "doc_type": "SPD",
    "title": "T",
    "plan_name": "",
    "sponsor": "",
    "effective_dates": [],
    "sections": [
        {"title": "Only", "section_type": "general", "summary": "",
         "page_start": 1, "page_end": 1}
    ],
    "glossary": [],
    "cross_references": [],
    "notes": "",
}
PASS2 = {"chunks": [{"text": "hello world", "keywords": [], "cross_references": []}]}


def _app(tmp_path, payloads=None):
    return create_app(
        str(tmp_path),
        client_factory=lambda: FakeClient(payloads or [PASS1, PASS2]),
    )


def _wait_done(client, job_id, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/jobs/{job_id}").json()
        if status["state"] in ("done", "error"):
            return status
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def test_full_flow_upload_process_review_curate(tmp_path):
    client = TestClient(_app(tmp_path))

    # Index page renders with the model picker.
    index = client.get("/")
    assert index.status_code == 200
    assert "Sonnet" in index.text and "Haiku" in index.text
    assert "Cache W" in index.text and "Cost" in index.text  # usage columns

    # Upload: raw body, no multipart.
    resp = client.post(
        "/jobs",
        params={"filename": "doc.pdf", "model": "claude-haiku-4-5",
                "max_request_mb": 25},
        content=make_pdf(1),
    )
    assert resp.status_code == 201
    job_id = resp.json()["job_id"]

    status = _wait_done(client, job_id)
    assert status["state"] == "done", status.get("error")
    usage = status["usage"]  # structured tracker snapshot for the jobs table
    assert usage["calls"] == 2  # Pass 1 + Pass 2
    assert usage["input_tokens"] == 200
    assert usage["output_tokens"] == 20
    assert usage["cache_creation_input_tokens"] == 10
    assert usage["cache_read_input_tokens"] == 100
    assert usage["estimated_cost_usd"] > 0  # haiku pricing is known
    assert "$" in usage["summary"]
    assert status["quality"]["fidelity_status"] in ("ok", "skipped")
    assert status["quality"]["exact_tokens"] in (True, False)

    # Job listing includes it.
    jobs = client.get("/jobs").json()
    assert [j["job_id"] for j in jobs] == [job_id]

    # Results page is the curation viewer with save-back wired in.
    results = client.get(f"/jobs/{job_id}/results")
    assert results.status_code == 200
    assert "Download curated JSONL" in results.text
    assert f"/jobs/{job_id}/curated" in results.text

    # Curate: drop nothing, edit the one chunk; server recounts tokens.
    lines = client.get(f"/jobs/{job_id}/files/chunks.jsonl").text.splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    rows[0]["text"] = "hello edited world"
    rows[0]["edited"] = True
    rows[0]["token_count"] = 999999  # browser estimate: must be replaced
    body = "\n".join(json.dumps(r) for r in rows) + "\n"
    saved = client.post(f"/jobs/{job_id}/curated", content=body)
    assert saved.status_code == 200
    out = saved.json()
    assert out["saved"] == len(rows)
    assert out["over_max_tokens"] == []  # recount replaced the bogus estimate
    assert "fidelity" in out

    on_disk = [
        json.loads(line)
        for line in client.get(
            f"/jobs/{job_id}/files/curated.jsonl"
        ).text.splitlines()
        if line.strip()
    ]
    assert on_disk[0]["text"] == "hello edited world"
    assert on_disk[0]["token_count"] < 100  # exact/heuristic recount, not 999999
    assert client.get(f"/jobs/{job_id}").json()["has_curated"] is True


def test_second_job_while_running_is_rejected(tmp_path):
    gate = threading.Event()

    class BlockingClient(FakeClient):
        def __init__(self):
            super().__init__([PASS1, PASS2])
            inner = self.messages._next

            def gated(kwargs):
                gate.wait(timeout=10)
                return inner(kwargs)

            self.messages._next = gated

    client = TestClient(create_app(str(tmp_path), client_factory=BlockingClient))
    first = client.post("/jobs", params={"filename": "a.pdf"}, content=make_pdf(1))
    assert first.status_code == 201
    second = client.post("/jobs", params={"filename": "b.pdf"}, content=make_pdf(1))
    assert second.status_code == 409
    gate.set()
    status = _wait_done(client, first.json()["job_id"])
    assert status["state"] == "done"


def test_bad_inputs(tmp_path):
    client = TestClient(_app(tmp_path))
    assert client.get("/jobs/ffffffffffff").status_code == 404
    assert client.get("/jobs/not-a-job-id").status_code == 404
    assert client.get("/jobs/../../etc/passwd").status_code == 404

    resp = client.post("/jobs", params={"filename": "doc.pdf"}, content=make_pdf(1))
    job_id = resp.json()["job_id"]
    _wait_done(client, job_id)
    bad = client.post(f"/jobs/{job_id}/curated", content="not json\n")
    assert bad.status_code == 400
    empty = client.post(f"/jobs/{job_id}/curated", content="")
    assert empty.status_code == 400
    assert client.get(f"/jobs/{job_id}/files/nope.txt").status_code == 404


def test_pipeline_error_is_reported(tmp_path):
    boom = Exception("Error code: 400 - something exploded")
    client = TestClient(_app(tmp_path, payloads=[boom]))
    resp = client.post("/jobs", params={"filename": "doc.pdf"}, content=make_pdf(1))
    status = _wait_done(client, resp.json()["job_id"])
    assert status["state"] == "error"
    assert "something exploded" in status["error"]
