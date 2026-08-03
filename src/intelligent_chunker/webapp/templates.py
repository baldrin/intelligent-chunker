"""The upload/progress index page (self-contained, no external assets)."""

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Intelligent Chunker</title>
<style>
  :root {
    --bg:#f6f7f9; --panel:#fff; --ink:#1d2330; --muted:#6b7280;
    --line:#e5e7eb; --accent:#2563eb; --chip:#eef2ff; --chip-ink:#3730a3;
    --ok:#dcfce7; --ok-ink:#166534; --warn:#fee2e2; --warn-ink:#991b1b;
    --amber:#fef3c7; --amber-ink:#92400e;
  }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         color:var(--ink); background:var(--bg); }
  .wrap { max-width:960px; margin:0 auto; padding:28px 20px; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:var(--muted); margin-bottom:22px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:12px;
          padding:18px 20px; margin-bottom:18px; }
  .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin:10px 0; }
  label { font-weight:600; min-width:110px; }
  select, input[type=file] { font:inherit; padding:7px 10px; border:1px solid var(--line);
          border-radius:8px; background:var(--panel); }
  button { font:inherit; padding:9px 16px; border-radius:8px; cursor:pointer;
           border:1px solid var(--accent); background:var(--accent); color:#fff; }
  button:disabled { opacity:.5; cursor:default; }
  .bar { height:10px; background:var(--line); border-radius:6px; overflow:hidden; margin:8px 0; }
  .bar > div { height:100%; width:0%; background:var(--accent); transition:width .4s; }
  .phase { color:var(--muted); font-size:13px; }
  .err { background:var(--warn); color:var(--warn-ink); border-radius:8px; padding:10px 12px;
         white-space:pre-wrap; }
  .gate { border-radius:8px; padding:10px 12px; margin-top:10px; font-size:13px; }
  .gate.ok { background:var(--ok); color:var(--ok-ink); }
  .gate.warn { background:var(--amber); color:var(--amber-ink); }
  .gate.bad { background:var(--warn); color:var(--warn-ink); }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; }
  .num { text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
  a { color:var(--accent); }
  .muted { color:var(--muted); }
</style>
</head>
<body>
<div class="wrap">
  <h1>Intelligent Chunker</h1>
  <div class="sub">Upload an SPD, process it, then review and curate the chunks.</div>

  <div class="card">
    <div class="row">
      <label for="pdf">SPD (PDF)</label>
      <input type="file" id="pdf" accept="application/pdf">
    </div>
    <div class="row">
      <label for="model">Model</label>
      <select id="model">/*__MODEL_OPTIONS__*/</select>
    </div>
    <div class="row">
      <label for="mb">Request budget</label>
      <select id="mb">
        <option value="3">3 MB — Databricks serving</option>
        <option value="25">25 MB — direct API</option>
      </select>
    </div>
    <div class="row">
      <button id="start">Process</button>
      <span class="muted" id="hint"></span>
    </div>
    <div id="progressWrap" style="display:none">
      <div class="bar"><div id="barFill"></div></div>
      <div class="phase" id="phase"></div>
    </div>
    <div id="msg"></div>
  </div>

  <div class="card">
    <h2 style="font-size:14px;margin:0 0 10px">Jobs</h2>
    <table>
      <thead><tr><th>File</th><th>Model</th><th>State</th><th>Quality</th>
        <th class="num">In</th><th class="num">Out</th><th class="num">Cache W</th><th class="num">Cache R</th><th class="num">Cost</th><th></th></tr></thead>
      <tbody id="jobRows"></tbody>
    </table>
  </div>
</div>
<script>
(function () {
  const el = (tag, cls, txt) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt != null) n.textContent = txt;
    return n;
  };
  const $ = id => document.getElementById(id);
  const DEFAULT_MB = /*__DEFAULT_MB__*/;
  $("mb").value = String(Math.round(DEFAULT_MB));

  const PHASE_LABELS = { pass1: "Pass 1 — mapping the document",
                         pass2: "Pass 2 — chunking sections",
                         fidelity: "Fidelity check" };

  function gateInfo(q) {
    // Quality gate: everything a reviewer should know before opening results.
    if (!q) return null;
    const problems = [];
    if (!q.exact_tokens) problems.push("HEURISTIC token sizing (exact GTE tokenizer unavailable) — chunk sizes are estimates");
    if (q.fidelity_status !== "ok") problems.push("fidelity not measured (" + q.fidelity_status + ")");
    if (q.coverage_ok === false) problems.push("coverage " + q.coverage + " below threshold");
    if (q.novelty_ok === false) problems.push("novelty " + q.novelty + " above threshold");
    const notes = [];
    if (q.coverage != null) notes.push("coverage " + q.coverage + " · novelty " + q.novelty);
    if (q.flagged_chunks) {
      notes.push(q.flagged_chunks + " flagged chunk(s): "
        + q.flags_misattributed + " misattributed · "
        + q.flags_not_in_layer + " not-in-layer — review these first");
    }
    return { problems, notes };
  }

  function gateNode(q) {
    const info = gateInfo(q);
    if (!info) return null;
    const cls = info.problems.length ? (info.problems.some(p => p.startsWith("HEURISTIC")) ? "bad" : "warn")
                                     : (q.flagged_chunks ? "warn" : "ok");
    const node = el("div", "gate " + cls);
    const lines = info.problems.concat(info.notes);
    node.textContent = lines.length ? lines.join("  |  ") : "quality checks passed";
    return node;
  }

  function fmtTokens(n) {
    if (n == null) return "—";
    if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (n >= 1e4) return Math.round(n / 1e3) + "k";
    return n.toLocaleString();
  }

  function usageCells(tr, j) {
    // Jobs from before structured usage persisted only a summary string.
    const u = (j.usage && typeof j.usage === "object") ? j.usage : null;
    const legacy = (typeof j.usage === "string") ? j.usage : null;
    ["input_tokens", "output_tokens",
     "cache_creation_input_tokens", "cache_read_input_tokens"].forEach(f => {
      const n = u ? u[f] : null;
      const td = el("td", "num", fmtTokens(n));
      if (n != null) td.title = n.toLocaleString() + " tokens";
      else if (legacy) td.title = legacy;
      tr.appendChild(td);
    });
    const cost = u ? u.estimated_cost_usd : null;
    const td = el("td", "num", cost != null ? "$" + cost.toFixed(4) : "—");
    if (u && cost == null) td.title = "no pricing known for this model";
    else if (legacy) td.title = legacy;
    tr.appendChild(td);
  }

  async function refreshJobs() {
    const rows = $("jobRows");
    let jobs = [];
    try { jobs = await (await fetch("/jobs")).json(); } catch (e) { return; }
    rows.innerHTML = "";
    jobs.forEach(j => {
      const tr = el("tr");
      tr.appendChild(el("td", null, j.filename || j.job_id));
      tr.appendChild(el("td", null, (j.model || "").replace("databricks-claude-", "").replace("claude-", "")));
      const st = el("td", null, j.state + (j.state === "running" && j.phase ? " (" + j.phase + " " + j.done + "/" + j.total + ")" : ""));
      tr.appendChild(st);
      const q = el("td");
      if (j.quality) {
        const g = gateNode(j.quality);
        if (g) { g.style.marginTop = "0"; q.appendChild(g); }
      } else if (j.error) {
        q.appendChild(el("span", "muted", "error"));
      }
      tr.appendChild(q);
      usageCells(tr, j);
      const links = el("td");
      if (j.state === "done") {
        const a = el("a", null, "review");
        a.href = "/jobs/" + j.job_id + "/results";
        links.appendChild(a);
        if (j.has_curated) {
          links.appendChild(document.createTextNode("  ·  "));
          const c = el("a", null, "curated.jsonl");
          c.href = "/jobs/" + j.job_id + "/files/curated.jsonl";
          links.appendChild(c);
        }
      }
      tr.appendChild(links);
      rows.appendChild(tr);
    });
  }

  async function poll(jobId) {
    $("progressWrap").style.display = "";
    for (;;) {
      let s;
      try { s = await (await fetch("/jobs/" + jobId)).json(); }
      catch (e) { await new Promise(r => setTimeout(r, 2000)); continue; }
      if (s.state === "running" && s.total) {
        $("barFill").style.width = Math.round(100 * s.done / s.total) + "%";
        $("phase").textContent = (PHASE_LABELS[s.phase] || s.phase) + " — " + s.done + " of " + s.total;
      } else {
        $("phase").textContent = s.state;
      }
      if (s.state === "done") {
        $("barFill").style.width = "100%";
        const msg = $("msg");
        msg.innerHTML = "";
        const g = gateNode(s.quality);
        if (g) msg.appendChild(g);
        const p = el("p");
        const a = el("a", null, "Open the review & curation page");
        a.href = "/jobs/" + jobId + "/results";
        p.appendChild(a);
        if (s.usage) p.appendChild(el("div", "muted",
          typeof s.usage === "object" ? s.usage.summary : s.usage));
        msg.appendChild(p);
        break;
      }
      if (s.state === "error") {
        $("msg").innerHTML = "";
        $("msg").appendChild(el("div", "err", s.error || "processing failed"));
        break;
      }
      await new Promise(r => setTimeout(r, 2000));
    }
    $("start").disabled = false;
    refreshJobs();
  }

  $("start").onclick = async () => {
    const file = $("pdf").files[0];
    if (!file) { $("hint").textContent = "Choose a PDF first."; return; }
    $("hint").textContent = "";
    $("msg").innerHTML = "";
    $("start").disabled = true;
    const params = new URLSearchParams({
      filename: file.name,
      model: $("model").value,
      max_request_mb: $("mb").value,
    });
    let resp;
    try {
      resp = await fetch("/jobs?" + params, { method: "POST", body: file });
    } catch (e) {
      $("msg").appendChild(el("div", "err", "upload failed: " + e));
      $("start").disabled = false;
      return;
    }
    if (!resp.ok) {
      const detail = (await resp.json().catch(() => ({}))).detail || resp.statusText;
      $("msg").appendChild(el("div", "err", detail));
      $("start").disabled = false;
      return;
    }
    const { job_id } = await resp.json();
    refreshJobs();
    poll(job_id);
  };

  refreshJobs();
  setInterval(refreshJobs, 10000);
})();
</script>
</body>
</html>
"""
