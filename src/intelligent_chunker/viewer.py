"""Build a self-contained HTML viewer/curation tool for a chunks.jsonl +
profile.json pair.

The data is embedded directly in the page (no server, no network), so the
output is a single file you can open in any browser. Chunk text is rendered via
the DOM (textContent), so document content can't break the page or inject HTML.

Beyond browsing, the page is a curation editor: a reviewer can exclude chunks
(or whole sections) via checkboxes, edit chunk text in place, and download the
curated result as a new chunks.jsonl -- excluded chunks dropped, edits applied
-- which feeds `export` and everything downstream unchanged. Decisions persist
in the browser's localStorage (keyed by source file + chunk count) so a review
survives closing the tab; Reset clears them. Edited chunks get a heuristic
token estimate and an ``edited: true`` marker, since the exact GTE tokenizer
isn't available in the browser.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def load_chunks(path: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_profile(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _embed(data: Any) -> str:
    """JSON-encode for safe inlining inside a <script> tag."""
    # Escape both sequences the HTML spec treats specially in script data:
    # "</" (closes the tag) and "<!--" (shifts the parser into escaped state).
    return (
        json.dumps(data, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
    )


def build_html(
    profile: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    save_url: str = "",
) -> str:
    """Render the viewer/curation page.

    ``save_url`` (when the page is served by the web app) adds a "Save
    curated to app" button that POSTs the curated JSONL back to the server;
    the standalone CLI viewer omits it and keeps download-only behavior.
    """
    payload = _embed(
        {"profile": profile, "chunks": chunks, "save_url": save_url}
    )
    return _TEMPLATE.replace("/*__DATA__*/", "const DATA = " + payload + ";")


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Intelligent Chunker — Review &amp; Curate</title>
<style>
  :root {
    --bg:#f6f7f9; --panel:#fff; --ink:#1d2330; --muted:#6b7280;
    --line:#e5e7eb; --accent:#2563eb; --chip:#eef2ff; --chip-ink:#3730a3;
    --ok:#dcfce7; --ok-ink:#166534; --warn:#fee2e2; --warn-ink:#991b1b;
  }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         color:var(--ink); background:var(--bg); }
  header { background:var(--panel); border-bottom:1px solid var(--line); padding:12px 20px; }
  .hrow { display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap; }
  header h1 { margin:0; font-size:16px; }
  header .sub { color:var(--muted); font-size:12px; margin-top:2px; }
  .hactions { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .hstats { color:var(--muted); font-size:12px; }
  .hbtn { font:inherit; font-size:13px; padding:7px 12px; border-radius:8px; cursor:pointer;
          border:1px solid var(--line); background:var(--panel); color:var(--ink); }
  .hbtn:hover { border-color:var(--accent); }
  .hbtn.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  .layout { display:flex; align-items:flex-start; gap:0; }
  aside { width:340px; min-width:340px; height:calc(100vh - 62px); overflow:auto;
          background:var(--panel); border-right:1px solid var(--line); padding:16px; position:sticky; top:0; }
  main { flex:1; padding:16px 20px; height:calc(100vh - 62px); overflow:auto; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:18px 0 8px; }
  .meta-row { display:flex; justify-content:space-between; gap:10px; padding:3px 0; border-bottom:1px dashed var(--line); }
  .meta-row .k { color:var(--muted); }
  .meta-row .v { text-align:right; font-weight:500; }
  .secrow { display:flex; align-items:center; gap:7px; margin:5px 0; }
  .secrow .spacer { width:16px; min-width:16px; }
  .seccb { width:16px; height:16px; accent-color:var(--accent); cursor:pointer; }
  .sec { display:flex; justify-content:space-between; align-items:center; gap:8px; flex:1;
         text-align:left; background:none; border:1px solid var(--line); border-radius:8px;
         padding:7px 10px; cursor:pointer; color:inherit; font:inherit; }
  .sec:hover { border-color:var(--accent); }
  .sec.active { background:var(--accent); color:#fff; border-color:var(--accent); }
  .sec .t { display:flex; flex-direction:column; }
  .sec .ttl { font-weight:600; }
  .sec .pg { font-size:11px; opacity:.8; }
  .sec .ct { font-size:11px; background:var(--chip); color:var(--chip-ink); border-radius:10px; padding:1px 8px; }
  .sec.active .ct { background:rgba(255,255,255,.25); color:#fff; }
  details { margin:6px 0; } summary { cursor:pointer; color:var(--muted); }
  .gloss { padding:5px 0; border-bottom:1px dashed var(--line); }
  .gloss .term { font-weight:600; }
  .gloss .def { color:var(--muted); font-size:13px; }
  .controls { display:flex; gap:10px; align-items:center; margin-bottom:14px; flex-wrap:wrap; }
  .controls input { flex:1; min-width:220px; padding:8px 12px; border:1px solid var(--line); border-radius:8px; font:inherit; }
  .stats { color:var(--muted); font-size:12px; }
  .chunk { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin-bottom:12px; }
  .chunk.excluded { opacity:.45; border-style:dashed; }
  .chunk .bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:8px; font-size:12px; color:var(--muted); }
  .chunk .idx { font-weight:700; color:var(--ink); }
  .chunk .sect { background:var(--chip); color:var(--chip-ink); border-radius:10px; padding:1px 8px; font-weight:600; }
  .chunk .tok { margin-left:auto; }
  .chunk .text { white-space:pre-wrap; }
  .inc { display:flex; align-items:center; gap:5px; cursor:pointer; color:var(--ink); user-select:none; }
  .inc input { width:15px; height:15px; accent-color:var(--accent); cursor:pointer; }
  .editbtn { font:inherit; font-size:12px; padding:3px 10px; border-radius:7px; cursor:pointer;
             border:1px solid var(--line); background:var(--panel); color:var(--ink); }
  .editbtn:hover { border-color:var(--accent); }
  .editor { width:100%; min-height:150px; font:inherit; padding:10px; border:1px solid var(--accent);
            border-radius:8px; resize:vertical; }
  .editrow { display:flex; gap:8px; margin-top:8px; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
  .chip { background:var(--chip); color:var(--chip-ink); border-radius:10px; padding:1px 8px; font-size:11px; }
  .chip.xref { background:#fef3c7; color:#92400e; }
  .chip.edited { background:var(--ok); color:var(--ok-ink); }
  .chip.flag { background:var(--warn); color:var(--warn-ink); font-weight:600; }
  .flagdetail { background:var(--warn); color:var(--warn-ink); border-radius:8px;
                padding:6px 10px; margin-bottom:8px; font-size:12px; }
  .savestat { color:var(--muted); font-size:12px; max-width:340px; }
  .empty { color:var(--muted); padding:30px; text-align:center; }
  .flagtoggle { display:flex; align-items:center; gap:5px; color:var(--ink);
                cursor:pointer; user-select:none; white-space:nowrap; }
</style>
</head>
<body>
<header>
  <div class="hrow">
    <div>
      <h1 id="docTitle"></h1>
      <div class="sub" id="docSub"></div>
    </div>
    <div class="hactions">
      <span class="hstats" id="curStats"></span>
      <span class="savestat" id="saveStat"></span>
      <button class="hbtn" id="resetBtn" title="Clear all curation decisions for this document">Reset</button>
      <button class="hbtn" id="saveBtn" style="display:none" title="Save the curated JSONL back to the app (server re-checks token counts and fidelity)">Save curated to app</button>
      <button class="hbtn primary" id="dlBtn" title="Download a chunks.jsonl with exclusions dropped and edits applied">Download curated JSONL</button>
    </div>
  </div>
</header>
<div class="layout">
  <aside>
    <h2>Document</h2>
    <div id="metaBox"></div>
    <h2>Sections</h2>
    <div id="sectionBox"></div>
    <div id="glossaryWrap"></div>
    <div id="xrefWrap"></div>
  </aside>
  <main>
    <div class="controls">
      <input id="search" type="search" placeholder="Search chunk text, keywords, section…">
      <label class="flagtoggle" id="flagToggleWrap" style="display:none">
        <input type="checkbox" id="flagOnly"> flagged only (<span id="flagCount"></span>)
      </label>
      <span class="stats" id="stats"></span>
    </div>
    <div id="chunkList"></div>
  </main>
</div>
<script>/*__DATA__*/</script>
<script>
(function () {
  const profile = DATA.profile || {};
  const chunks = DATA.chunks || [];
  const saveUrl = DATA.save_url || "";
  let activeSection = null;
  let query = "";
  let flaggedOnly = false;

  // Fidelity flags (profile.fidelity.chunks) keyed by chunk_index: the
  // triage the reviewer should clear first.
  const flagByIndex = {};
  (((profile.fidelity || {}).chunks) || []).forEach(f => {
    flagByIndex[f.chunk_index] = f;
  });
  const flaggedTotal = Object.keys(flagByIndex).length;

  // --- curation state (persisted in localStorage so a review survives a
  // closed tab; keyed by document so different SPDs don't collide) ---------
  const storeKey = "chunker-curation:" + (profile.source_file || "doc") + ":" + chunks.length;
  let decisions = {};   // chunk_index -> { inc: false } and/or { text: "..." }
  try { decisions = JSON.parse(localStorage.getItem(storeKey) || "{}") || {}; }
  catch (e) { decisions = {}; }
  function saveDecisions() {
    try { localStorage.setItem(storeKey, JSON.stringify(decisions)); } catch (e) {}
  }
  const dec = c => decisions[c.chunk_index] || {};
  const isIncluded = c => dec(c).inc !== false;
  const isEdited = c => dec(c).text != null;
  const effectiveText = c => (dec(c).text != null ? dec(c).text : c.text);
  function touch(c) { return decisions[c.chunk_index] || (decisions[c.chunk_index] = {}); }
  function prune(c) {
    const d = decisions[c.chunk_index];
    if (d && d.inc === undefined && d.text === undefined) delete decisions[c.chunk_index];
  }
  function setIncluded(c, v) {
    const d = touch(c);
    if (v) delete d.inc; else d.inc = false;
    prune(c); saveDecisions();
  }
  function setText(c, t) {
    const d = touch(c);
    if (t === c.text) delete d.text; else d.text = t;
    prune(c); saveDecisions();
  }
  // Mirrors the pipeline's HeuristicTokenCounter: the exact GTE tokenizer
  // isn't available in a browser, so edited chunks get an estimate.
  function estTokens(t) {
    const pieces = (t.match(/\w+|[^\w\s]/g) || []).length;
    return Math.floor(pieces * 1.3) + 1;
  }

  function curatedJsonl() {
    const lines = [];
    chunks.forEach(c => {
      if (!isIncluded(c)) return;
      const out = Object.assign({}, c);
      if (isEdited(c)) {
        out.text = effectiveText(c);
        out.token_count = estTokens(out.text);
        out.edited = true;
      }
      lines.push(JSON.stringify(out));
    });
    return lines.join("\n") + "\n";
  }

  function downloadCurated() {
    const base = (profile.source_file || "chunks").replace(/\.pdf$/i, "");
    const blob = new Blob([curatedJsonl()], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = base + ".curated.jsonl";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  async function saveCurated() {
    const stat = document.getElementById("saveStat");
    stat.textContent = "saving…";
    let resp;
    try {
      resp = await fetch(saveUrl, { method: "POST", body: curatedJsonl() });
    } catch (e) {
      stat.textContent = "save failed: " + e;
      return;
    }
    if (!resp.ok) {
      const detail = (await resp.json().catch(() => ({}))).detail || resp.statusText;
      stat.textContent = "save failed: " + detail;
      return;
    }
    const r = await resp.json();
    const parts = ["saved " + r.saved];
    if (r.fidelity && r.fidelity.coverage != null) {
      parts.push("coverage " + r.fidelity.coverage + " · novelty " + r.fidelity.novelty);
    }
    if (!r.exact_token_counts) parts.push("WARNING: heuristic token counts");
    if (r.over_max_tokens && r.over_max_tokens.length) {
      parts.push("OVER TOKEN LIMIT: chunk " + r.over_max_tokens.join(", "));
    }
    stat.textContent = parts.join("  ·  ");
  }

  const el = (tag, cls, txt) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt != null) n.textContent = txt;
    return n;
  };

  // Header + metadata
  document.getElementById("docTitle").textContent =
    profile.title || profile.source_file || "Document";
  document.getElementById("docSub").textContent =
    [profile.doc_type, profile.source_file, profile.page_count ? profile.page_count + " pages" : ""]
      .filter(Boolean).join("  ·  ");
  document.getElementById("dlBtn").onclick = downloadCurated;
  if (saveUrl) {
    const saveBtn = document.getElementById("saveBtn");
    saveBtn.style.display = "";
    saveBtn.onclick = saveCurated;
  }
  if (flaggedTotal) {
    document.getElementById("flagToggleWrap").style.display = "";
    document.getElementById("flagCount").textContent = String(flaggedTotal);
    document.getElementById("flagOnly").addEventListener("change", e => {
      flaggedOnly = e.target.checked;
      render();
    });
  }
  document.getElementById("resetBtn").onclick = () => {
    if (!Object.keys(decisions).length) return;
    if (!confirm("Clear all include/exclude decisions and edits for this document?")) return;
    decisions = {};
    try { localStorage.removeItem(storeKey); } catch (e) {}
    render();
  };

  const metaBox = document.getElementById("metaBox");
  const metaRows = [
    ["Plan", profile.plan_name],
    ["Sponsor", profile.sponsor],
    ["Effective", (profile.effective_dates || []).join(", ")],
    ["Sections", (profile.sections || []).length],
    ["Chunks", chunks.length],
  ];
  metaRows.forEach(([k, v]) => {
    if (v === "" || v == null) return;
    const row = el("div", "meta-row");
    row.appendChild(el("span", "k", k));
    row.appendChild(el("span", "v", String(v)));
    metaBox.appendChild(row);
  });

  // Counts per section
  const counts = {};
  chunks.forEach(c => { counts[c.section_title] = (counts[c.section_title] || 0) + 1; });

  // Section list: a bulk include/exclude checkbox + a filter button per row.
  const sectionBox = document.getElementById("sectionBox");
  function makeSectionButton(title, pages, n, type) {
    const b = el("button", "sec");
    b.dataset.section = title;
    const t = el("div", "t");
    t.appendChild(el("span", "ttl", title));
    const pg = [pages, type].filter(Boolean).join("  ·  ");
    if (pg) t.appendChild(el("span", "pg", pg));
    b.appendChild(t);
    b.appendChild(el("span", "ct", String(n || 0)));
    b.onclick = () => { activeSection = (activeSection === title) ? null : title; render(); };
    return b;
  }
  const allRow = el("div", "secrow");
  allRow.appendChild(el("span", "spacer"));
  const allBtn = makeSectionButton("All sections", "", chunks.length, "");
  allBtn.dataset.section = "";
  allBtn.onclick = () => { activeSection = null; render(); };
  allRow.appendChild(allBtn);
  sectionBox.appendChild(allRow);
  (profile.sections || []).forEach(s => {
    const row = el("div", "secrow");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.className = "seccb";
    cb.dataset.sec = s.title;
    cb.title = "Include/exclude every chunk in this section";
    cb.onchange = () => {
      chunks.filter(c => c.section_title === s.title)
            .forEach(c => setIncluded(c, cb.checked));
      render();
    };
    row.appendChild(cb);
    const pages = "p" + s.page_start + (s.page_end !== s.page_start ? "–" + s.page_end : "");
    row.appendChild(makeSectionButton(s.title, pages, counts[s.title] || 0, s.section_type));
    sectionBox.appendChild(row);
  });

  // Glossary
  const gl = profile.glossary || [];
  if (gl.length) {
    const wrap = document.getElementById("glossaryWrap");
    const d = el("details");
    d.appendChild(el("summary", null, "Glossary (" + gl.length + ")"));
    gl.forEach(g => {
      const row = el("div", "gloss");
      row.appendChild(el("div", "term", g.term));
      if (g.definition) row.appendChild(el("div", "def", g.definition));
      d.appendChild(row);
    });
    wrap.appendChild(el("h2", null, "Glossary"));
    wrap.appendChild(d);
  }

  // Cross references
  const xr = profile.cross_references || [];
  if (xr.length) {
    const wrap = document.getElementById("xrefWrap");
    const d = el("details");
    d.appendChild(el("summary", null, "Cross-references (" + xr.length + ")"));
    xr.forEach(x => d.appendChild(el("div", "gloss", x)));
    wrap.appendChild(el("h2", null, "Cross-references"));
    wrap.appendChild(d);
  }

  const list = document.getElementById("chunkList");
  const stats = document.getElementById("stats");
  const curStats = document.getElementById("curStats");

  function matches(c) {
    if (flaggedOnly && !flagByIndex[c.chunk_index]) return false;
    if (activeSection && c.section_title !== activeSection) return false;
    if (!query) return true;
    const hay = (effectiveText(c) + " " + (c.keywords || []).join(" ") + " " + c.section_title).toLowerCase();
    return hay.includes(query);
  }

  function beginEdit(card, textDiv, c) {
    const ta = document.createElement("textarea");
    ta.className = "editor";
    ta.value = effectiveText(c);
    const row = el("div", "editrow");
    const save = el("button", "editbtn", "Save");
    const cancel = el("button", "editbtn", "Cancel");
    row.appendChild(save);
    row.appendChild(cancel);
    card.replaceChild(ta, textDiv);
    card.insertBefore(row, ta.nextSibling);
    ta.focus();
    save.onclick = () => { setText(c, ta.value); render(); };
    cancel.onclick = () => render();
  }

  function chunkCard(c) {
    const card = el("div", "chunk" + (isIncluded(c) ? "" : " excluded"));
    const bar = el("div", "bar");
    const lab = el("label", "inc");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = isIncluded(c);
    cb.onchange = () => { setIncluded(c, cb.checked); render(); };
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode("include"));
    bar.appendChild(lab);
    bar.appendChild(el("span", "idx", "#" + c.chunk_index));
    bar.appendChild(el("span", "sect", c.section_title));
    const pages = "p" + c.page_start + (c.page_end !== c.page_start ? "–" + c.page_end : "");
    bar.appendChild(el("span", null, pages));
    if (c.section_type) bar.appendChild(el("span", null, c.section_type));
    if (isEdited(c)) {
      bar.appendChild(el("span", "chip edited", "edited · ~" + estTokens(effectiveText(c)) + " tok"));
    }
    bar.appendChild(el("span", "tok",
      isEdited(c) ? "was " + (c.token_count != null ? c.token_count : "?") + " tok"
                  : (c.token_count != null ? c.token_count : "?") + " tok"));
    const editBtn = el("button", "editbtn", "Edit");
    bar.appendChild(editBtn);
    card.appendChild(bar);
    const flag = flagByIndex[c.chunk_index];
    if (flag) {
      bar.insertBefore(
        el("span", "chip flag", "⚠ novelty " + flag.novelty), editBtn
      );
      const lines = flag.novel_lines || [];
      const hints = flag.novel_line_hints || [];
      const details = lines.map(
        (l, i) => "“" + l + "”" + (hints[i] ? "  [" + hints[i] + "]" : "")
      );
      card.appendChild(el(
        "div", "flagdetail",
        details.length
          ? "Fidelity: " + details.join("   ")
          : "Fidelity: chunk text poorly matched to its claimed pages"
      ));
    }
    const textDiv = el("div", "text", effectiveText(c));
    card.appendChild(textDiv);
    editBtn.onclick = () => beginEdit(card, textDiv, c);
    const chips = el("div", "chips");
    (c.keywords || []).forEach(k => chips.appendChild(el("span", "chip", k)));
    (c.cross_references || []).forEach(x => chips.appendChild(el("span", "chip xref", "→ " + x)));
    if (chips.childNodes.length) card.appendChild(chips);
    return card;
  }

  function render() {
    document.querySelectorAll(".sec").forEach(b => {
      const s = b.dataset.section;
      b.classList.toggle("active", (activeSection || "") === s);
    });
    document.querySelectorAll(".seccb").forEach(cb => {
      const secChunks = chunks.filter(c => c.section_title === cb.dataset.sec);
      const inc = secChunks.filter(isIncluded).length;
      cb.checked = secChunks.length > 0 && inc === secChunks.length;
      cb.indeterminate = inc > 0 && inc < secChunks.length;
    });
    const shown = chunks.filter(matches);
    list.innerHTML = "";
    if (!shown.length) {
      list.appendChild(el("div", "empty", "No chunks match."));
    } else {
      shown.forEach(c => list.appendChild(chunkCard(c)));
    }
    const toks = shown.map(c => c.token_count || 0).filter(t => t > 0).sort((a, b) => a - b);
    const med = toks.length ? toks[Math.floor(toks.length / 2)] : 0;
    const exShown = shown.filter(c => !isIncluded(c)).length;
    stats.textContent = shown.length + " of " + chunks.length + " chunks"
      + (exShown ? "  ·  " + exShown + " excluded here" : "")
      + (toks.length ? "  ·  median " + med + " tok  ·  range " + toks[0] + "–" + toks[toks.length - 1] : "");
    const included = chunks.filter(isIncluded).length;
    const edited = chunks.filter(isEdited).length;
    curStats.textContent = "curated: " + included + " of " + chunks.length + " included"
      + (edited ? "  ·  " + edited + " edited" : "");
  }

  document.getElementById("search").addEventListener("input", e => {
    query = e.target.value.trim().toLowerCase();
    render();
  });

  render();
})();
</script>
</body>
</html>
"""
