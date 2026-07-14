"""Build a self-contained HTML viewer for a chunks.jsonl + profile.json pair.

The data is embedded directly in the page (no server, no network), so the
output is a single file you can open in any browser. Chunk text is rendered via
the DOM (textContent), so document content can't break the page or inject HTML.
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


def build_html(profile: Dict[str, Any], chunks: List[Dict[str, Any]]) -> str:
    payload = _embed({"profile": profile, "chunks": chunks})
    return _TEMPLATE.replace("/*__DATA__*/", "const DATA = " + payload + ";")


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Intelligent Chunker — Viewer</title>
<style>
  :root {
    --bg:#f6f7f9; --panel:#fff; --ink:#1d2330; --muted:#6b7280;
    --line:#e5e7eb; --accent:#2563eb; --chip:#eef2ff; --chip-ink:#3730a3;
  }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         color:var(--ink); background:var(--bg); }
  header { background:var(--panel); border-bottom:1px solid var(--line); padding:14px 20px; }
  header h1 { margin:0; font-size:16px; }
  header .sub { color:var(--muted); font-size:12px; margin-top:2px; }
  .layout { display:flex; align-items:flex-start; gap:0; }
  aside { width:320px; min-width:320px; height:calc(100vh - 56px); overflow:auto;
          background:var(--panel); border-right:1px solid var(--line); padding:16px; position:sticky; top:0; }
  main { flex:1; padding:16px 20px; height:calc(100vh - 56px); overflow:auto; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:18px 0 8px; }
  .meta-row { display:flex; justify-content:space-between; gap:10px; padding:3px 0; border-bottom:1px dashed var(--line); }
  .meta-row .k { color:var(--muted); }
  .meta-row .v { text-align:right; font-weight:500; }
  .sec { display:flex; justify-content:space-between; align-items:center; gap:8px; width:100%;
         text-align:left; background:none; border:1px solid var(--line); border-radius:8px;
         padding:7px 10px; margin:5px 0; cursor:pointer; color:inherit; font:inherit; }
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
  .chunk .bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:8px; font-size:12px; color:var(--muted); }
  .chunk .idx { font-weight:700; color:var(--ink); }
  .chunk .sect { background:var(--chip); color:var(--chip-ink); border-radius:10px; padding:1px 8px; font-weight:600; }
  .chunk .tok { margin-left:auto; }
  .chunk .text { white-space:pre-wrap; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
  .chip { background:var(--chip); color:var(--chip-ink); border-radius:10px; padding:1px 8px; font-size:11px; }
  .chip.xref { background:#fef3c7; color:#92400e; }
  .empty { color:var(--muted); padding:30px; text-align:center; }
</style>
</head>
<body>
<header>
  <h1 id="docTitle"></h1>
  <div class="sub" id="docSub"></div>
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
  let activeSection = null;
  let query = "";

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

  // Section list (click to filter)
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
  const allBtn = makeSectionButton("All sections", "", chunks.length, "");
  allBtn.dataset.section = "";
  allBtn.onclick = () => { activeSection = null; render(); };
  sectionBox.appendChild(allBtn);
  (profile.sections || []).forEach(s => {
    const pages = "p" + s.page_start + (s.page_end !== s.page_start ? "–" + s.page_end : "");
    sectionBox.appendChild(makeSectionButton(s.title, pages, counts[s.title] || 0, s.section_type));
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

  function matches(c) {
    if (activeSection && c.section_title !== activeSection) return false;
    if (!query) return true;
    const hay = (c.text + " " + (c.keywords || []).join(" ") + " " + c.section_title).toLowerCase();
    return hay.includes(query);
  }

  function chunkCard(c) {
    const card = el("div", "chunk");
    const bar = el("div", "bar");
    bar.appendChild(el("span", "idx", "#" + c.chunk_index));
    bar.appendChild(el("span", "sect", c.section_title));
    const pages = "p" + c.page_start + (c.page_end !== c.page_start ? "–" + c.page_end : "");
    bar.appendChild(el("span", null, pages));
    if (c.section_type) bar.appendChild(el("span", null, c.section_type));
    bar.appendChild(el("span", "tok", (c.token_count != null ? c.token_count : "?") + " tok"));
    card.appendChild(bar);
    card.appendChild(el("div", "text", c.text));
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
    const shown = chunks.filter(matches);
    list.innerHTML = "";
    if (!shown.length) {
      list.appendChild(el("div", "empty", "No chunks match."));
    } else {
      shown.forEach(c => list.appendChild(chunkCard(c)));
    }
    const toks = shown.map(c => c.token_count || 0).filter(t => t > 0).sort((a, b) => a - b);
    const med = toks.length ? toks[Math.floor(toks.length / 2)] : 0;
    stats.textContent = shown.length + " of " + chunks.length + " chunks"
      + (toks.length ? "  ·  median " + med + " tok  ·  range " + toks[0] + "–" + toks[toks.length - 1] : "");
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
