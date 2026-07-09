"""Timestep × horizon grid: the recurrent analog of the J-lens slice view.

Columns are sequence positions; row k shows, for the state at each position,
the token the lens says that state is disposed to emit k steps in the future
(row 0 is the model's actual next-token prediction — the logit lens). Cell
shade encodes the lens's top-1 probability on a log-binned sequential ramp; a
ring marks cells whose top-1 matches the token the sequence actually contains
at the predicted position.

``build_grid_data`` does the tensor work; ``render_html`` is a pure
data -> HTML function producing a self-contained page (inline CSS/JS, light
and dark themes).
"""

from __future__ import annotations

import bisect
import html
import json

import torch
from torch import Tensor

from jspace.data import Vocab
from jspace.lens import HorizonLens
from jspace.model import LSTMLanguageModel

TOP_K = 5

# Cell shade encodes the *percentile* of the cell's top-1 weight within its
# own grid section (centered-readout softmax values are tiny in absolute
# terms over a 10k+ vocabulary, so absolute thresholds would flatten the
# shading channel).
N_BINS = 10

# Sequential blue ramp (reference palette). In dark mode the ramp is reversed
# so that near-zero recedes toward the dark surface and high probability pops.
SEQ_LIGHT = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#5598e7",
    "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b",
]
SEQ_DARK = list(reversed(SEQ_LIGHT))


def _assign_percentile_bins(section: dict) -> None:
    """Set each cell's ``bin`` to the percentile decile of its top-1 weight
    within this section, so the full ramp is always exercised."""
    weights = sorted(
        cell["top"][0]["prob"] for row in section["rows"] for cell in row["cells"]
    )
    n = len(weights)
    for row in section["rows"]:
        for cell in row["cells"]:
            rank = bisect.bisect_right(weights, cell["top"][0]["prob"])
            cell["bin"] = min(N_BINS - 1, rank * N_BINS // max(1, n))


def build_grid_data(
    model: LSTMLanguageModel,
    lens: HorizonLens,
    vocab: Vocab,
    token_ids: Tensor,
    blocks: list[str],
    max_horizon: int,
    min_token_count: int = 0,
    token_counts: Tensor | None = None,
) -> dict:
    """Compute per-cell top-k readouts for each (block, horizon, position).

    ``min_token_count`` (with ``token_counts``, e.g. training-corpus counts)
    masks rare tokens out of the decoded top-k. Rare tokens have
    disproportionately large unembedding norms and dominate long-horizon
    centered readouts as noise; masking them is a display filter only and
    does not touch the underlying readout.
    """
    horizons = list(range(1, max_horizon + 1))
    # Centered readout: content view (state-specific disposition), not an
    # absolute prediction of the future logits.
    lens_logits, model_logits = lens.apply(
        model, token_ids, blocks, horizons, mode="centered"
    )
    tokens = vocab.decode(token_ids)
    T = len(tokens)

    mask = None
    if min_token_count > 0:
        if token_counts is None:
            raise ValueError("min_token_count requires token_counts")
        mask = token_counts.to(model_logits.device) < min_token_count
        mask[0] = True  # always hide <unk>

    token_ids_cpu = token_ids.cpu()

    def cells_from_logits(logits: Tensor, horizon: int) -> list[dict]:
        if mask is not None:
            logits = logits.masked_fill(mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        top = probs.topk(TOP_K, dim=-1)
        # One device->host transfer per row; per-scalar .item() calls would
        # each sync the GPU and dominate the request latency.
        values, indices = top.values.cpu().tolist(), top.indices.cpu().tolist()
        cells = []
        for p in range(T):
            entries = [
                {"token": vocab.id_to_token[i], "prob": v}
                for v, i in zip(values[p], indices[p], strict=True)
            ]
            # Lens at position p, horizon k approximates the logits at
            # position p+k, which predict the token at position p+k+1.
            predicted_pos = p + horizon + 1
            hit = predicted_pos < T and indices[p][0] == int(token_ids_cpu[predicted_pos])
            cells.append({"top": entries, "hit": hit, "bin": 0})
        return cells

    sections = []
    for block in blocks:
        rows = []
        if lens.blocks.get(block) == lens.blocks["h_top"]:
            rows.append({"k": 0, "cells": cells_from_logits(model_logits, horizon=0)})
        for k in horizons:
            rows.append({"k": k, "cells": cells_from_logits(lens_logits[(block, k)], k)})
        section = {"block": block, "rows": rows}
        _assign_percentile_bins(section)
        sections.append(section)

    return {
        "tokens": tokens,
        "sections": sections,
        "meta": {
            "model": lens.model_config,
            "fit": lens.fit_config,
        },
    }


def render_html(data: dict, standalone: bool = True) -> str:
    """Render grid data as a self-contained HTML page (no external assets)."""
    body = _render_body(data)
    if not standalone:
        return body
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Horizon lens — timestep × horizon grid</title>\n</head>\n<body>\n"
        f"{body}\n</body>\n</html>\n"
    )


def _render_body(data: dict) -> str:
    sections_html = "\n".join(_render_section(s, data["tokens"]) for s in data["sections"])
    legend_html = _render_legend()
    meta = data["meta"]
    meta_line = html.escape(
        f"d_model={meta['model']['d_model']} layers={meta['model']['num_layers']} · "
        f"fitted on {meta['fit']['num_sequences']} sequences × {meta['fit']['seq_len']} tokens"
    )
    payload = json.dumps(data["sections"])
    return f"""
<style>{_CSS}</style>
<div class="viz-root">
  <h1>Horizon lens — what the recurrent state is disposed to say</h1>
  <p class="sub">
    Column = sequence position (input token shown in the header row). Row <em>k</em> = the
    token that position's recurrent state is disposed to make the model emit
    <em>k</em> steps in the future, read through the corpus-averaged Jacobian transport
    (centered on the corpus-mean state, so cells show what is specific to this context).
    Row 0 is the model's actual next-token prediction. Shade = top-1 readout weight;
    a ring marks readouts confirmed by the token the text actually contains there.
  </p>
  <p class="meta">{meta_line}</p>
  {legend_html}
  {sections_html}
  <div id="tooltip" role="tooltip" hidden></div>
</div>
<script>const GRID_DATA = {payload};{_JS}</script>
"""


def _render_section(section: dict, tokens: list[str]) -> str:
    head_cells = "".join(
        f'<th scope="col"><span>{html.escape(t)}</span></th>' for t in tokens
    )
    rows_html = []
    for r, row in enumerate(section["rows"]):
        label = "k=0 (model)" if row["k"] == 0 else f"k={row['k']}"
        cells = "".join(
            _render_cell(cell, section["block"], r, c)
            for c, cell in enumerate(row["cells"])
        )
        rows_html.append(f'<tr><th scope="row">{label}</th>{cells}</tr>')
    return f"""
<section>
  <h2>block: <code>{html.escape(section["block"])}</code></h2>
  <div class="scroll">
    <table>
      <thead><tr><th scope="col" class="corner">token</th>{head_cells}</tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
  </div>
</section>
"""


def _render_cell(cell: dict, block: str, row: int, col: int) -> str:
    top = cell["top"][0]
    classes = f"cell b{cell['bin']}" + (" hit" if cell["hit"] else "")
    return (
        f'<td class="{classes}" data-block="{html.escape(block)}" '
        f'data-row="{row}" data-col="{col}">'
        f"<span>{html.escape(top['token'])}</span></td>"
    )


def render_app_shell(config: dict) -> str:
    """Render the interactive viewer page (served by ``jspace.server``).

    ``config`` carries lens limits and control defaults; the page fetches
    grids from ``POST /api/grid`` and renders them client-side with the same
    classes and CSS as the static export.
    """
    meta_line = html.escape(
        f"d_model={config['model']['d_model']} layers={config['model']['num_layers']} · "
        f"lens horizons 1..{config['max_horizon']} · device {config['device']}"
    )
    payload = json.dumps(config)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Horizon lens — live viewer</title>
</head>
<body>
<style>{_CSS}{_APP_CSS}</style>
<div class="viz-root">
  <h1>Horizon lens — live viewer</h1>
  <p class="sub">
    Type a prompt and read the model's recurrent state through the horizon lens.
    Column = sequence position. Row <em>k</em> = the token that position's state is
    disposed to make the model emit <em>k</em> steps in the future (row 0 = the model's
    actual next-token prediction). Shade = top-1 readout strength (percentile within
    grid); a ring marks readouts confirmed by the prompt's actual future token.
  </p>
  <p class="meta">{meta_line}</p>
  <div class="controls">
    <textarea id="prompt" rows="3" spellcheck="false"></textarea>
    <div class="controls-row">
      <label>horizon <input id="horizon" type="number" min="1"></label>
      <span id="blocks" class="blocks"></span>
      <label>min&nbsp;token&nbsp;count <input id="mincount" type="number" min="0"></label>
      <button id="read" type="button">Read workspace</button>
      <span id="status" class="meta" role="status"></span>
    </div>
  </div>
  {_render_legend()}
  <div id="sections"></div>
  <div id="tooltip" role="tooltip" hidden></div>
</div>
<script>const APP_CONFIG = {payload};{_APP_JS}</script>
</body>
</html>
"""


def _render_legend() -> str:
    swatches = "".join(
        f'<span class="swatch b{i}" title="≥ {10 * i}th percentile"></span>'
        for i in range(N_BINS)
    )
    return f"""
<div class="legend">
  <span class="legend-label">top-1 readout strength (percentile within grid)</span>
  <span class="legend-label">low</span>{swatches}<span class="legend-label">high</span>
  <span class="legend-gap"></span>
  <span class="swatch hit-demo"></span>
  <span class="legend-label">= confirmed by the actual future token</span>
</div>
"""


_LIGHT_VARS = "\n".join(
    f"  --b{i}: {c}; --fg{i}: {'#0b0b0b' if i < 6 else '#ffffff'};"
    for i, c in enumerate(SEQ_LIGHT)
)
_DARK_VARS = "\n".join(
    f"  --b{i}: {c}; --fg{i}: {'#ffffff' if i < 6 else '#0b0b0b'};"
    for i, c in enumerate(SEQ_DARK)
)
_CELL_RULES = "\n".join(
    f".b{i} {{ background: var(--b{i}); color: var(--fg{i}); }}" for i in range(10)
)

_CSS = f"""
.viz-root {{
  --surface: #fcfcfb; --page: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --ring: #0b0b0b;
{_LIGHT_VARS}
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--ink); background: var(--page);
  padding: 24px; max-width: 100%;
}}
@media (prefers-color-scheme: dark) {{
  .viz-root {{
    --surface: #1a1a19; --page: #0d0d0d;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --ring: #ffffff;
{_DARK_VARS}
  }}
}}
:root[data-theme="light"] .viz-root {{
  --surface: #fcfcfb; --page: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --grid: #e1e0d9; --ring: #0b0b0b;
{_LIGHT_VARS}
}}
:root[data-theme="dark"] .viz-root {{
  --surface: #1a1a19; --page: #0d0d0d;
  --ink: #ffffff; --ink-2: #c3c2b7; --grid: #2c2c2a; --ring: #ffffff;
{_DARK_VARS}
}}
.viz-root h1 {{ font-size: 18px; margin: 0 0 6px; }}
.viz-root h2 {{ font-size: 14px; margin: 20px 0 8px; color: var(--ink-2); }}
.viz-root .sub {{ color: var(--ink-2); font-size: 13px; max-width: 72ch; margin: 0 0 4px; }}
.viz-root .meta {{ color: var(--muted); font-size: 12px; margin: 0 0 12px; }}
.legend {{ display: flex; align-items: center; gap: 3px; font-size: 12px; color: var(--ink-2); }}
.legend .swatch {{ width: 16px; height: 12px; border-radius: 2px; display: inline-block; }}
.legend .hit-demo {{ background: transparent; box-shadow: inset 0 0 0 2px var(--ring); }}
.legend .legend-label {{ margin: 0 5px; }}
.legend .legend-gap {{ width: 18px; }}
.scroll {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--grid);
  border-radius: 8px; padding: 8px; }}
table {{ border-collapse: separate; border-spacing: 2px; }}
th {{ font-weight: 500; color: var(--muted); font-size: 11px; padding: 2px 4px;
  text-align: left; white-space: nowrap; }}
thead th span {{ display: inline-block; max-width: 72px; overflow: hidden;
  text-overflow: ellipsis; color: var(--ink); font-weight: 600; }}
tbody th {{ position: sticky; left: 0; background: var(--surface); }}
.cell {{ font-size: 11px; padding: 3px 5px; border-radius: 3px; white-space: nowrap;
  max-width: 72px; overflow: hidden; text-overflow: ellipsis; cursor: default; }}
.cell.hit {{ box-shadow: inset 0 0 0 2px var(--ring); }}
{_CELL_RULES}
#tooltip {{ position: fixed; z-index: 10; background: var(--surface); color: var(--ink);
  border: 1px solid var(--grid); border-radius: 6px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.18); pointer-events: none; min-width: 140px; }}
#tooltip table {{ border-spacing: 0; }}
#tooltip td {{ padding: 1px 6px 1px 0; }}
#tooltip .p {{ font-variant-numeric: tabular-nums; color: var(--ink-2); text-align: right; }}
"""

_APP_CSS = """
.controls { display: flex; flex-direction: column; gap: 8px; margin: 0 0 14px; }
.controls textarea {
  font: 13px/1.5 ui-monospace, Consolas, monospace; color: var(--ink);
  background: var(--surface); border: 1px solid var(--grid); border-radius: 8px;
  padding: 10px 12px; max-width: 72ch; resize: vertical;
}
.controls-row { display: flex; flex-wrap: wrap; align-items: center; gap: 14px;
  font-size: 12px; color: var(--ink-2); }
.controls-row input[type="number"] {
  width: 56px; font: inherit; color: var(--ink); background: var(--surface);
  border: 1px solid var(--grid); border-radius: 4px; padding: 3px 6px;
}
.blocks { display: inline-flex; gap: 10px; }
.blocks label { display: inline-flex; align-items: center; gap: 4px; }
.blocks code { font-size: 11px; }
.controls button {
  font: 600 12px system-ui, sans-serif; color: #ffffff; background: #2a78d6;
  border: none; border-radius: 6px; padding: 7px 14px; cursor: pointer;
}
.controls button:hover { background: #1c5cab; }
.controls button:focus-visible, .controls textarea:focus-visible,
.controls input:focus-visible { outline: 2px solid #2a78d6; outline-offset: 1px; }
.controls button[disabled] { opacity: 0.6; cursor: wait; }
"""

_APP_JS = """
(() => {
  const $ = (id) => document.getElementById(id);
  const promptEl = $("prompt"), horizonEl = $("horizon"), minCountEl = $("mincount");
  const readBtn = $("read"), statusEl = $("status"), sectionsEl = $("sections");
  const tip = $("tooltip");
  let gridData = null;

  promptEl.value = APP_CONFIG.default_prompt;
  horizonEl.value = APP_CONFIG.default_horizon;
  horizonEl.max = APP_CONFIG.max_horizon;
  minCountEl.value = APP_CONFIG.default_min_token_count;
  for (const b of APP_CONFIG.blocks) {
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox"; box.value = b;
    box.checked = APP_CONFIG.default_blocks.includes(b);
    const name = document.createElement("code");
    name.textContent = b;
    label.append(box, name);
    $("blocks").append(label);
  }

  async function readWorkspace() {
    const blocks = [...document.querySelectorAll("#blocks input:checked")].map((b) => b.value);
    if (!blocks.length) { statusEl.textContent = "select at least one block"; return; }
    readBtn.disabled = true;
    statusEl.textContent = "reading\\u2026";
    const t0 = performance.now();
    try {
      const resp = await fetch("/api/grid", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: promptEl.value,
          blocks,
          max_horizon: +horizonEl.value,
          min_token_count: +minCountEl.value,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) { statusEl.textContent = data.error || "request failed"; return; }
      gridData = data;
      renderGrid(data);
      statusEl.textContent =
        `${data.tokens.length} positions \\u00d7 horizons 0..${horizonEl.value} ` +
        `in ${Math.round(performance.now() - t0)} ms`;
    } catch (err) {
      statusEl.textContent = "server unreachable: " + err.message;
    } finally {
      readBtn.disabled = false;
    }
  }

  function renderGrid(data) {
    sectionsEl.replaceChildren();
    data.sections.forEach((section, s) => {
      const h2 = document.createElement("h2");
      h2.append("block: ");
      const code = document.createElement("code");
      code.textContent = section.block;
      h2.append(code);
      const table = document.createElement("table");
      const headRow = document.createElement("tr");
      const corner = document.createElement("th");
      corner.scope = "col"; corner.textContent = "token";
      headRow.append(corner);
      for (const t of data.tokens) {
        const th = document.createElement("th");
        th.scope = "col";
        const span = document.createElement("span");
        span.textContent = t;
        th.append(span);
        headRow.append(th);
      }
      table.createTHead().append(headRow);
      const tbody = table.createTBody();
      section.rows.forEach((row, r) => {
        const tr = document.createElement("tr");
        const th = document.createElement("th");
        th.scope = "row";
        th.textContent = row.k === 0 ? "k=0 (model)" : "k=" + row.k;
        tr.append(th);
        row.cells.forEach((cell, c) => {
          const td = document.createElement("td");
          td.className = `cell b${cell.bin}` + (cell.hit ? " hit" : "");
          td.dataset.section = s; td.dataset.row = r; td.dataset.col = c;
          const span = document.createElement("span");
          span.textContent = cell.top[0].token;
          td.append(span);
          tr.append(td);
        });
        tbody.append(tr);
      });
      const scroll = document.createElement("div");
      scroll.className = "scroll";
      scroll.append(table);
      const sec = document.createElement("section");
      sec.append(h2, scroll);
      sectionsEl.append(sec);
    });
  }

  const fmt = (p) => (p >= 0.01 ? (100 * p).toFixed(1) + "%" : (100 * p).toPrecision(2) + "%");
  sectionsEl.addEventListener("mousemove", (e) => {
    const cell = e.target.closest(".cell");
    if (!cell || !gridData) { tip.hidden = true; return; }
    const data = gridData.sections[+cell.dataset.section]
      .rows[+cell.dataset.row].cells[+cell.dataset.col];
    const table = document.createElement("table");
    data.top.forEach((t, i) => {
      const tr = document.createElement("tr");
      const rank = document.createElement("td");
      rank.textContent = `${i + 1}.`;
      const tok = document.createElement("td");
      tok.textContent = t.token;
      const p = document.createElement("td");
      p.className = "p"; p.textContent = fmt(t.prob);
      tr.append(rank, tok, p);
      table.append(tr);
    });
    tip.replaceChildren(table);
    tip.hidden = false;
    const pad = 14;
    tip.style.left = Math.min(e.clientX + pad, window.innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = Math.min(e.clientY + pad, window.innerHeight - tip.offsetHeight - 8) + "px";
  });
  sectionsEl.addEventListener("mouseleave", () => (tip.hidden = true));

  readBtn.addEventListener("click", readWorkspace);
  promptEl.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") readWorkspace();
  });
  readWorkspace();
})();
"""

_JS = """
(() => {
  const tip = document.getElementById("tooltip");
  const fmt = (p) => (p >= 0.01 ? (100 * p).toFixed(1) + "%" : (100 * p).toPrecision(2) + "%");
  document.querySelectorAll(".cell").forEach((cell) => {
    cell.addEventListener("mousemove", (e) => {
      const section = GRID_DATA.find((s) => s.block === cell.dataset.block);
      const data = section.rows[+cell.dataset.row].cells[+cell.dataset.col];
      tip.innerHTML =
        "<table>" +
        data.top
          .map(
            (t, i) =>
              `<tr><td>${i + 1}.</td><td>${t.token
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")}</td><td class="p">${fmt(t.prob)}</td></tr>`
          )
          .join("") +
        "</table>";
      tip.hidden = false;
      const pad = 14;
      const w = tip.offsetWidth, h = tip.offsetHeight;
      tip.style.left = Math.min(e.clientX + pad, window.innerWidth - w - 8) + "px";
      tip.style.top = Math.min(e.clientY + pad, window.innerHeight - h - 8) + "px";
    });
    cell.addEventListener("mouseleave", () => (tip.hidden = true));
  });
})();
"""
