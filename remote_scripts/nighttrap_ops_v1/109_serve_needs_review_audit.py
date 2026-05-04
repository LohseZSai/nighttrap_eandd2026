#!/usr/bin/env python3
"""Serve a blind needs-review plausibility audit UI.

The UI hides benchmark labels and asks the annotator to assign the binary
routine/review decision from event frames plus context-lite fields.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


ALLOWED_IMAGE_ROOTS = [Path("<REMOTE_DATA_ROOT>/dataset").resolve()]
MANUAL_LABELS = ["", "routine", "review", "uncertain"]
SUPPORT_OPTIONS = ["", "yes", "no", "uncertain"]
ERROR_TYPES = [
    "",
    "no_error",
    "ambiguous_evidence",
    "context_insufficient",
    "image_unusable",
    "label_too_strong",
    "label_too_weak",
    "other",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp.replace(path)


def is_allowed_image(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except FileNotFoundError:
        resolved = path
    return any(str(resolved).startswith(str(root) + os.sep) for root in ALLOWED_IMAGE_ROOTS)


def option_html(options: list[str], selected: str) -> str:
    return "".join(
        f'<option value="{html.escape(option)}" {"selected" if option == selected else ""}>'
        f'{html.escape(option or "select")}</option>'
        for option in options
    )


def context_html(raw: str) -> str:
    if not raw:
        return "<p class=\"muted\">No context-lite fields.</p>"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return f"<p>{html.escape(raw)}</p>"
    rows = []
    for key in [
        "season",
        "site_common_species_names",
        "top_k_source",
        "used_sqlite_fallback",
        "exposes_current_species",
        "exposes_counts",
        "exposes_novelty_or_rarity",
    ]:
        if key not in data:
            continue
        val = data[key]
        if isinstance(val, list):
            val = ", ".join(str(x) for x in val) if val else "none"
        rows.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(val))}</td></tr>")
    return "<table class=\"context\">" + "".join(rows) + "</table>" if rows else "<p class=\"muted\">No context-lite fields.</p>"


def optional_model_html(row: dict[str, str]) -> str:
    keys = [
        ("model_name", "model"),
        ("model_setting", "setting"),
        ("model_pred_label", "model prediction"),
        ("model_needs_review_score", "needs-review score"),
        ("model_rank", "model rank"),
        ("model_choice_probs", "choice probabilities"),
    ]
    rows = []
    for key, label in keys:
        val = row.get(key, "")
        if val == "":
            continue
        rows.append(f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(val))}</td></tr>")
    if not rows:
        return ""
    return (
        '<div class="modelbox"><b>Model suggestion shown for audit</b>'
        '<p class="muted">This is the model output for the event. The benchmark label remains hidden.</p>'
        '<table class="context">'
        + "".join(rows)
        + "</table></div>"
    )


class AuditState:
    def __init__(self, batch_csv: Path, out_csv: Path):
        self.batch_csv = batch_csv
        self.out_csv = out_csv
        self.rows = read_csv(batch_csv)
        self.fieldnames = list(self.rows[0].keys()) if self.rows else []
        for field in ["manual_label", "manual_binary_label", "image_support", "context_support", "error_type", "notes"]:
            if field not in self.fieldnames:
                self.fieldnames.append(field)
        if out_csv.exists():
            previous = {row.get("sample_id", ""): row for row in read_csv(out_csv)}
            for row in self.rows:
                prev = previous.get(row.get("sample_id", ""))
                if not prev:
                    continue
                for field in ["manual_label", "manual_binary_label", "image_support", "context_support", "error_type", "notes"]:
                    row[field] = prev.get(field, row.get(field, ""))

    def save(self) -> None:
        write_csv(self.out_csv, self.rows, self.fieldnames)

    def completed_count(self) -> int:
        return sum(1 for row in self.rows if row.get("manual_label") in {"routine", "review", "uncertain"})


def make_handler(state: AuditState, page_size: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args):  # noqa: A003
            return

        def send_text(self, text: str, content_type: str = "text/html; charset=utf-8") -> None:
            data = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def redirect(self, target: str) -> None:
            self.send_response(303)
            self.send_header("Location", target)
            self.end_headers()

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/image":
                raw = unquote(parse_qs(parsed.query).get("path", [""])[0])
                path = Path(raw)
                if not raw or not is_allowed_image(path) or not path.exists():
                    self.send_response(404)
                    self.end_headers()
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path == "/export":
                state.save()
                self.send_text(state.out_csv.read_text(encoding="utf-8"), "text/csv; charset=utf-8")
                return
            if parsed.path == "/status":
                payload = {
                    "batch_csv": str(state.batch_csv),
                    "out_csv": str(state.out_csv),
                    "n": len(state.rows),
                    "completed": state.completed_count(),
                    "mode": "blind binary needs-review plausibility audit",
                }
                self.send_text(json.dumps(payload, indent=2), "application/json; charset=utf-8")
                return
            start = int(parse_qs(parsed.query).get("start", ["0"])[0] or 0)
            start = max(0, min(start, max(0, len(state.rows) - 1)))
            start = (start // page_size) * page_size
            self.send_text(render_page(start))

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/save_batch":
                self.send_response(404)
                self.end_headers()
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8")
            form = parse_qs(body)
            for raw_idx in form.get("idx", []):
                idx = int(raw_idx or 0)
                if not 0 <= idx < len(state.rows):
                    continue
                row = state.rows[idx]
                manual = form.get(f"manual_label_{idx}", [row.get("manual_label", "")])[0]
                row["manual_label"] = manual
                row["manual_binary_label"] = {"routine": "0", "review": "1", "uncertain": "uncertain"}.get(manual, "")
                row["image_support"] = form.get(f"image_support_{idx}", [row.get("image_support", "")])[0]
                row["context_support"] = form.get(f"context_support_{idx}", [row.get("context_support", "")])[0]
                row["error_type"] = form.get(f"error_type_{idx}", [row.get("error_type", "")])[0]
                row["notes"] = form.get(f"notes_{idx}", [row.get("notes", "")])[0]
            state.save()
            self.redirect(f"/?start={quote(form.get('next', ['0'])[0])}")

    def render_card(idx: int) -> str:
        row = state.rows[idx]
        image_paths = [
            ("first", row.get("image_path_first", "")),
            ("middle", row.get("image_path_middle", "")),
            ("last", row.get("image_path_last", "")),
        ]
        thumbs = "".join(
            f'<div class="thumb"><div class="thumblabel">{label}</div>'
            f'<img src="/image?path={quote(path)}" alt="{label} frame"></div>'
            for label, path in image_paths
        )
        return f"""
    <section class="card">
      <div class="cardtop">
        <h3>Event {idx + 1}</h3>
        <span class="pill">{html.escape(row.get("dataset", ""))} | {html.escape(row.get("source_set", ""))}</span>
      </div>
      <div class="frames">{thumbs}</div>
      <table class="meta">
        <tr><th>event_id</th><td>{html.escape(row.get("event_id", ""))}</td></tr>
        <tr><th>site_id</th><td>{html.escape(row.get("site_id", ""))}</td></tr>
        <tr><th>source species label</th><td>{html.escape(row.get("species_label_audit_only", "not available"))}</td></tr>
        <tr><th>source count label</th><td>{html.escape(row.get("count_label_audit_only", "not available"))}</td></tr>
        <tr><th>species in site common list</th><td>{html.escape(row.get("species_in_site_common_species_audit_only", "unknown"))}</td></tr>
      </table>
      {optional_model_html(row)}
      <div class="contextbox"><b>Allowed context-lite fields</b>{context_html(row.get("context_lite_json", ""))}</div>
      <input type="hidden" name="idx" value="{idx}">
      <label>manual_label</label>
      <select name="manual_label_{idx}">{option_html(MANUAL_LABELS, row.get("manual_label", ""))}</select>
      <label>image_support</label>
      <select name="image_support_{idx}">{option_html(SUPPORT_OPTIONS, row.get("image_support", ""))}</select>
      <label>context_support</label>
      <select name="context_support_{idx}">{option_html(SUPPORT_OPTIONS, row.get("context_support", ""))}</select>
      <label>error_type</label>
      <select name="error_type_{idx}">{option_html(ERROR_TYPES, row.get("error_type", ""))}</select>
      <label>notes</label>
      <textarea name="notes_{idx}">{html.escape(row.get("notes", ""))}</textarea>
    </section>"""

    def render_page(start: int) -> str:
        end = min(len(state.rows), start + page_size)
        prev_start = max(0, start - page_size)
        next_start = min(max(0, len(state.rows) - page_size), start + page_size)
        cards = "".join(render_card(i) for i in range(start, end))
        completed = state.completed_count()
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>NightTrap Needs-review Audit</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 18px; color: #111827; background: #f8fafc; }}
.wrap {{ max-width: 1500px; margin: 0 auto; }}
.top {{ display:flex; justify-content:space-between; align-items:center; gap:12px; }}
.progress {{ color:#374151; font-size:14px; }}
.hint {{ background:#eef6ff; border:1px solid #bdd7ff; padding:10px; border-radius:6px; margin:10px 0; font-size:14px; }}
.grid {{ display:grid; grid-template-columns: 1fr; gap:14px; margin-top:12px; }}
.card {{ background:#fff; border:1px solid #d9dee7; border-radius:8px; padding:12px; }}
.cardtop {{ display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:8px; }}
h3 {{ margin:0; font-size:16px; }}
.pill {{ border:1px solid #d9dee7; border-radius:999px; padding:3px 8px; font-size:12px; color:#374151; background:#f8fafc; }}
.frames {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:8px; background:#111; padding:8px; border-radius:8px; }}
.thumblabel {{ color:#d1d5db; font-size:12px; margin-bottom:4px; }}
img {{ display:block; width:100%; max-height:260px; object-fit:contain; background:#000; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }}
th {{ width:155px; text-align:left; vertical-align:top; color:#374151; padding:5px 6px; border-bottom:1px solid #edf0f5; }}
td {{ word-break:break-word; padding:5px 6px; border-bottom:1px solid #edf0f5; }}
.contextbox {{ margin-top:9px; padding:8px; border:1px solid #edf0f5; border-radius:6px; background:#fbfdff; }}
.modelbox {{ margin-top:9px; padding:8px; border:1px solid #e4d191; border-radius:6px; background:#fff9e6; }}
label {{ display:block; font-weight:700; margin-top:10px; }}
select, textarea {{ width:100%; box-sizing:border-box; margin-top:5px; font-size:15px; padding:8px; }}
textarea {{ min-height:64px; }}
.buttons {{ display:flex; gap:8px; margin-top:14px; flex-wrap:wrap; }}
button, a.btn {{ border:1px solid #0b2a6f; background:#0b2a6f; color:#fff; padding:9px 12px; border-radius:6px; text-decoration:none; font-size:14px; cursor:pointer; }}
a.secondary {{ background:#fff; color:#0b2a6f; }}
.muted {{ color:#6b7280; margin:4px 0 0; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <h2>NightTrap Needs-review Plausibility Audit: events {start + 1}-{end} / {len(state.rows)}</h2>
    <div class="progress">completed: {completed} / {len(state.rows)} | <a href="/export">export CSV</a> | <a href="/status">status</a></div>
  </div>
  <div class="hint">
    Label-aware plausibility audit: the benchmark routine/review label is hidden, but source animal labels are shown because this
    is a rulebook plausibility check rather than an ecological identification test. Choose <b>routine</b> if the event can be deprioritized under the rulebook;
    choose <b>review</b> if identity, count, or event interpretation should enter the human-review queue; choose <b>uncertain</b>
    if visual/context evidence is insufficient for a stable decision.
  </div>
  <form method="post" action="/save_batch">
    <div class="buttons">
      <button type="submit" name="next" value="{next_start}">Save this page and next 10</button>
      <button type="submit" name="next" value="{start}">Save this page</button>
      <a class="btn secondary" href="/?start={prev_start}">Previous 10</a>
      <a class="btn secondary" href="/?start={next_start}">Next 10 without saving</a>
    </div>
    <div class="grid">{cards}</div>
    <div class="buttons">
      <button type="submit" name="next" value="{next_start}">Save this page and next 10</button>
      <button type="submit" name="next" value="{start}">Save this page</button>
      <a class="btn secondary" href="/?start={prev_start}">Previous 10</a>
      <a class="btn secondary" href="/?start={next_start}">Next 10 without saving</a>
    </div>
  </form>
</div>
</body>
</html>"""

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-csv", required=True)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--page-size", type=int, default=10)
    args = parser.parse_args()

    batch_csv = Path(args.batch_csv).resolve()
    out_csv = Path(args.out_csv).resolve() if args.out_csv else batch_csv.with_name(batch_csv.stem + "_completed.csv")
    state = AuditState(batch_csv, out_csv)
    state.save()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, max(1, args.page_size)))
    print(f"Serving {batch_csv} on http://{args.host}:{args.port}")
    print(f"Writing annotations to {out_csv}")
    server.serve_forever()


if __name__ == "__main__":
    main()
