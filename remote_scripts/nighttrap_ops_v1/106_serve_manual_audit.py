#!/usr/bin/env python3
"""Serve a small local web UI for NightTrap manual audit batches.

The server is intentionally dependency-free and only serves images under
allowed source-data roots. It writes completed annotations to a CSV file next
to the batch file.
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
MANUAL_LABELS = ["", "empty", "nonempty", "uncertain"]
ERROR_TYPES = [
    "",
    "no_error",
    "animal_visible",
    "no_animal_visible",
    "uncertain_low_quality",
    "partial_or_occluded",
    "overexposed_or_near_black",
    "image_missing",
    "other",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(path)


def is_allowed_image(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except FileNotFoundError:
        resolved = path
    return any(str(resolved).startswith(str(root) + os.sep) for root in ALLOWED_IMAGE_ROOTS)


def option_html(options: list[str], selected: str) -> str:
    return "".join(
        f'<option value="{html.escape(o)}" {"selected" if o == selected else ""}>{html.escape(o or "select")}</option>'
        for o in options
    )


class AuditState:
    def __init__(self, batch_csv: Path, out_csv: Path):
        self.batch_csv = batch_csv
        self.out_csv = out_csv
        self.rows = read_csv(batch_csv)
        self.fieldnames = list(self.rows[0].keys()) if self.rows else []
        for f in ["manual_label", "error_type", "notes"]:
            if f not in self.fieldnames:
                self.fieldnames.append(f)
        if out_csv.exists():
            previous = {r.get("event_id", ""): r for r in read_csv(out_csv)}
            for row in self.rows:
                if row.get("event_id", "") in previous:
                    prev = previous[row["event_id"]]
                    row["manual_label"] = prev.get("manual_label", row.get("manual_label", ""))
                    row["error_type"] = prev.get("error_type", row.get("error_type", ""))
                    row["notes"] = prev.get("notes", row.get("notes", ""))

    def save(self) -> None:
        write_csv(self.out_csv, self.rows, self.fieldnames)

    def completed_count(self) -> int:
        return sum(1 for r in self.rows if r.get("manual_label") in {"empty", "nonempty", "uncertain"})


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
                params = parse_qs(parsed.query)
                raw = unquote(params.get("path", [""])[0])
                path = Path(raw)
                if not raw or not is_allowed_image(path) or not path.exists():
                    self.send_response(404)
                    self.end_headers()
                    return
                ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
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
                }
                self.send_text(json.dumps(payload, indent=2), "application/json; charset=utf-8")
                return
            params = parse_qs(parsed.query)
            if "i" in params:
                idx = int(params.get("i", ["0"])[0] or 0)
                start = (max(0, idx) // page_size) * page_size
            else:
                start = int(params.get("start", ["0"])[0] or 0)
            start = max(0, min(start, max(0, len(state.rows) - 1)))
            start = (start // page_size) * page_size
            self.send_text(render_page(start))

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path not in {"/save", "/save_batch"}:
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            form = parse_qs(body)
            if parsed.path == "/save_batch":
                for idx_raw in form.get("idx", []):
                    idx = int(idx_raw or 0)
                    if 0 <= idx < len(state.rows):
                        row = state.rows[idx]
                        row["manual_label"] = form.get(f"manual_label_{idx}", [row.get("manual_label", "")])[0]
                        row["error_type"] = form.get(f"error_type_{idx}", [row.get("error_type", "")])[0]
                        row["notes"] = form.get(f"notes_{idx}", [row.get("notes", "")])[0]
                state.save()
                nxt = form.get("next", ["0"])[0]
                self.redirect(f"/?start={quote(nxt)}")
                return
            idx = int(form.get("idx", ["0"])[0] or 0)
            if 0 <= idx < len(state.rows):
                row = state.rows[idx]
                row["manual_label"] = form.get("manual_label", [""])[0]
                row["error_type"] = form.get("error_type", [""])[0]
                row["notes"] = form.get("notes", [""])[0]
                state.save()
            nxt = form.get("next", [str(idx + 1)])[0]
            self.redirect(f"/?i={quote(nxt)}")

    def render_card(idx: int) -> str:
        row = state.rows[idx]
        image_path = row.get("image_path", "")
        image_url = "/image?path=" + quote(image_path)
        label_options = option_html(MANUAL_LABELS, row.get("manual_label", ""))
        error_options = option_html(ERROR_TYPES, row.get("error_type", ""))
        notes = html.escape(row.get("notes", ""))
        meta_items = [
            ("event_id", row.get("event_id", "")),
            ("source", row.get("source", "")),
            ("mode", row.get("mode", "")),
            ("pred_label", row.get("pred_label", "")),
            ("detector_score", row.get("detector_score", "")),
            ("confidence_bucket", row.get("confidence_bucket", "")),
            ("audit_gold_label", row.get("gold_label", "")),
            ("image_path", image_path),
        ]
        meta_html = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in meta_items
        )
        return f"""
    <section class="card">
      <div class="cardtop">
        <h3>Item {idx + 1}</h3>
        <span class="pill">{html.escape(row.get("pred_label", ""))}</span>
      </div>
      <div class="imagebox"><img src="{image_url}" alt="camera trap image"></div>
      <table>{meta_html}</table>
      <input type="hidden" name="idx" value="{idx}">
      <label>manual_label</label>
      <select name="manual_label_{idx}">{label_options}</select>
      <label>error_type</label>
      <select name="error_type_{idx}">{error_options}</select>
      <label>notes</label>
      <textarea name="notes_{idx}">{notes}</textarea>
    </section>"""

    def render_page(start: int) -> str:
        end = min(len(state.rows), start + page_size)
        prev_start = max(0, start - page_size)
        next_start = min(max(0, len(state.rows) - page_size), start + page_size)
        completed = state.completed_count()
        cards_html = "".join(render_card(i) for i in range(start, end))
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>NightTrap Audit</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 18px; color: #111827; background: #f8fafc; }}
.wrap {{ max-width: 1440px; margin: 0 auto; }}
.top {{ display:flex; justify-content:space-between; align-items:center; gap:12px; }}
.progress {{ color:#374151; font-size:14px; }}
.grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:14px; margin-top:12px; }}
.card {{ background:#fff; border:1px solid #d9dee7; border-radius:8px; padding:12px; }}
.cardtop {{ display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:8px; }}
h3 {{ margin:0; font-size:16px; }}
.pill {{ border:1px solid #d9dee7; border-radius:999px; padding:3px 8px; font-size:12px; color:#374151; background:#f8fafc; }}
.imagebox {{ background:#111; border-radius:8px; padding:6px; }}
img {{ display:block; width:100%; max-height:300px; margin:0 auto; object-fit:contain; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ width:125px; text-align:left; vertical-align:top; color:#374151; padding:5px 6px; border-bottom:1px solid #edf0f5; }}
td {{ word-break:break-all; padding:5px 6px; border-bottom:1px solid #edf0f5; }}
label {{ display:block; font-weight:700; margin-top:12px; }}
select, textarea {{ width:100%; box-sizing:border-box; margin-top:5px; font-size:15px; padding:8px; }}
textarea {{ min-height:86px; }}
.buttons {{ display:flex; gap:8px; margin-top:14px; flex-wrap:wrap; }}
button, a.btn {{ border:1px solid #0b2a6f; background:#0b2a6f; color:#fff; padding:9px 12px; border-radius:6px; text-decoration:none; font-size:14px; cursor:pointer; }}
a.secondary {{ background:#fff; color:#0b2a6f; }}
.hint {{ background:#fff8e6; border:1px solid #f1d28a; padding:9px; border-radius:6px; margin-top:10px; font-size:13px; }}
@media (max-width: 980px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <h2>NightTrap MegaDetector Audit: items {start + 1}-{end} / {len(state.rows)}</h2>
    <div class="progress">completed: {completed} / {len(state.rows)} | <a href="/export">export CSV</a> | <a href="/status">status</a></div>
  </div>
  <div class="hint">
    Label from visual evidence. <b>empty</b>: no animal or valid animal subject is visible.
    <b>nonempty</b>: animal or likely animal subject is visible.
    <b>uncertain</b>: not safely judgeable due to darkness, blur, occlusion, cropping, or missing image.
  </div>
  <form method="post" action="/save_batch">
    <div class="buttons">
      <button type="submit" name="next" value="{next_start}">Save this page and next 10</button>
      <button type="submit" name="next" value="{start}">Save this page</button>
      <a class="btn secondary" href="/?start={prev_start}">Previous 10</a>
      <a class="btn secondary" href="/?start={next_start}">Next 10 without saving</a>
    </div>
    <div class="grid">{cards_html}</div>
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
    parser.add_argument("--port", type=int, default=8765)
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
