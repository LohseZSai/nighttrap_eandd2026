from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results/nighttrap_tables/table_needs_review_dense_v09.tex"
OUT_BASE = ROOT / "figures/nighttrap_task5_core_performance"

COLORS = {
    "auroc": "#9BD3E5",
    "auprc": "#BFE5F1",
    "macro_f1": "#E78686",
    "p50": "#8FCCDF",
    "p100": "#D8EEF5",
    "ndcg": "#F4D982",
    "auto_pass": "#F7E6A4",
    "grid": "#B8B8B8",
    "text": "#171717",
    "muted": "#666666",
}


def clean_tex(value: str) -> str:
    value = value.replace("\\rowcolor{NightTrapGray}", "")
    value = value.replace("\\missingmetric", "")
    value = value.replace("\\%", "")
    value = value.replace("$\\times$", "x")
    value = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", value)
    value = re.sub(r"\\[a-zA-Z]+\{[^{}]*\}", "", value)
    return value.strip()


def parse_float(value: str) -> float | None:
    value = clean_tex(value)
    if not value:
        return None
    value = value.replace("x", "").replace("\\", "").strip()
    try:
        return float(value)
    except ValueError:
        return None


def parse_dense_table() -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    text = TABLE.read_text(encoding="utf-8")
    queue_rows: list[dict[str, float | str]] = []
    hard_rows: list[dict[str, float | str]] = []

    for raw in text.splitlines():
        if "&" not in raw or "\\\\" not in raw:
            continue
        line = clean_tex(raw).rstrip("\\").strip()
        if not line or line.startswith("Split ") or line.startswith("\\"):
            continue
        cells = [clean_tex(c) for c in line.split("&")]

        if len(cells) == 12 and cells[0] in {"6000 routine-stress", "913 CLIP split"}:
            queue_rows.append(
                {
                    "split": cells[0],
                    "model": cells[1],
                    "n": parse_float(cells[2]),
                    "base": parse_float(cells[3]),
                    "auroc": parse_float(cells[4]),
                    "auprc": parse_float(cells[5]),
                    "macro_f1": parse_float(cells[6]),
                    "p50": parse_float(cells[7]),
                    "p100": parse_float(cells[8]),
                    "enrich50": parse_float(cells[9]),
                    "ndcg": parse_float(cells[10]),
                    "auto_pass": parse_float(cells[11]),
                }
            )
        elif len(cells) == 6 and cells[0] == "983 hardset":
            hard_rows.append(
                {
                    "split": cells[0],
                    "model": cells[1],
                    "n": parse_float(cells[2]),
                    "macro_f1": parse_float(cells[3]),
                    "p50": parse_float(cells[4]),
                    "ndcg": parse_float(cells[5]),
                }
            )

    return queue_rows, hard_rows


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    raw = hex_color.lstrip("#")
    return tuple(int(raw[i : i + 2], 16) for i in (0, 2, 4))


def short_model(name: str) -> str:
    replacements = {
        "Qwen3-VL-8B, full context-lite": "Qwen3-VL-8B",
        "CLIP/context-lite logistic ranker": "CLIP/context logistic\n913 CLIP split",
        "XGBoost image-only ranker": "XGBoost image-only\n913 CLIP split",
        "LightGBM image-only ranker": "LightGBM image-only\n913 CLIP split",
        "XGBoost clean-context-lite ranker": "XGBoost context-only\n913 CLIP split",
        "XGBoost CLIP+clean-context ranker": "XGBoost CLIP+context\n913 CLIP split",
        "LightGBM clean-context-lite ranker": "LightGBM context-only\n913 CLIP split",
        "LightGBM CLIP+clean-context ranker": "LightGBM CLIP+context\n913 CLIP split",
        "Rule-only, context-lite": "Rule-only\n983 hardset",
        "Qwen2-VL-2B, full context-lite": "Qwen2-VL-2B\n983 hardset",
        "NVILA-8B, full context-lite": "NVILA-8B\n983 hardset",
        "GPT-5.4 reference, full context-lite": "GPT-5.4 reference\n983 hardset",
    }
    return replacements.get(name, name)


def row_label(row: dict[str, float | str]) -> str:
    name = short_model(str(row["model"]))
    if "\n" in name:
        return name
    split = str(row.get("split", ""))
    if split == "6000 routine-stress":
        return f"{name}\n6000 routine-stress"
    if split == "913 CLIP split":
        return f"{name}\n913 CLIP split"
    if split == "983 hardset":
        return f"{name}\n983 hardset"
    return name


class PdfPainter:
    def __init__(self, path: Path, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.c = canvas.Canvas(str(path), pagesize=(width, height))

    def color(self, value: str) -> HexColor:
        return HexColor(value)

    def line(self, x1, y1, x2, y2, color="#000000", width=1.0, dash=None) -> None:
        self.c.setStrokeColor(self.color(color))
        self.c.setLineWidth(width)
        if dash:
            self.c.setDash(dash)
        else:
            self.c.setDash()
        self.c.line(x1, self.height - y1, x2, self.height - y2)
        self.c.setDash()

    def rect(self, x, y, w, h, fill, stroke="#333333", width=0.45) -> None:
        self.c.setFillColor(self.color(fill))
        self.c.setStrokeColor(self.color(stroke))
        self.c.setLineWidth(width)
        self.c.rect(x, self.height - y - h, w, h, stroke=1, fill=1)

    def text(self, x, y, value, size=8, color="#000000", bold=False, anchor="left") -> None:
        self.c.setFillColor(self.color(color))
        self.c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        yy = self.height - y - size * 0.75
        if anchor == "center":
            self.c.drawCentredString(x, yy, value)
        elif anchor == "right":
            self.c.drawRightString(x, yy, value)
        else:
            self.c.drawString(x, yy, value)

    def save(self) -> None:
        self.c.save()


class PngPainter:
    def __init__(self, path: Path, width: int, height: int, scale: int = 3) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.scale = scale
        self.img = Image.new("RGB", (width * scale, height * scale), "white")
        self.d = ImageDraw.Draw(self.img)
        self.font_regular = self._font(False)
        self.font_bold = self._font(True)

    def _font(self, bold: bool, size: int = 8) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size * self.scale)
            except Exception:
                continue
        return ImageFont.load_default()

    def _xy(self, x, y) -> tuple[int, int]:
        return int(round(x * self.scale)), int(round(y * self.scale))

    def line(self, x1, y1, x2, y2, color="#000000", width=1.0, dash=None) -> None:
        xy = (*self._xy(x1, y1), *self._xy(x2, y2))
        if dash:
            seg = 4 * self.scale
            gap = 3 * self.scale
            total = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 * self.scale
            if total == 0:
                return
            dx = (x2 - x1) * self.scale / total
            dy = (y2 - y1) * self.scale / total
            start = 0
            while start < total:
                end = min(start + seg, total)
                sx = x1 * self.scale + dx * start
                sy = y1 * self.scale + dy * start
                ex = x1 * self.scale + dx * end
                ey = y1 * self.scale + dy * end
                self.d.line((sx, sy, ex, ey), fill=hex_to_rgb(color), width=max(1, int(width * self.scale)))
                start += seg + gap
        else:
            self.d.line(xy, fill=hex_to_rgb(color), width=max(1, int(width * self.scale)))

    def rect(self, x, y, w, h, fill, stroke="#333333", width=0.45) -> None:
        x1, y1 = self._xy(x, y)
        x2, y2 = self._xy(x + w, y + h)
        self.d.rectangle((x1, y1, x2, y2), fill=hex_to_rgb(fill), outline=hex_to_rgb(stroke), width=max(1, int(width * self.scale)))

    def text(self, x, y, value, size=8, color="#000000", bold=False, anchor="left") -> None:
        font = self._font(bold, size)
        sx, sy = self._xy(x, y)
        bbox = self.d.textbbox((sx, sy), value, font=font)
        tw = bbox[2] - bbox[0]
        if anchor == "center":
            sx -= tw // 2
        elif anchor == "right":
            sx -= tw
        self.d.text((sx, sy), value, fill=hex_to_rgb(color), font=font)

    def save(self) -> None:
        self.img.save(self.path)


def draw_label(p, x: float, y: float, label: str, size=7.4) -> None:
    parts = label.split("\n")
    p.text(x, y - 7, parts[0], size=size, color=COLORS["text"], bold=True, anchor="right")
    if len(parts) > 1:
        p.text(x, y + 3, parts[1], size=size - 0.8, color=COLORS["muted"], bold=False, anchor="right")


def draw_grouped_horizontal(p, x, y, plot_w, rows, metrics, title, xlim, row_gap=30) -> None:
    p.text(x, y - 35, title, size=10.5, color=COLORS["text"], bold=True)
    p.line(x, y - 9, x + plot_w, y - 9, color="#222222", width=0.8)
    p.line(x, y - 9, x, y + row_gap * len(rows) - 10, color="#222222", width=0.8)

    for tick in range(0, int(xlim[1]) + 1, 20):
        tx = x + plot_w * (tick - xlim[0]) / (xlim[1] - xlim[0])
        p.line(tx, y - 9, tx, y + row_gap * len(rows) - 10, color=COLORS["grid"], width=0.6, dash=(3, 3))
        p.text(tx, y + row_gap * len(rows) - 4, str(tick), size=6.8, color=COLORS["muted"], anchor="center")

    bar_h = 4.2 if len(metrics) >= 4 else 5.0
    offsets = [(-1.5 + i) * (bar_h + 1.7) for i in range(len(metrics))] if len(metrics) == 4 else [(-1 + i) * (bar_h + 2.0) for i in range(len(metrics))]

    for row_idx, row in enumerate(rows):
        yy = y + row_idx * row_gap
        draw_label(p, x - 9, yy, row_label(row))
        for offset, (key, _label, color) in zip(offsets, metrics):
            val = row.get(key)
            if val is None:
                continue
            bar_w = plot_w * float(val) / (xlim[1] - xlim[0])
            p.rect(x, yy + offset, bar_w, bar_h, fill=color)
            if float(val) >= xlim[1] * 0.10:
                p.text(x + bar_w + 3, yy + offset - 1.2, f"{float(val):.1f}", size=6.5, color=COLORS["muted"])

    p.text(x + plot_w / 2, y + row_gap * len(rows) + 13, "Score (%)", size=7.6, color=COLORS["text"], anchor="center")


def draw_legend(p, items, x, y) -> None:
    cursor = x
    for label, color in items:
        p.rect(cursor, y, 10, 7, fill=color, stroke="#333333", width=0.45)
        p.text(cursor + 14, y - 0.5, label, size=7.2, color=COLORS["text"])
        cursor += 14 + len(label) * 4.7 + 10


def draw_figure(p, queue_rows, hard_rows) -> None:
    p.text(390, 20, "Task 5 needs-review performance across measured models", size=13, color=COLORS["text"], bold=True, anchor="center")
    draw_legend(
        p,
        [
            ("Macro-F1", COLORS["macro_f1"]),
            ("P@50", COLORS["p50"]),
            ("P@100", COLORS["p100"]),
            ("NDCG@50/100", COLORS["ndcg"]),
            ("Auto-pass@95R", COLORS["auto_pass"]),
        ],
        180,
        43,
    )
    draw_grouped_horizontal(
        p,
        150,
        92,
        265,
        queue_rows,
        [
            ("macro_f1", "Macro-F1", COLORS["macro_f1"]),
            ("p50", "P@50", COLORS["p50"]),
            ("p100", "P@100", COLORS["p100"]),
            ("auto_pass", "Auto-pass@95R", COLORS["auto_pass"]),
        ],
        "A. Operational queue metrics",
        (0, 72),
        row_gap=31,
    )
    draw_grouped_horizontal(
        p,
        565,
        92,
        170,
        hard_rows,
        [
            ("macro_f1", "Macro-F1", COLORS["macro_f1"]),
            ("p50", "P@50", COLORS["p50"]),
            ("ndcg", "NDCG@50", COLORS["ndcg"]),
        ],
        "B. 983-event hardset check",
        (0, 86),
        row_gap=31,
    )
    p.text(
        390,
        366,
        "Panel A combines the 6000-event routine-stress row with 913-event supervised ranker checks; Panel B is a separate hardset diagnostic. Source: table_needs_review_dense_v09.tex.",
        size=6.9,
        color=COLORS["muted"],
        anchor="center",
    )


def main() -> None:
    queue_rows, hard_rows = parse_dense_table()

    queue_order = [
        "Qwen3-VL-8B, full context-lite",
        "XGBoost image-only ranker",
        "LightGBM image-only ranker",
        "XGBoost CLIP+clean-context ranker",
        "LightGBM CLIP+clean-context ranker",
        "CLIP/context-lite logistic ranker",
        "XGBoost clean-context-lite ranker",
        "LightGBM clean-context-lite ranker",
    ]
    hard_order = [
        "Rule-only, context-lite",
        "Qwen2-VL-2B, full context-lite",
        "Qwen3-VL-8B, full context-lite",
        "NVILA-8B, full context-lite",
        "GPT-5.4 reference, full context-lite",
    ]

    queue_by_name = {str(row["model"]): row for row in queue_rows}
    hard_by_name = {str(row["model"]): row for row in hard_rows}
    queue_rows = [queue_by_name[name] for name in queue_order if name in queue_by_name]
    hard_rows = [hard_by_name[name] for name in hard_order if name in hard_by_name]

    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    pdf = PdfPainter(OUT_BASE.with_suffix(".pdf"), 780, 382)
    draw_figure(pdf, queue_rows, hard_rows)
    pdf.save()

    png = PngPainter(OUT_BASE.with_suffix(".png"), 780, 382, scale=3)
    draw_figure(png, queue_rows, hard_rows)
    png.save()

    print(f"Wrote {OUT_BASE.with_suffix('.pdf')}")
    print(f"Wrote {OUT_BASE.with_suffix('.png')}")


if __name__ == "__main__":
    main()
