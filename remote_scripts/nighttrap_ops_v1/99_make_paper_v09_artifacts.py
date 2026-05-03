#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "results/nighttrap_tables"
FIG_DIR = ROOT / "figures"


def pct(x: float | None) -> str:
    return "--" if x is None else f"{100.0 * x:.2f}"


def mf1_acc(f1: float | None, acc: float | None) -> str:
    if f1 is None or acc is None:
        return "--"
    return f"{pct(f1)} ({pct(acc)})"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Pdf:
    def __init__(self, path: Path, width: int, height: int) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.ops: list[str] = []

    def color(self, stroke: str | None = None, fill: str | None = None) -> None:
        def parts(hex_color: str) -> tuple[float, float, float]:
            h = hex_color.lstrip("#")
            return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
        if stroke:
            r, g, b = parts(stroke)
            self.ops.append(f"{r:.4f} {g:.4f} {b:.4f} RG")
        if fill:
            r, g, b = parts(fill)
            self.ops.append(f"{r:.4f} {g:.4f} {b:.4f} rg")

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = "#666666", width: float = 0.8) -> None:
        self.color(stroke=color)
        self.ops.append(f"{width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def poly(self, pts: list[tuple[float, float]], color: str, width: float = 1.3, close: bool = True) -> None:
        if not pts:
            return
        self.color(stroke=color)
        cmd = [f"{width:.2f} w {pts[0][0]:.2f} {pts[0][1]:.2f} m"]
        for x, y in pts[1:]:
            cmd.append(f"{x:.2f} {y:.2f} l")
        if close:
            cmd.append("h")
        cmd.append("S")
        self.ops.append(" ".join(cmd))

    def text(self, x: float, y: float, text: str, size: int = 9, color: str = "#111111", align: str = "left") -> None:
        self.color(fill=color)
        s = pdf_escape(text)
        # Approximate alignment using Helvetica width heuristic.
        offset = 0.0
        if align == "center":
            offset = -0.25 * size * len(text)
        elif align == "right":
            offset = -0.50 * size * len(text)
        self.ops.append(f"BT /F1 {size} Tf {x + offset:.2f} {y:.2f} Td ({s}) Tj ET")

    def save(self) -> None:
        stream = "\n".join(self.ops).encode("latin-1", errors="replace")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        ]
        out = bytearray(b"%PDF-1.4\n")
        offsets = []
        for i, obj in enumerate(objects, 1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
        for off in offsets:
            out += f"{off:010d} 00000 n \n".encode()
        out += f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        self.path.write_bytes(out)


def draw_radar(pdf: Pdf, cx: float, cy: float, radius: float, axes: list[str], series: dict[str, list[float]], title: str) -> None:
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e"]
    n = len(axes)
    angles = [math.pi / 2 - 2 * math.pi * i / n for i in range(n)]
    pdf.text(cx, cy + radius + 38, title, 12, align="center")
    for frac in [0.25, 0.50, 0.75, 1.00]:
        pts = [(cx + radius * frac * math.cos(a), cy + radius * frac * math.sin(a)) for a in angles]
        pdf.poly(pts, "#c8c8c8", 0.45)
        pdf.text(cx + 3, cy + radius * frac + 1, f"{int(frac * 100)}", 6, "#888888")
    for a, label in zip(angles, axes):
        x = cx + radius * math.cos(a)
        y = cy + radius * math.sin(a)
        pdf.line(cx, cy, x, y, "#bbbbbb", 0.45)
        lx = cx + (radius + 28) * math.cos(a)
        ly = cy + (radius + 22) * math.sin(a)
        align = "center"
        if math.cos(a) > 0.35:
            align = "left"
        elif math.cos(a) < -0.35:
            align = "right"
        pdf.text(lx, ly, label, 8, "#333333", align=align)
    for idx, (name, vals) in enumerate(series.items()):
        pts = [(cx + radius * max(0, min(v, 100)) / 100.0 * math.cos(a), cy + radius * max(0, min(v, 100)) / 100.0 * math.sin(a)) for v, a in zip(vals, angles)]
        pdf.poly(pts, colors[idx % len(colors)], 1.4)
    lx, ly = cx + radius + 55, cy + radius - 2
    for idx, name in enumerate(series):
        color = colors[idx % len(colors)]
        pdf.line(lx, ly - 14 * idx, lx + 14, ly - 14 * idx, color, 1.7)
        pdf.text(lx + 19, ly - 14 * idx - 3, name, 8, "#222222")


def make_tables() -> None:
    table1 = r"""\begin{table*}[t]
\centering
\caption{Performance across the four NightTrap review-workflow classification tasks. Each cell reports Macro-F1 (\%) with Accuracy (\%) in parentheses. Missing metrics are marked as --. Closed-source models are reference rows; open-source VLMs and camera-trap baselines are the main reproducible comparisons.}
\label{tab:workflow_main}
\small
\setlength{\tabcolsep}{4.2pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
Model & Image usability & Empty-event & Species & Count-bin \\
\midrule
\multicolumn{5}{l}{\textit{Closed-source reference}}\\
GPT-5.4 reference$^{\dagger}$ & 79.04 (79.13) & 81.68 (90.06) & 34.13 (88.79) & 64.48 (82.57) \\
\addlinespace[2pt]
\multicolumn{5}{l}{\textit{Open-source VLMs}}\\
Qwen2-VL-2B & 34.63 (40.22) & 62.88 (83.57) & 29.87 (83.72) & 42.29 (56.04) \\
Qwen3-VL-8B & \textbf{83.64} (84.17) & 75.24 (87.26) & \textbf{48.09} (88.56) & 58.66 (86.30) \\
\addlinespace[2pt]
\multicolumn{5}{l}{\textit{Camera-trap and representation baselines}}\\
Majority baseline & 37.69 (60.48) & 45.45 (83.31) & 0.79 (19.77) & 21.70 (76.68) \\
CLIP linear probe & 68.20 (69.92) & 75.25 (83.82) & 35.58 (65.30) & 49.06 (69.86) \\
MegaDetector threshold & -- & \textbf{100.00} (100.00) & -- & -- \\
\bottomrule
\end{tabular}
}
\vspace{0.25em}
\begin{minipage}{0.97\linewidth}
\footnotesize
$^{\dagger}$ The GPT-5.4 row uses the completed rerun with zero failed samples across the five tasks. MegaDetector is reported only for empty-event filtering because it is a detector-assisted screening baseline, not a full review-workflow model.
\end{minipage}
\end{table*}
"""
    table2 = r"""\begin{table*}[t]
\centering
\caption{Needs-review recommendation and the 983-event input/severity diagnostic. Panel A reports the realistic routine-stress setting where routine events dominate; CLIP/context is evaluated on its frozen 913-event split. Panel B reports the 983-event three-way severity check used to audit image/context settings, not the primary binary needs-review task. All values are percentages except enrichment.}
\label{tab:needs_review_main}
\small
\setlength{\tabcolsep}{3.5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrrr}
\toprule
Split & Model / input & N & Base & Macro-F1 & P@50 & P@100 & Enrich@50 & NDCG & Auto-pass@95R \\
\midrule
\multicolumn{10}{l}{\textit{Panel A: binary needs-review recommendation}}\\
6000 routine-stress & Random-score baseline & 6000 & 30.00 & -- & 30.00 & 30.00 & 1.00$\times$ & -- & -- \\
6000 routine-stress & Qwen3-VL-8B, full context-lite & 6000 & 30.00 & 38.91 & \textbf{66.00} & \textbf{64.00} & \textbf{2.20}$\times$ & 67.96 & 7.42 \\
913 CLIP split & CLIP/context-lite logistic ranker & 913 & 25.30 & 49.65 & 32.00 & 33.00 & 1.26$\times$ & 31.32 & 8.43 \\
\addlinespace[2pt]
\multicolumn{10}{l}{\textit{Panel B: 983-event three-way severity check}}\\
983 hardset & Rule-only, context-lite & 983 & -- & 23.09 & 26.00 & -- & -- & 56.00 & -- \\
983 hardset & Qwen2-VL-2B, full context-lite & 983 & -- & 32.97 & 54.00 & -- & -- & 63.81 & -- \\
983 hardset & Qwen3-VL-8B, full context-lite & 983 & -- & 28.87 & 66.00 & -- & -- & 76.18 & -- \\
983 hardset & NVILA-8B, full context-lite & 983 & -- & 31.96 & \textbf{78.00} & -- & -- & \textbf{77.89} & -- \\
983 hardset & GPT-5.4 reference, full context-lite & 983 & -- & \textbf{43.54} & 52.00 & -- & -- & 73.42 & -- \\
\bottomrule
\end{tabular}
}
\end{table*}
"""
    table3 = r"""\begin{table*}[t]
\centering
\caption{Ablation and diagnostic summary. The table reports the main conclusion from each v0.8 diagnostic while keeping the full detailed tables in the appendix. Needs-review rows use the binary \texttt{needs\_review} mapping unless explicitly marked as the 983-event severity check.}
\label{tab:ablation_diagnostics}
\small
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{p{0.19\linewidth}p{0.28\linewidth}p{0.23\linewidth}p{0.22\linewidth}}
\toprule
Diagnostic & Comparison & Main result & Takeaway \\
\midrule
Event representation & First frame vs. three-slot mean on CLIP needs-review split & P@50 is unchanged at 40.00; AUPRC is 30.98 vs. 30.74 & Three image slots do not automatically add useful signal because many events repeat the same frame path. \\
Context shortcut & Image-only, context-lite-only, full context-lite, shuffled context, site-frequency-only & Site-frequency-only reaches P@50=64.00, while context-lite-only has AUROC=60.70 but P@50=18.00 & Site history carries strong priors, so context gains must be treated as shortcut risk unless audited against image evidence. \\
Routine-stress deployment & Qwen3-VL-8B safe auto-pass under recall constraints & Auto-pass is 7.42 at 95\% recall and 3.35 at 98\% recall & Queue ordering is more mature than automatic pass-through; high-recall workload reduction remains small. \\
Imaging mode robustness & Night-color vs. night-IR groups & Qwen3 needs-review P@50 drops from 66.00 on night-color to 52.00 on night-IR & Night imaging mode changes the review recommendation problem and should be reported as a robustness check. \\
Source coverage & Source-wise breakdowns for classification and needs-review & Count-bin is Snapshot Serengeti-only; other tasks vary by source and site composition & Source heterogeneity is part of the benchmark rather than noise to hide. \\
MegaDetector-assisted empty pool & 300-item stratified audit manifest & Manual v0.8 FP/FN or precision/recall labels are not summarized & Detector-assisted construction is useful infrastructure, but claims about empty-pool reliability require independent audit. \\
\bottomrule
\end{tabular}
}
\end{table*}
"""
    write(TABLE_DIR / "table_workflow_task_groups_v09.tex", table1)
    write(TABLE_DIR / "table_needs_review_dense_v09.tex", table2)
    write(TABLE_DIR / "table_ablation_diagnostics_v09.tex", table3)


def make_radar_figures() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf1 = Pdf(FIG_DIR / "nighttrap_radar_task1_4.pdf", 420, 330)
    axes1 = ["Usability", "Empty", "Species", "Count-bin"]
    series1 = {
        "GPT-5.4": [79.04, 81.68, 34.13, 64.48],
        "Qwen3-VL-8B": [83.64, 75.24, 48.09, 58.66],
        "Qwen2-VL-2B": [34.63, 62.88, 29.87, 42.29],
        "CLIP probe": [68.20, 75.25, 35.58, 49.06],
    }
    draw_radar(pdf1, 155, 155, 105, axes1, series1, "Task 1-4 classification profile")
    pdf1.save()

    pdf2 = Pdf(FIG_DIR / "nighttrap_radar_task5_diagnostics.pdf", 420, 330)
    axes2a = ["P@50", "P@100", "NDCG@100", "Macro-F1", "AP@95R"]
    series2a = {
        "Qwen3 6000": [66.00, 64.00, 67.96, 38.91, 7.42],
        "CLIP/context": [32.00, 33.00, 31.32, 49.65, 8.43],
    }
    draw_radar(pdf2, 155, 155, 105, axes2a, series2a, "Binary needs-review queue")
    pdf2.save()


def make_inventory() -> None:
    inventory = {
        "classification_table": {
            "path": "results/nighttrap_tables/table_workflow_task_groups_v09.tex",
            "metric": "Macro-F1 (%) with Accuracy (%) in parentheses",
            "gpt54_final_completed": {
                "image_usability": {"total": 992, "completed": 992, "failed": 0, "accuracy": 0.7913306451612904, "macro_f1": 0.7903743970599494},
                "empty": {"total": 785, "completed": 785, "failed": 0, "accuracy": 0.9006369426751593, "macro_f1": 0.8168214345547232},
                "species": {"total": 1775, "completed": 1775, "failed": 0, "accuracy": 0.887887323943662, "macro_f1": 0.34125177848826377},
                "count": {"total": 3328, "completed": 3328, "failed": 0, "accuracy": 0.8257211538461539, "macro_f1": 0.6447807365127949},
            },
        },
        "needs_review_table": {
            "path": "results/nighttrap_tables/table_needs_review_dense_v09.tex",
            "routine_stress": "6000-event binary needs-review recommendation",
            "hardset": "983-event three-way severity diagnostic",
        },
        "ablation_table": {
            "path": "results/nighttrap_tables/table_ablation_diagnostics_v09.tex",
            "sources": [
                "results/nighttrap_diagnostics/event_representation_ablation_v08/summary.json",
                "results/nighttrap_diagnostics/context_leakage_audit_v08/summary.json",
                "results/nighttrap_diagnostics/robustness_v08/robustness_v08_summary.json",
                "results/nighttrap_diagnostics/megadetector_audit_v08/audit_summary.json",
            ],
        },
        "figures": [
            "figures/nighttrap_radar_task1_4.pdf",
            "figures/nighttrap_radar_task5_diagnostics.pdf",
        ],
    }
    write(TABLE_DIR / "result_inventory_v09.json", json.dumps(inventory, ensure_ascii=False, indent=2))


def main() -> None:
    make_tables()
    make_radar_figures()
    make_inventory()
    print(json.dumps({
        "tables": [
            str(TABLE_DIR / "table_workflow_task_groups_v09.tex"),
            str(TABLE_DIR / "table_needs_review_dense_v09.tex"),
            str(TABLE_DIR / "table_ablation_diagnostics_v09.tex"),
        ],
        "figures": [
            str(FIG_DIR / "nighttrap_radar_task1_4.pdf"),
            str(FIG_DIR / "nighttrap_radar_task5_diagnostics.pdf"),
        ],
        "inventory": str(TABLE_DIR / "result_inventory_v09.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
