"""
Build submission PDF from all notebooks — no mentions of Claude/Anthropic.
Output: FINETUNING_004_submission.pdf
"""
import json, re, textwrap
from pathlib import Path
from fpdf import FPDF, XPos, YPos

REPO = Path(__file__).parent
FONT_DIR = Path("/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf")

NOTEBOOKS = [
    ("shared.ipynb",              "Shared Library  (shared.ipynb)"),
    ("00_environment_check.ipynb","00 — Environment Check"),
    ("01_data.ipynb",             "01 — Data Preparation"),
    ("02_train.ipynb",            "02 — Training"),
    ("03_evaluate_explain.ipynb", "03 — Evaluation & Explainability"),
    ("04_gradio_demo.ipynb",      "04 — Interactive Demo"),
    ("05_showcase.ipynb",         "05 — Visual Showcase"),
]

_STRIP = re.compile(
    r"(co.?authored.?by|generated with|#\s*claude\b)",
    re.IGNORECASE,
)

def clean(text):
    return "\n".join(
        line for line in text.splitlines()
        if not _STRIP.search(line)
    )


class SubmissionPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(20, 20, 20)
        self.set_auto_page_break(auto=True, margin=20)
        self.add_font("Sans",  fname=str(FONT_DIR / "DejaVuSans.ttf"))
        self.add_font("Sans",  style="B", fname=str(FONT_DIR / "DejaVuSans-Bold.ttf"))
        self.add_font("Mono",  fname=str(FONT_DIR / "DejaVuSansMono.ttf"))
        self.add_font("Mono",  style="B", fname=str(FONT_DIR / "DejaVuSansMono-Bold.ttf"))
        self._usable = self.w - self.l_margin - self.r_margin

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Sans", size=7.5)
        self.set_text_color(160, 160, 160)
        self.set_x(self.l_margin)
        self.cell(self._usable, 5,
                  "FINETUNING_004 — Stain-Robust Pathology Classifier",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def footer(self):
        self.set_y(-14)
        self.set_font("Sans", size=7.5)
        self.set_text_color(160, 160, 160)
        self.set_x(self.l_margin)
        self.cell(self._usable, 5, f"Page {self.page_no()}", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── Cover ──────────────────────────────────────────────────────────────────
    def cover(self):
        self.add_page()
        self.set_y(55)
        self.set_x(self.l_margin)
        self.set_font("Sans", style="B", size=24)
        self.set_text_color(20, 20, 20)
        self.multi_cell(self._usable, 13, "FINETUNING_004", align="C")
        self.ln(3)
        self.set_x(self.l_margin)
        self.set_font("Sans", style="B", size=15)
        self.set_text_color(50, 50, 50)
        self.multi_cell(self._usable, 9, "Stain-Robust Pathology Classifier", align="C")
        self.ln(2)
        self.set_x(self.l_margin)
        self.set_font("Sans", size=11)
        self.set_text_color(100, 100, 100)
        self.multi_cell(self._usable, 7, "Fine-tuned ViT-L/16 on AMD MI300X", align="C")
        self.ln(5)
        self.set_x(self.l_margin)
        self.set_font("Sans", style="B", size=11)
        self.set_text_color(30, 30, 30)
        self.multi_cell(self._usable, 7, "P. Mahipal Rao   *   Sai Kalyan Guddeti", align="C")
        self.ln(1)
        self.set_x(self.l_margin)
        self.set_font("Sans", size=9.5)
        self.set_text_color(100, 100, 100)
        self.multi_cell(self._usable, 6,
            "p.mahipal@tcs.com   *   saikalyan.guddeti@tcs.com", align="C")
        self.ln(2)
        self.set_x(self.l_margin)
        self.multi_cell(self._usable, 6, "AMD Developer Challenge", align="C")
        self.ln(8)
        self._rule()
        self.ln(8)
        details = [
            ("Dataset",      "NCT-CRC-HE-100K (100 K patches) + CRC-VAL-HE-7K (7 K patches)"),
            ("Architecture", "ViT-L/16 (307 M params), full fine-tune, bfloat16, torch.compile"),
            ("Hardware",     "Single AMD MI300X GPU  (192 GB HBM3)"),
            ("Clean acc.",   "97.3 %"),
            ("OOD acc.",     "84.4 %  (synthetically stain-shifted validation set)"),
            ("Robustness",   "3.4x smaller accuracy drop vs baseline  (44 pp → 12.9 pp)"),
            ("Models",       "ResNet-50, ViT-B/16, ViT-L/16, Phikon, ViT-L/16 no-aug ablation"),
        ]
        for label, value in details:
            self.set_x(self.l_margin)
            self.set_font("Sans", style="B", size=9.5)
            self.set_text_color(40, 40, 40)
            self.cell(32, 6.5, label + ":", new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.set_font("Sans", size=9.5)
            self.set_text_color(70, 70, 70)
            self.multi_cell(self._usable - 32, 6.5, value)
        self.ln(8)
        self._rule()
        self.ln(6)
        self.set_x(self.l_margin)
        self.set_font("Sans", size=9)
        self.set_text_color(100, 100, 100)
        toc = "  ·  ".join(nb for nb, _ in NOTEBOOKS)
        self.multi_cell(self._usable, 6, "Notebooks: " + toc)

    def _rule(self):
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.4)
        y = self.get_y()
        self.line(self.l_margin, y, self.l_margin + self._usable, y)

    # ── Section title ──────────────────────────────────────────────────────────
    def section_title(self, title):
        self.add_page()
        self.set_x(self.l_margin)
        self.set_font("Sans", style="B", size=13)
        self.set_text_color(15, 15, 15)
        self.set_fill_color(238, 238, 238)
        self.cell(self._usable, 10, title, fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    # ── Markdown prose ─────────────────────────────────────────────────────────
    def markdown_block(self, text):
        text = clean(text).strip()
        if not text:
            return
        for line in text.splitlines():
            line = line.rstrip()
            safe = line.encode("latin-1", errors="replace").decode("latin-1")
            self.set_x(self.l_margin)
            if line.startswith("### "):
                self.set_font("Sans", style="B", size=9)
                self.set_text_color(30, 30, 30)
                self.multi_cell(self._usable, 5.5, safe[4:])
            elif line.startswith("## "):
                self.set_font("Sans", style="B", size=10.5)
                self.set_text_color(20, 20, 20)
                self.multi_cell(self._usable, 6, safe[3:])
            elif line.startswith("# "):
                self.set_font("Sans", style="B", size=12)
                self.set_text_color(10, 10, 10)
                self.multi_cell(self._usable, 7, safe[2:])
            else:
                self.set_font("Sans", size=8.5)
                self.set_text_color(65, 65, 65)
                self.multi_cell(self._usable, 5, safe)
        self.ln(2)

    # ── Code block ────────────────────────────────────────────────────────────
    def code_block(self, text):
        text = clean(text).strip()
        if not text:
            return
        self.set_font("Mono", size=7.2)
        char_w = self.get_string_width("m")
        max_chars = max(20, int(self._usable / char_w) - 2)
        line_h = 4.0
        wrapped = []
        for line in text.splitlines():
            safe = line.encode("latin-1", errors="replace").decode("latin-1")
            if len(safe) <= max_chars:
                wrapped.append(safe)
            else:
                for chunk in textwrap.wrap(
                    safe, max_chars,
                    break_long_words=True,
                    subsequent_indent="    ",
                    expand_tabs=False,
                ):
                    wrapped.append(chunk)

        block_h = len(wrapped) * line_h + 5
        if self.get_y() + block_h > self.h - self.b_margin - 5:
            self.add_page()

        x0, y0 = self.l_margin, self.get_y()
        self.set_fill_color(247, 247, 247)
        self.set_draw_color(215, 215, 215)
        self.set_line_width(0.3)
        self.rect(x0, y0, self._usable, block_h, style="FD")

        self.set_text_color(25, 25, 25)
        self.set_y(y0 + 2.5)
        for line in wrapped:
            self.set_x(x0 + 2)
            self.cell(self._usable - 4, line_h, line,
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)


def build():
    pdf = SubmissionPDF()
    pdf.cover()

    for filename, title in NOTEBOOKS:
        path = REPO / filename
        if not path.exists():
            print(f"  SKIP: {filename}")
            continue
        print(f"  {filename} ...")
        nb = json.loads(path.read_text())
        pdf.section_title(title)
        for cell in nb.get("cells", []):
            src = "".join(cell.get("source", []))
            if not src.strip():
                continue
            if cell["cell_type"] == "markdown":
                pdf.markdown_block(src)
            elif cell["cell_type"] == "code":
                pdf.code_block(src)

    out = REPO / "FINETUNING_004_submission.pdf"
    pdf.output(str(out))
    size_kb = out.stat().st_size // 1024
    print(f"\nDone  →  {out.name}  ({size_kb} KB)")


if __name__ == "__main__":
    build()
