"""
Build combined.pdf  =  cover page  +  paper  +  notebook code
"""
import io
from pathlib import Path
from fpdf import FPDF, XPos, YPos
from pypdf import PdfWriter, PdfReader

REPO     = Path(__file__).parent
FONT_DIR = Path("/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf")
PAPER    = REPO / "paper" / "main.pdf"
CODE     = REPO / "FINETUNING_004_submission.pdf"
OUT      = REPO / "combined.pdf"


def make_cover() -> bytes:
    """Return a one-page cover as PDF bytes."""

    class CoverPDF(FPDF):
        def __init__(self):
            super().__init__(orientation="P", unit="mm", format="A4")
            self.set_margins(22, 22, 22)
            self.add_font("Sans",  fname=str(FONT_DIR / "DejaVuSans.ttf"))
            self.add_font("Sans",  style="B", fname=str(FONT_DIR / "DejaVuSans-Bold.ttf"))
            self.add_font("Mono",  fname=str(FONT_DIR / "DejaVuSansMono.ttf"))
            self._W = self.w - 44  # usable width

    pdf = CoverPDF()
    pdf.add_page()

    def rule(y_offset=0):
        y = pdf.get_y() + y_offset
        pdf.set_draw_color(180, 180, 180)
        pdf.set_line_width(0.5)
        pdf.line(22, y, 22 + pdf._W, y)

    # ── Title block ────────────────────────────────────────────────────────────
    pdf.set_y(48)
    pdf.set_x(22)
    pdf.set_font("Sans", style="B", size=22)
    pdf.set_text_color(15, 15, 15)
    pdf.multi_cell(pdf._W, 12,
        "Stain-Robust Histology Classification at Scale",
        align="C")
    pdf.ln(2)
    pdf.set_x(22)
    pdf.set_font("Sans", size=12)
    pdf.set_text_color(70, 70, 70)
    pdf.multi_cell(pdf._W, 7,
        "Full Fine-Tuning of Vision Transformers on a Single AMD MI300X",
        align="C")
    pdf.ln(5)
    rule()
    pdf.ln(5)

    # ── Authors / affiliation ──────────────────────────────────────────────────
    pdf.set_x(22)
    pdf.set_font("Sans", style="B", size=11)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(pdf._W, 7, "P. Mahipal Rao   *   Sai Kalyan Guddeti", align="C")
    pdf.ln(1)
    pdf.set_x(22)
    pdf.set_font("Sans", size=9.5)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(pdf._W, 6, "p.mahipal@tcs.com   *   saikalyan.guddeti@tcs.com", align="C")
    pdf.ln(2)
    pdf.set_x(22)
    pdf.multi_cell(pdf._W, 6, "AMD Developer Challenge", align="C")
    pdf.ln(8)
    rule()
    pdf.ln(8)

    # ── What's in this document ────────────────────────────────────────────────
    pdf.set_x(22)
    pdf.set_font("Sans", style="B", size=11)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(pdf._W, 7, "What's in this document", align="L")
    pdf.ln(3)

    sections = [
        (
            "Part 1  —  Research Paper  (pages 2 – 9)",
            [
                "Problem: stain-colour variation causes pathology AI models to collapse "
                "on out-of-distribution slides.",
                "Solution: HED colour-space augmentation trained into a ViT-L/16, reducing "
                "the stain-shift accuracy drop from 44 pp to 12.9 pp (3.4× improvement).",
                "Results across five architectures: ResNet-50, ViT-B/16, ViT-L/16, Phikon "
                "(foundation model), and a no-aug ablation.",
                "Hardware: single AMD MI300X (192 GB HBM3), bfloat16, torch.compile.",
            ],
        ),
        (
            "Part 2  —  Full Notebook Code  (pages 10 – 39)",
            [
                "shared.ipynb  —  shared library: augmentation, models, Grad-CAM, metrics.",
                "00_environment_check.ipynb  —  GPU / package validation.",
                "01_data.ipynb  —  dataset download and preprocessing.",
                "02_train.ipynb  —  training loop for all five models.",
                "03_evaluate_explain.ipynb  —  accuracy, F1, confusion matrices, Grad-CAM.",
                "04_gradio_demo.ipynb  —  interactive H&E patch classifier with Grad-CAM overlay.",
                "05_showcase.ipynb  —  visual showcase: side-by-side Grad-CAM + probability charts.",
            ],
        ),
    ]

    for heading, bullets in sections:
        pdf.set_x(22)
        pdf.set_font("Sans", style="B", size=10)
        pdf.set_text_color(30, 30, 30)
        pdf.set_fill_color(238, 238, 238)
        pdf.cell(pdf._W, 8, heading, fill=True,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
        for b in bullets:
            safe = b.encode("latin-1", errors="replace").decode("latin-1")
            pdf.set_x(26)
            pdf.set_font("Sans", size=9)
            pdf.set_text_color(55, 55, 55)
            pdf.multi_cell(pdf._W - 4, 5.5, "•  " + safe)
        pdf.ln(4)

    rule()
    pdf.ln(6)

    # ── Key numbers ────────────────────────────────────────────────────────────
    pdf.set_x(22)
    pdf.set_font("Sans", style="B", size=10)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(pdf._W, 6, "Key results at a glance", align="L")
    pdf.ln(2)

    stats = [
        ("Dataset",       "NCT-CRC-HE-100K (100 K train) + CRC-VAL-HE-7K (7 K val), 9 tissue classes"),
        ("Best model",    "ViT-L/16  —  97.3% clean accuracy,  84.4% under synthetic stain shift"),
        ("Robustness",    "3.4× smaller drop vs baseline  (44.0 pp  →  12.9 pp)"),
        ("Hardware",      "Single AMD MI300X · 192 GB HBM3 · bfloat16 · torch.compile"),
    ]
    for label, value in stats:
        pdf.set_x(22)
        pdf.set_font("Sans", style="B", size=9)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(30, 6, label + ":", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Sans", size=9)
        pdf.set_text_color(70, 70, 70)
        safe = value.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(pdf._W - 30, 6, safe)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def build():
    print("Building cover page ...")
    cover_bytes = make_cover()

    writer = PdfWriter()

    # Cover
    cover_reader = PdfReader(io.BytesIO(cover_bytes))
    for page in cover_reader.pages:
        writer.add_page(page)

    # Paper
    print(f"Appending paper ({PAPER.name}) ...")
    paper_reader = PdfReader(str(PAPER))
    for page in paper_reader.pages:
        writer.add_page(page)

    # Code
    print(f"Appending code ({CODE.name}) ...")
    code_reader = PdfReader(str(CODE))
    for page in code_reader.pages:
        writer.add_page(page)

    with open(OUT, "wb") as f:
        writer.write(f)

    size_kb = OUT.stat().st_size // 1024
    total   = len(writer.pages)
    print(f"\nDone  →  {OUT.name}  ({total} pages, {size_kb} KB)")


if __name__ == "__main__":
    build()
