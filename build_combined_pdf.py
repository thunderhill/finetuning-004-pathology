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
    """Return a single-page cover as PDF bytes."""

    class CoverPDF(FPDF):
        def __init__(self):
            super().__init__(orientation="P", unit="mm", format="A4")
            self.set_margins(22, 22, 22)
            self.set_auto_page_break(False)   # force single page
            self.add_font("Sans",  fname=str(FONT_DIR / "DejaVuSans.ttf"))
            self.add_font("Sans",  style="B", fname=str(FONT_DIR / "DejaVuSans-Bold.ttf"))
            self._W = self.w - 44

    pdf = CoverPDF()
    pdf.add_page()

    def rule():
        y = pdf.get_y()
        pdf.set_draw_color(180, 180, 180)
        pdf.set_line_width(0.5)
        pdf.line(22, y, 22 + pdf._W, y)

    # ── Title ─────────────────────────────────────────────────────────────────
    pdf.set_y(36)
    pdf.set_x(22)
    pdf.set_font("Sans", style="B", size=20)
    pdf.set_text_color(15, 15, 15)
    pdf.multi_cell(pdf._W, 11, "Stain-Robust Histology Classification at Scale", align="C")
    pdf.ln(1)
    pdf.set_x(22)
    pdf.set_font("Sans", size=11)
    pdf.set_text_color(70, 70, 70)
    pdf.multi_cell(pdf._W, 6.5, "Full Fine-Tuning of Vision Transformers on a Single AMD MI300X", align="C")
    pdf.ln(4)
    rule()
    pdf.ln(4)

    # ── Authors ───────────────────────────────────────────────────────────────
    pdf.set_x(22)
    pdf.set_font("Sans", style="B", size=11)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(pdf._W, 6.5, "P. Mahipal Rao     Sai Kalyan Guddeti", align="C")
    pdf.ln(0.5)
    pdf.set_x(22)
    pdf.set_font("Sans", size=9.5)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(pdf._W, 5.5, "p.mahipal@tcs.com     saikalyan.guddeti@tcs.com", align="C")
    pdf.ln(1)
    pdf.set_x(22)
    pdf.multi_cell(pdf._W, 5.5, "AMD Developer Challenge", align="C")
    pdf.ln(5)
    rule()
    pdf.ln(5)

    # ── Contents ──────────────────────────────────────────────────────────────
    pdf.set_x(22)
    pdf.set_font("Sans", style="B", size=10.5)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(pdf._W, 6, "Contents", align="L")
    pdf.ln(2)

    sections = [
        (
            "Part 1  —  Research Paper  (pages 2 – 9)",
            [
                "Problem: stain-colour variation causes pathology AI models to collapse on out-of-distribution slides.",
                "Solution: HED colour-space augmentation reducing accuracy drop from 44 pp to 12.9 pp (3.4x improvement).",
                "Results: ResNet-50, ViT-B/16, ViT-L/16, Phikon, and a no-aug ablation on AMD MI300X.",
            ],
        ),
        (
            "Part 2  —  Full Notebook Code  (pages 10 – 39)",
            [
                "shared.ipynb  —  augmentation, models, Grad-CAM, metrics.",
                "00  environment check   01  data preparation   02  training",
                "03  evaluation & explainability   04  interactive demo   05  visual showcase",
            ],
        ),
    ]

    for heading, bullets in sections:
        pdf.set_x(22)
        pdf.set_font("Sans", style="B", size=9.5)
        pdf.set_text_color(30, 30, 30)
        pdf.set_fill_color(238, 238, 238)
        pdf.cell(pdf._W, 7, heading, fill=True,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(0.5)
        for b in bullets:
            safe = b.encode("latin-1", errors="replace").decode("latin-1")
            pdf.set_x(26)
            pdf.set_font("Sans", size=8.5)
            pdf.set_text_color(55, 55, 55)
            pdf.multi_cell(pdf._W - 4, 5, "* " + safe)
        pdf.ln(3)

    rule()
    pdf.ln(5)

    # ── Key numbers ───────────────────────────────────────────────────────────
    pdf.set_x(22)
    pdf.set_font("Sans", style="B", size=9.5)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(pdf._W, 5.5, "Key results at a glance", align="L")
    pdf.ln(1.5)

    stats = [
        ("Dataset",      "NCT-CRC-HE-100K (100K train) + CRC-VAL-HE-7K (7K val), 9 tissue classes"),
        ("Best model",   "ViT-L/16  —  97.3% clean accuracy,  84.4% under synthetic stain shift"),
        ("Robustness",   "3.4x smaller drop vs baseline  (44.0 pp  to  12.9 pp)"),
        ("Hardware",     "Single AMD MI300X  192 GB HBM3  bfloat16  torch.compile"),
    ]
    for label, value in stats:
        pdf.set_x(22)
        pdf.set_font("Sans", style="B", size=8.5)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(28, 5.5, label + ":", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Sans", size=8.5)
        pdf.set_text_color(70, 70, 70)
        safe = value.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(pdf._W - 28, 5.5, safe)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def make_author_patch() -> bytes:
    """
    One A4 page: white rectangle over the old author block, then correct author text.
    Coordinates measured from the compiled main.pdf (points from bottom):
      Team FINETUNING_004      y≈591
      AMD Developer Challenge  y≈577
      {author}@{institution}  y≈563
    In fpdf2 (mm from top, A4=297mm): 88–101mm.
    """
    class PatchPDF(FPDF):
        def __init__(self):
            super().__init__(orientation="P", unit="mm", format="A4")
            self.set_margins(0, 0, 0)
            self.set_auto_page_break(False)
            self.add_font("Sans", fname=str(FONT_DIR / "DejaVuSans.ttf"))
            self.add_font("Sans", style="B", fname=str(FONT_DIR / "DejaVuSans-Bold.ttf"))

    pdf = PatchPDF()
    pdf.add_page()

    # White rectangle covering old author lines (85–105 mm from top)
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(0, 85, 210, 20, style="F")

    # New author text, centred in the same area
    pdf.set_font("Sans", style="B", size=11)
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(0, 87)
    pdf.cell(210, 6, "P. Mahipal Rao     Sai Kalyan Guddeti", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Sans", size=9.5)
    pdf.set_text_color(80, 80, 80)
    pdf.set_x(0)
    pdf.cell(210, 5.5, "p.mahipal@tcs.com     saikalyan.guddeti@tcs.com", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(0)
    pdf.cell(210, 5.5, "AMD Developer Challenge", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def build():
    print("Building cover page ...")
    cover_bytes = make_cover()

    print("Building author patch for paper page 1 ...")
    patch_bytes  = make_author_patch()
    patch_reader = PdfReader(io.BytesIO(patch_bytes))
    patch_page   = patch_reader.pages[0]

    writer = PdfWriter()

    # Cover
    cover_reader = PdfReader(io.BytesIO(cover_bytes))
    for page in cover_reader.pages:
        writer.add_page(page)

    # Paper — patch author block on page 0 only
    print(f"Appending paper ({PAPER.name}) ...")
    paper_reader = PdfReader(str(PAPER))
    for i, page in enumerate(paper_reader.pages):
        if i == 0:
            page.merge_page(patch_page)
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
