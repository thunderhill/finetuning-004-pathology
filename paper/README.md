# Paper — Stain-Robust Histology Classification on AMD MI300X

LaTeX source for the manuscript.

## Files
- `main.tex` — the manuscript (single-column `article`, portable packages only).
- `references.bib` — 23 references (all real, verifiable).
- `main.pdf` — last compiled output (8 pages).

## Build
```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```
Compiles on a stock TeX Live (and on Overleaf — just upload `main.tex` + `references.bib`).
It deliberately avoids `booktabs`/`xcolor`/`caption`/`multirow` so it builds on minimal installs;
swap in `booktabs` for nicer rules if your TeX install has it.

## ⚠️ Before submitting anywhere real
The quantitative numbers in the **Results** section are **projected targets**, not measured data —
the MI300X evaluation campaign has not been run. Every such cell is:
- flagged by the italic note under "Results" and the table captions, and
- marked with a `% TODO` comment in `main.tex` (grep for `TODO`).

Replace them with the measured values and regenerate the figures:
1. Run `02_train.ipynb` for `resnet50`, `vit_b`, `vit_l` (stain + no-stain), and `phikon`.
2. Run `03_evaluate_explain.ipynb` to produce `outputs/*_confusion.png`, `*_gradcam.png`,
   and `*_metrics.json` (incl. the OOD ablation).
3. Copy those PNGs into `paper/figures/` and uncomment the `\includegraphics` lines (replacing the
   `\fakefig` placeholders) in Figures 1–2.
4. Fill Tables 2–3 from the `*_metrics.json` files.

Presenting the projected numbers as measured results would be misrepresentation — don't.
