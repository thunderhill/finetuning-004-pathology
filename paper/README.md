# Paper — Stain-Robust Histology Classification on AMD MI300X

LaTeX source for the manuscript.

## Files
- `main.tex` — the manuscript (single-column `article`, portable packages only).
- `references.bib` — 23 references (all real, verifiable).
- `main.pdf` — last compiled output (8 pages, predates the measured-results update below).
- `figures/` — `confusion_matrix.png` and `gradcam.png` (ViT-L/16, copied from
  `outputs/vit_l_stain_confusion.png` / `outputs/vit_l_stain_gradcam.png`).

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

## Status
The quantitative numbers in the **Results** section are now **measured** results from the full
MI300X training + evaluation campaign (5 models: `resnet50_stain`, `vit_b_stain`, `vit_l_stain`,
`vit_l_nostain`, `phikon_stain`, evaluated on NCT-CRC-HE-100K / CRC-VAL-HE-7K plus a synthetic OOD
stain-shift set). Source of truth: `/workspace/shared/ft004/outputs/*_metrics.json`.

`main.pdf` predates this update and has not been recompiled (this environment has no
`pdflatex`/`bibtex`). To get an updated PDF, run the Build steps above on a machine with TeX Live,
or upload `main.tex` + `references.bib` + `figures/` to Overleaf.
