# Results — measured outputs from the MI300X run

All numbers in `paper/main.tex` and `paper/PRESENTATION_OUTLINE.md` come from the
`*_metrics.json` files here, produced by `03_evaluate_explain.ipynb`.

| Run | In-dist Acc / Macro-F1 | OOD Acc / Macro-F1 | OOD drop |
|---|---|---|---|
| resnet50_stain | 94.57 / 91.63 | 74.01 / 70.12 | 20.56 |
| vit_b_stain | 97.42 / 96.52 | 81.94 / 78.84 | 15.49 |
| vit_l_stain | 97.30 / 96.13 | 84.36 / 81.00 | 12.94 |
| vit_l_nostain | 97.92 / 97.19 | 53.94 / 51.52 | 43.98 |
| phikon_stain | 96.66 / 94.97 | 78.02 / 75.42 | 18.64 |

- `*_confusion.png` — row-normalized confusion matrix on CRC-VAL-HE-7K (in-distribution).
- `*_gradcam.png` — per-class Grad-CAM overlays (not produced for `phikon_stain`: Phikon
  is a frozen non-timm-ViT/ResNet backbone, unsupported by `make_gradcam`).
- `demo_offline_examples.png` / `.json` — one example per class run through the
  `04_gradio_demo.ipynb` `classify()` function (model: `vit_l_stain`), with predicted
  class, confidence, and Grad-CAM overlay — captured for offline/video use.

OOD = accuracy on a synthetic stain-shifted copy of CRC-VAL-HE-7K (HED jitter θ=0.15 +
strong HSV perturbation), see `main.tex` Section 5.4.
