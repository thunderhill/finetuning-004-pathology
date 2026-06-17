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

### Systems + robustness measurements (produced by `benchmark_and_ood.py`)
- `mi300x_benchmark.json` — measured ViT-L/16 training-step performance on the MI300X at
  batch 512, native bf16: throughput (640 eager / 725 compiled img/s), step time, and peak
  HBM (90.4 GB — exceeds an 80 GB-class card). `torch.compile` speedup 1.13×.
- `scanner_ood_metrics.json` — second-mechanism OOD eval of `vit_l_stain`: a corruption
  family disjoint from training (JPEG + blur + downscale + RGB cast + gamma; no HED). Drop
  is only 4.58 pts (97.30 → 92.72), evidence the robustness is not mere inversion of the
  training augmentation.
- `colorbias_baseline.json` — colour-bias probe: logistic regression on per-image mean+std
  RGB (6 features) reaches 72.1% acc / 63.9% macro-F1 on the 9-class val set vs 11.1%
  chance — quantifies why colour-invariant training matters.

- `*_confusion.png` — row-normalized confusion matrix on CRC-VAL-HE-7K (in-distribution).
- `*_gradcam.png` — per-class Grad-CAM overlays (not produced for `phikon_stain`: Phikon
  is a frozen non-timm-ViT/ResNet backbone, unsupported by `make_gradcam`).
- `demo_offline_examples.png` / `.json` — one example per class run through the
  `04_gradio_demo.ipynb` `classify()` function (model: `vit_l_stain`), with predicted
  class, confidence, and Grad-CAM overlay — captured for offline/video use.

OOD = accuracy on a synthetic stain-shifted copy of CRC-VAL-HE-7K (HED jitter θ=0.15 +
strong HSV perturbation), see `main.tex` Section 5.4.
