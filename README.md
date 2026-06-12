# FINETUNING_004 — Stain-Robust Pathology Image Classifier (AMD MI300X)

9-class colorectal-histology classification on **NCT-CRC-HE-100K** / **CRC-VAL-HE-7K**, built to
be **resilient to H&E stain variation** — the #1 real-world failure mode in computational
pathology. Authored for **AMD's Jupyter (notebooks.amd.com)** running on a single **MI300X
(192GB HBM3)** with native **bf16** and `torch.compile`.

## What's here (notebook-only)

| File | Role |
|---|---|
| `00_environment_check.ipynb` | ROCm/bf16 sanity, `pip install`, `wandb login`, set `PERSIST_DIR`. **Run first each day.** |
| `shared.ipynb` | Core library pulled in via `%run shared.ipynb`: dataset, Albumentations + `HEDStainJitter`, Macenko, `build_model` (timm + frozen-Phikon), Grad-CAM, metrics, checkpoints. |
| `01_data.ipynb` | Download/extract data, class-distribution EDA, augmentation preview grid. |
| `02_train.ipynb` | Native bf16 training loop, `torch.compile`, W&B logging, **resume-safe** checkpoints. |
| `03_evaluate_explain.ipynb` | Accuracy / macro-F1 / per-class F1, confusion matrix, **OOD stain-shift ablation**, Grad-CAM overlays. |
| `04_gradio_demo.ipynb` | CPU Gradio demo: patch → class + confidence + 9-class distribution + Grad-CAM. |

> `_build_notebooks.py` is the generator used to author the notebooks; it is not needed at runtime.

## Design choices

- **Stain robustness (the differentiator):** `HEDStainJitter` perturbs the H&E (HED) colour space
  during training (Tellez et al. 2018) + aggressive spatial transforms (rotate/flip/elastic/
  distortion) via **Albumentations**, forcing the model onto morphology rather than colour. Toggle
  with `use_stain_aug` to reproduce the ablation.
- **Two model configs** (`build_model`): full fine-tune via **timm** (`resnet50` / `vit_b` / `vit_l`
  / `swin_b`, SDPA attention) **or** a frozen **Phikon** (`owkin/phikon`, un-gated) feature
  extractor + trainable MLP head.
- **MI300X bf16:** `torch.autocast(dtype=bfloat16)` with **no GradScaler** (native bf16); an
  fp16+GradScaler branch keeps it portable off-MI300X.
- **Resume-safe:** atomic `last.pth` every epoch + persisted W&B run-id, so the run continues
  across the three separate 6-hour GPU windows.

## Quickstart (on notebooks.amd.com)

1. Open `00_environment_check.ipynb` → run all. Confirm `GPU available: True`, `bf16 native: True`,
   and set `PERSIST_DIR` to your **mounted/persistent** folder (default `/workspace/shared/ft004`).
2. `01_data.ipynb` → downloads ~10 GB once into `PERSIST_DIR/data` (cached after).
3. `02_train.ipynb` → set the **CONFIG** cell `model` and run. Re-run after a session ends to resume.
4. `03_evaluate_explain.ipynb` → set `RUN_NAME`, produces metrics + confusion matrix + OOD ablation
   + Grad-CAM PNGs in `PERSIST_DIR/outputs`.
5. `04_gradio_demo.ipynb` → launches the CPU demo on port 7860.

## Suggested 3-day schedule (6h windows)

- **Day 1:** `00` → `01` → train `resnet50` (bank ~94–95%) then start `vit_b`.
- **Day 2:** train `vit_l` (hero, batch up to 512) + a `vit_l` run with `use_stain_aug=False`
  (ablation) + `phikon`.
- **Day 3:** `03` metrics/ablation/Grad-CAM assets, `04` demo, finalize plots + deck.

## Prerequisites / gotchas

- **W&B:** needs an API key (`wandb login`); set `WANDB_MODE=offline` if egress is blocked.
- **`PERSIST_DIR`** must be the persistent mount or Day 2/3 lose data + checkpoints.
- **DataLoader workers:** the container needs `--ipc=host` / a large `--shm-size`; otherwise lower
  `num_workers` to avoid "unexpected bus error".
- **Foundation models:** only Phikon is wired (un-gated). UNI / Prov-GigaPath are gated — add your
  HF token and a new branch in `build_model` if you obtain access.

## Data & references

- NCT-CRC-HE-100K / CRC-VAL-HE-7K — Zenodo record **1214456** (public domain, Macenko pre-normalized).
- Classes: `ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM`.
