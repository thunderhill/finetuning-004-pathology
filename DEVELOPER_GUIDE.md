# Developer Guide — FINETUNING_004

This guide is for developers extending or maintaining the pipeline. For a quickstart, see
[`README.md`](README.md). This document is about *how the code works and how to change it*.

---

## 1. Architecture at a glance

It's a **computer-vision image classifier** (no LLM): 224×224 H&E patches → 1 of 9 tissue classes
(`ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM`). The differentiator is **stain-variation
robustness** via aggressive augmentation.

```
                       ┌──────────────────────────────────────────┐
                       │  shared.ipynb  (the library, %run-imported)│
                       │  Phase A: data + augmentation              │
                       │  Phase B: model factory                    │
                       │  Helpers: metrics, Grad-CAM, checkpoints   │
                       └──────────────────────────────────────────┘
                              ▲          ▲          ▲          ▲
                  %run        │          │          │          │
        ┌─────────────────────┘   ┌──────┘   ┌──────┘   ┌───────┘
   00_environment_check     01_data    02_train   03_evaluate_explain   04_gradio_demo
   (sanity, deps, wandb)    (download,  (bf16 loop,(metrics, OOD         (CPU demo)
                             EDA, aug)   resume)    ablation, Grad-CAM)
```

**Key idea:** every notebook starts with `%run shared.ipynb`, which loads all classes/functions
into the namespace with **no side effects** (no downloads, no model builds on import). This keeps
the suite DRY while staying "notebook-only".

### Data flow

```
ImageFolder (class subdirs)
   └─ PathologyPatchDataset  →  Albumentations Compose  →  (CHW float32 tensor, label int)
        └─ make_dataloaders  →  DataLoader (bf16-ready batches)
             └─ build_model(cfg)  →  logits [B, 9]
                  └─ CrossEntropy(label_smoothing) / metrics / Grad-CAM
```

---

## 2. The build system

The notebooks are **generated** from [`_build_notebooks.py`](_build_notebooks.py) — do not hand-edit
the `.ipynb` files if you want changes to survive a regenerate. The generator is the source of truth.

```bash
python _build_notebooks.py        # regenerates all 6 .ipynb files
```

Helpers inside it:
- `nb(cells)` — wraps a cell list into nbformat-v4 JSON.
- `md(s)` / `code(s)` — build a markdown / code cell (`source` is split into lines with `keepends`).
- `save(name, cells)` — writes the `.ipynb`.

**Authoring rules (learned the hard way):** cell bodies are stored as `r"""..."""` raw strings so
escapes like `\r` survive verbatim into the notebook. Therefore:
- Use `'''triple-single-quote'''` for docstrings *inside* cells (a `"""` would close the wrapper).
- Never put a literal `"""` inside a cell body.

**Validation after regenerating** (JSON + Python syntax of every code cell, ignoring magics):

```bash
python - <<'PY'
import json, glob, ast, re
for f in sorted(glob.glob("*.ipynb")):
    for i, c in enumerate(json.load(open(f))["cells"]):
        if c["cell_type"] != "code": continue
        src = "".join(c["source"])
        clean = "\n".join(("# "+l if re.match(r"\s*[%!]", l) else l) for l in src.split("\n"))
        try: ast.parse(clean)
        except SyntaxError as e: print(f"{f} cell#{i}: {e}")
print("done")
PY
```

---

## 3. Phase A — data & augmentation (`shared.ipynb`)

### `PathologyPatchDataset(root, classes=None, transform=None, macenko=None)`
- Scans `root/<class>/*.{png,jpg,tif,...}` into a `(path, label)` list; `classes` defaults to the
  sorted subdir names. Pass an explicit `classes` for the val set to **guarantee the same label
  order** as train.
- `__getitem__` loads via PIL→`np.ndarray` (HWC uint8, what Albumentations expects), optionally
  Macenko-normalizes, applies `transform(image=img)["image"]`, returns `(tensor, label)`.
- Corrupt-file safe: a failed read logs a warning and falls through to the next index.

### Augmentation builders
| Function | Use | Contents |
|---|---|---|
| `build_train_aug(use_stain_aug, img_size, theta)` | training | spatial (RandomRotate90/Flip/Affine/Elastic/Grid/Optical) + (if `use_stain_aug`) `HEDStainJitter` + HSV/brightness, then Normalize+ToTensorV2 |
| `build_eval_aug(img_size)` | val / inference | Resize + Normalize + ToTensorV2 only |
| `build_ood_aug(img_size, theta)` | OOD ablation | **strong** HEDStainJitter (theta≈0.15) + big HSV shift — simulates a different lab/scanner |

### `HEDStainJitter(theta=0.05, p=0.5)` — the innovation
An `albumentations.ImageOnlyTransform`. Converts RGB→HED (`skimage.color.rgb2hed`), perturbs each
of the 3 stain channels by `channel * U(1-θ, 1+θ) + U(-θ, θ)`, converts back. Larger `θ` = more
aggressive stain shift. This is what makes the model robust to inter-lab colour variation.

**To tune augmentation strength:** edit `build_train_aug` (probabilities/limits) or pass a different
`hed_theta` in the training `cfg`. To add a transform, insert it into the `tfms` list before the
`Normalize`/`ToTensorV2` tail.

### `ensure_dataset(persist_dir, which=(...), delete_zip=True)`
Downloads + extracts the Zenodo zips (record 1214456) into `persist_dir/data`, idempotently.
`_find_class_root` locates the dir that actually contains the 9 class folders (handles nested
extraction). Set `delete_zip=False` to keep the archives (costs disk; respect the shared-folder
ceiling). To use a **different/local dataset**, just place an ImageFolder-layout dir and point the
notebooks at it — skip `ensure_dataset`.

### `make_dataloaders(train_dir, val_dir, cfg)`
Builds train (shuffled, `drop_last`) and val loaders. Reads `num_workers`, `pin_memory`,
`prefetch_factor` from `cfg`; enables `persistent_workers` when `num_workers > 0`. Returns
`(train_loader, val_loader, classes)`.

---

## 4. Phase B — model factory (`shared.ipynb`)

### `build_model(cfg, num_classes=9)`
Single entry point for both configs, keyed on `cfg["model"]`:

| `cfg["model"]` | Backbone | Trainable |
|---|---|---|
| `resnet50` | timm `resnet50.a1_in1k` | all (full fine-tune) |
| `vit_b` | timm ViT-Base/16 (in21k→1k) | all |
| `vit_l` | timm ViT-Large/16 (in21k→1k) — hero | all |
| `swin_b` | timm Swin-Base | all |
| `phikon` | `owkin/phikon` (HF, frozen) | MLP head only |

`pretrained` defaults to `True`; set `cfg["pretrained"]=False` for random init (used by the smoke
test to avoid downloads).

### Adding a new timm model
Add an entry to `TIMM_MAP`:
```python
TIMM_MAP["convnext_b"] = "convnext_base.fb_in22k_ft_in1k"
```
Then `cfg["model"] = "convnext_b"`. If its classifier sub-module isn't named `head`/`fc`/
`classifier`, also extend `head_keys` in `get_param_groups` so layer-wise LR splits correctly.
For Grad-CAM, add a branch in `make_gradcam` pointing at the right target layer.

### Adding a gated foundation model (UNI / Prov-GigaPath)
These need an HF token + accepted access. Mirror `PhikonClassifier`:
```python
class UNIClassifier(PhikonClassifier):
    def __init__(self, num_classes=9, **kw):
        super().__init__(num_classes, model_id="MahmoodLab/UNI", **kw)
```
…then branch in `build_model`. Set `huggingface-cli login` (or `HF_TOKEN`) first. The CLS-token
extraction (`last_hidden_state[:, 0]`) and frozen-backbone pattern carry over; check
`backbone.config.hidden_size` matches the head input.

### `get_param_groups(model, cfg)` — layer-wise LR
- ViT/Swin/CNN: two groups — **head at `lr`**, **backbone at `lr * backbone_lr_mult`** (default 0.1).
  Prevents catastrophic forgetting of pretrained features while the head adapts fast.
- Phikon: one group (only head params have `requires_grad=True`).

---

## 5. Phase C — training loop (`02_train.ipynb`)

### The `cfg` dict (top CONFIG cell) — every knob
```
model, use_stain_aug, hed_theta, img_size,
epochs, batch_size, eval_batch_size,
lr, backbone_lr_mult, weight_decay, warmup_epochs, label_smoothing,
num_workers, pin_memory, prefetch_factor,
use_compile, resume
```
`run_name = f"{model}_{'stain' if use_stain_aug else 'nostain'}"` → checkpoint dir
`PERSIST_DIR/checkpoints/<run_name>`.

### bf16 on MI300X
`get_amp_dtype(device)` returns **bf16** on MI300X (native, **no GradScaler**), fp16 on other CUDA,
fp32 on CPU. The loop:
```python
with torch.autocast(device_type="cuda", dtype=amp_dtype):
    out = model(x); loss = criterion(out.float(), y)
loss.backward(); optimizer.step()          # scaler only created/used when amp_dtype == float16
```

### `torch.compile`
Wrapped in try/except (`mode="reduce-overhead"`); on ROCm graph-break failure it logs and falls
back to eager. **Important:** compiled models prefix state-dict keys with `_orig_mod.` — always use
`raw_module(model)` when saving/loading state dicts so checkpoints stay portable.

### Resume-safety (the crux for 3× 6-hour windows)
- `save_checkpoint` writes to `*.tmp` then atomically `replace()`s — a killed kernel never corrupts
  `last.pth`.
- `last.pth` is written **every epoch**; `best.pth` on macro-F1 improvement. Each holds
  `epoch, model, optimizer, scheduler, best_metric, metrics, cfg, classes`.
- On start, if `cfg["resume"]` and `last.pth` exists → restore model/opt/scheduler and continue from
  `epoch+1`.
- W&B run-id is persisted to `wandb_run_id.txt` and reused with `resume="allow"`, so a restarted
  session continues the **same** W&B run instead of fragmenting metrics.

### Metrics logged per epoch
`train/loss`, `train/acc`, `val/acc`, `val/macro_f1`, `lr`, per-class `val/f1_<CLASS>`, and a W&B
confusion-matrix plot.

---

## 6. Evaluation, OOD ablation & Grad-CAM (`03_evaluate_explain.ipynb`)

- Set `RUN_NAME` to a checkpoint dir; it loads `best.pth`, rebuilds the model from the stored `cfg`.
- **In-dist metrics** via `run_inference` + `compute_metrics` + `plot_confusion_matrix` (saved PNG).
- **OOD ablation:** re-evaluates the *same* val set through `build_ood_aug` (strong stain shift) and
  reports the accuracy drop. Run it for both the `*_stain` and `*_nostain` checkpoints — the
  difference in drop is the headline result. Metrics dumped to `outputs/<run>_metrics.json`.
- **Grad-CAM:** `make_gradcam` returns `GradCAMResNet` (hooks `layer4[-1]`) or `GradCAMViT` (hooks
  `blocks[-1].norm1`, drops the CLS token, reshapes patch tokens to a square grid). `overlay_heatmap`
  composites onto the denormalized input. Always run Grad-CAM on the **uncompiled** module
  (`raw_module(model)`).

---

## 7. Demo (`04_gradio_demo.ipynb`)
Loads `best.pth` on CPU (`map_location="cpu"`), defines `classify(image, use_macenko)`, and serves a
`gr.Blocks` UI (prediction + confidence + 9-class `gr.Label` + Grad-CAM overlay) on port 7860,
`share=False`. CPU-only by design, to preserve the GPU budget.

---

## 8. Testing

The CPU smoke test (`/tmp/ft004_smoke.py` during development) execs `shared.ipynb`'s code cells and
exercises augmentations → dataset → dataloaders → `build_model` (timm, random init) → a train step →
metrics → Grad-CAM on dummy images. Re-run it after any change to `shared.ipynb` to catch
shape/import/version regressions without a GPU. It import-guards Phikon/transformers and
Macenko/torchstain so it runs with a minimal dep set.

GPU-only behaviour (real bf16 throughput, `torch.compile` on ROCm, Phikon download) can only be
verified on notebooks.amd.com.

---

## 9. Common pitfalls

| Symptom | Cause / fix |
|---|---|
| `DataLoader ... unexpected bus error` | Container shared memory too small → start with `--ipc=host` / bigger `--shm-size`, or lower `num_workers`. |
| Checkpoint keys prefixed `_orig_mod.` | Saved a compiled model directly → use `raw_module(model).state_dict()`. |
| W&B metrics split across runs after a restart | `wandb_run_id.txt` missing/changed → keep `PERSIST_DIR` stable; it pins the run id. |
| Re-downloads dataset every session | `PERSIST_DIR` not on the persistent mount, or `delete_zip` removed an interrupted extraction → verify `data/<name>/` has the 9 class folders. |
| Albumentations transform errors after a version bump | API drift (e.g. param renames) → re-run the smoke test; adjust `build_train_aug`. |
| Phikon download/auth error | Should be un-gated; if blocked, check network egress / HF status. Gated UNI/GigaPath need a token. |
| OOM on ViT-L | Drop `batch_size` (512→256), or enable grad checkpointing — though the 192GB MI300X should fit 512 in bf16. |

---

## 10. File map

```
finetuning_004/
├── shared.ipynb              # the library (edit via _build_notebooks.py)
├── 00_environment_check.ipynb
├── 01_data.ipynb
├── 02_train.ipynb
├── 03_evaluate_explain.ipynb
├── 04_gradio_demo.ipynb
├── _build_notebooks.py       # generator — source of truth for the notebooks
├── README.md                 # quickstart + 3-day schedule
├── DEVELOPER_GUIDE.md         # this file
└── .gitignore                # excludes data/, checkpoints/, outputs/, *.pth, wandb/
```
Runtime artifacts (`data/`, `checkpoints/`, `outputs/`) live under `PERSIST_DIR`, not in the repo.
