# Execution Guide (step-by-step)

The simplest path to run everything on **notebooks.amd.com**. Run the notebooks **in order**.
Each step says what to do and what you should see.

---

## One-time setup (start of every GPU session)

**Step 1 — Upload the project**
Upload the `finetuning_004/` folder to your AMD Jupyter workspace (or `git clone` your repo).

**Step 2 — Open `00_environment_check.ipynb` → Run All**
- Installs the libraries (first time only; takes a few minutes).
- Should print `GPU available: True` and `bf16 native: True` with the MI300X name.
- Sets `PERSIST_DIR`. ⚠️ Edit this line if your persistent/mounted folder isn't
  `/workspace/shared/ft004` — otherwise data and checkpoints vanish when the session ends.
- At the W&B cell, paste your API key when prompted (or it falls back to offline mode).

✅ When you see the device name and `PERSIST_DIR = ...`, setup is done.

---

## Day 1 — get the data + bank a baseline

**Step 3 — Open `01_data.ipynb` → Run All**
- Downloads ~10 GB once (cached afterwards).
- Shows the class-count bar chart and an augmentation preview grid.

✅ You should see 9 classes, ~100k train / ~7,180 val images.

**Step 4 — Open `02_train.ipynb`**
1. In the **CONFIG cell**, set `model="resnet50"`. Leave everything else.
2. Run All.
3. Watch the log: loss drops, `val acc` rises each epoch. A `best.pth` is saved on improvement.

✅ Done when it prints `DONE resnet50_stain: best macro-F1 ...`.

**Step 5 — Train ViT-B (same notebook)**
- Change CONFIG `model="vit_b"`, Run All again.

> If a session ends mid-training, just **re-run `02_train.ipynb` with the same CONFIG** next day —
> it resumes automatically from `last.pth`.

---

## Day 2 — the hero model + the ablation

**Step 6 — Train ViT-L (the main model)**
In `02_train.ipynb` CONFIG: `model="vit_l"`, `batch_size=512`. Run All.

**Step 7 — Train the ablation pair**
Run `02_train.ipynb` twice more:
- `model="vit_l"`, `use_stain_aug=True`  (already done in Step 6 — skip if so)
- `model="vit_l"`, `use_stain_aug=False` (this is the "no stain aug" counterpart)

**Step 8 — (optional) Train the foundation-model head**
CONFIG `model="phikon"`. Run All. (Only the small head trains; it's fast.)

✅ After Day 2 you have checkpoints for `resnet50`, `vit_b`, `vit_l_stain`, `vit_l_nostain`,
`phikon` under `PERSIST_DIR/checkpoints/`.

---

## Day 3 — metrics, charts, demo

**Step 9 — Open `03_evaluate_explain.ipynb`**
1. Set `RUN_NAME="vit_l_stain"` → Run All.
   - Prints accuracy, macro-F1, per-class report.
   - Saves `outputs/vit_l_stain_confusion.png`, `_gradcam.png`, `_metrics.json`.
   - Prints the OOD stain-shift drop.
2. Repeat with `RUN_NAME="vit_l_nostain"` to get the ablation comparison.

**Step 10 — Open `04_gradio_demo.ipynb` → Run All**
- Launches the demo on port 7860. Upload an H&E patch → class + confidence + Grad-CAM.

✅ All figures/metrics land in `PERSIST_DIR/outputs/`.

---

## What to run when (cheat sheet)

| Notebook | When | Edit before running |
|---|---|---|
| `00_environment_check` | every session, first | `PERSIST_DIR` (once) |
| `01_data` | Day 1 only | — |
| `02_train` | Days 1–2, once per model | CONFIG cell: `model`, `use_stain_aug`, `batch_size` |
| `03_evaluate_explain` | Day 3 | `RUN_NAME` |
| `04_gradio_demo` | Day 3 | — |

## If something breaks
- **`unexpected bus error` in the data loader** → lower `num_workers` to `2` in CONFIG.
- **Out of memory on ViT-L** → set `batch_size=256`.
- **Re-downloads the dataset each time** → `PERSIST_DIR` isn't on the persistent mount; fix Step 2.
- **Training didn't finish before the session ended** → re-run the same CONFIG; it resumes.

For deeper details see `README.md` and `DEVELOPER_GUIDE.md`.
