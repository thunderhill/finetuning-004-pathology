# Demo Video Script — FINETUNING_004: Stain-Robust Pathology Classifier

Target length: ~2:40. Screen-record the Gradio UI from `04_gradio_demo.ipynb`
(`demo.launch(server_name="0.0.0.0", server_port=<port>)`) plus a couple of static slides
/ figures. Sample patches to upload live are listed in Step 3 below (paths from
`outputs/demo_offline_examples.json`, all correctly classified at 89–91% confidence).

---

### 0:00 – 0:15 — Intro
**Visual**: Title slide (Slide 1 of `paper/PRESENTATION_OUTLINE.md`).
**Voiceover**:
> "This is FINETUNING_004 — a stain-robust pathology image classifier, fully fine-tuned
> on a single AMD MI300X. The problem we're solving: pathology AI models that score 95%+
> in validation often collapse when run on slides from a different lab or scanner,
> because of differences in staining colour — not tissue content."

### 0:15 – 0:30 — Pipeline Overview
**Visual**: Slide 6 (Solution Architecture) — the 5-notebook pipeline diagram.
**Voiceover**:
> "Our pipeline is five notebooks: environment check, data prep, training, evaluation and
> explainability, and this live demo. We trained five models end-to-end on a single
> MI300X — three full fine-tunes (ResNet-50, ViT-B/16, ViT-L/16), one frozen
> pathology-foundation-model probe (Phikon), and an ablation of ViT-L/16 without our
> stain augmentation."

### 0:30 – 1:30 — Live Demo
**Visual**: Gradio UI, screen-recorded live.
**Voiceover / actions**:
> "Let's try it. I'll upload a few patches from the held-out validation set — each from a
> different tissue class."

Upload these patches one at a time (each is correctly classified, 89–91% confidence):
1. `data/CRC-VAL-HE-7K/CRC-VAL-HE-7K/TUM/TUM-TCGA-AAHIDTWA.tif` — tumour epithelium
2. `data/CRC-VAL-HE-7K/CRC-VAL-HE-7K/STR/STR-TCGA-AAMALCER.tif` — cancer-associated stroma
3. `data/CRC-VAL-HE-7K/CRC-VAL-HE-7K/LYM/LYM-TCGA-AAWGSCHH.tif` — lymphocytes

For each: point out the predicted class + confidence, the full 9-class probability bar
chart, and the Grad-CAM overlay.
> "Notice the Grad-CAM overlay lights up the tumour epithelium for the TUM patch, and the
> connective-tissue bands for the STR patch — the model is attending to the structures a
> pathologist would actually look at."

### 1:30 – 1:50 — Macenko Normalization Toggle
**Visual**: Check the "Macenko normalize upload" box, re-run one of the uploads.
**Voiceover**:
> "There's also an optional Macenko stain-normalization toggle for severely
> out-of-distribution inputs — but the headline result is that our model doesn't need it
> at inference time, because robustness is trained in."

### 1:50 – 2:20 — The Headline Result
**Visual**: Cut to Slide 11 (Stain-Robustness Ablation) and/or
`outputs/vit_l_stain_confusion.png`.
**Voiceover**:
> "Here's the core measurement. We took the same ViT-L/16 architecture and trained it
> twice — once without our HED stain augmentation, once with it — then tested both on a
> synthetically stain-shifted copy of the validation set. Without augmentation, accuracy
> collapses from 97.9% to 53.9% — a 44-point drop. With augmentation, it only drops to
> 84.4% — a 12.9-point drop. That's a 3.4x reduction in the robustness gap, for a cost of
> just 0.6 points of clean accuracy. Across all four augmented architectures, ViT-L/16
> has the smallest drop of all — making it our most deployment-robust model, even though
> ViT-B/16 edges it out slightly on clean accuracy."

### 2:20 – 2:40 — Wrap-up
**Visual**: Slide 13 (Systems/Performance) or back to title slide.
**Voiceover**:
> "All of this — five models, full evaluation, Grad-CAM, and this demo — ran end-to-end
> on a single AMD MI300X, using native bfloat16 with no loss scaling and torch.compile
> for graph execution. The 192GB of HBM3 let us fully fine-tune a 307-million-parameter
> ViT-L/16 at batch size 512 with no checkpointing tricks. Code, configs, and the full
> writeup are linked below."

**Visual**: End card with code repository link / submission tarball reference.
