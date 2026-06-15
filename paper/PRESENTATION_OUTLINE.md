# Presentation Outline — FINETUNING_004: Stain-Robust Pathology Classifier

Slide-by-slide content for the hackathon PPT. Paste each section into its own slide. All
numbers are measured (see `main.tex` / `/workspace/shared/ft004/outputs/*_metrics.json`).

---

## 1. Title
- **Stain-Robust Histology Classification at Scale**
- Full fine-tuning of Vision Transformers on a single AMD MI300X
- Team FINETUNING_004 — AMD Developer Challenge / TCS Ultimatix Hackathon

## 2. Elevator Pitch (What / Why / How)
- **What**: A 9-class colorectal-tissue classifier (NCT-CRC-HE-100K) that stays accurate
  when the input comes from a different lab/scanner.
- **Why**: Stain variation across labs/scanners is the #1 cause of pathology-AI models
  failing after deployment — high accuracy in validation, collapse in the real world.
- **How**: Aggressive HED (Hematoxylin-Eosin-DAB) colour-space augmentation during
  training forces the model to learn tissue *morphology*, not colour. We measure the
  effect directly with a controlled out-of-distribution (OOD) stain-shift test.

## 3. Problem Statement & Current Gaps
- Computational pathology models routinely report >95% accuracy on their training
  cohort, then degrade sharply on data from another institution.
- Root cause: differences in staining reagents, protocols, and scanner colour response —
  not a difference in the underlying tissue.
- Existing fixes (stain normalization at inference time) add a fragile preprocessing step
  and don't generalize to unseen appearances.
- **Gap**: few public pipelines *quantify* how much stain augmentation actually helps,
  with a reproducible before/after measurement.

## 4. Target Users / Stakeholders / Market
- Digital pathology software vendors (slide-scanner + LIS integrators)
- Hospital pathology departments adopting AI-assisted screening
- Pathology foundation-model developers needing a robustness benchmark
- Regulatory/QA teams who need evidence of cross-site generalization before clinical use

## 5. Mapped Hackathon Challenge
- *(Fill in: track name / problem statement from the hackathon portal this submission
  responds to — AI for Healthcare / Computer Vision on AMD MI300X, etc.)*

## 6. Solution Architecture / Workflow
- 5-notebook pipeline, fully reproducible, runs end-to-end on a single MI300X:
  1. `00_environment_check.ipynb` — ROCm/PyTorch/GPU sanity check
  2. `01_data.ipynb` — download + verify NCT-CRC-HE-100K / CRC-VAL-HE-7K
  3. `02_train.ipynb` — unified model factory + training loop (5 configs)
  4. `03_evaluate_explain.ipynb` — accuracy, macro-F1, confusion matrix, OOD ablation,
     Grad-CAM
  5. `04_gradio_demo.ipynb` — interactive web demo (upload → class + confidence +
     Grad-CAM)
- **Model factory** — one interface, two regimes:
  - *Full fine-tuning*: ResNet-50, ViT-B/16, ViT-L/16 (ImageNet-pretrained via `timm`)
  - *Frozen foundation model*: Phikon (pathology SSL) + trainable MLP head

## 7. AI Approach
- **Stain-space augmentation**: decompose each patch into Hematoxylin/Eosin/DAB channels,
  jitter each channel's scale and offset (θ=0.05 train, θ=0.15 for OOD eval), recompose
  to RGB — on top of strong spatial augmentation (rotations, flips, affine, elastic
  distortion).
- **Two transfer-learning regimes** compared head-to-head: end-to-end fine-tuning vs.
  frozen pathology foundation model + linear/MLP probe.
- **Controlled OOD protocol**: same validation images, re-rendered under a stronger
  synthetic stain shift — isolates the effect of augmentation from everything else.
- **Explainability**: Grad-CAM for both ViT (patch-token grid) and CNN (last conv stage)
  architectures.

## 8. Key Technologies & Datasets
- **Hardware**: AMD MI300X, 192 GB HBM3, ROCm
- **Framework**: PyTorch (ROCm build), native `bfloat16` autocast (no loss scaler),
  `torch.compile` (reduce-overhead mode)
- **Libraries**: `timm`, `transformers` (Phikon), Albumentations, Gradio
- **Datasets**: NCT-CRC-HE-100K (100,000 training patches), CRC-VAL-HE-7K (7,180
  patient-disjoint validation patches), 9 tissue classes (ADI, BACK, DEB, LYM, MUC, MUS,
  NORM, STR, TUM)

## 9. What Was Built During the Hackathon
- All 5 model configurations trained to convergence (12 epochs each) on a single MI300X
- Full evaluation suite: in-distribution + OOD accuracy/macro-F1, confusion matrices,
  Grad-CAM overlays for 4/5 models
- Interactive Gradio demo with live Grad-CAM and optional Macenko stain normalization
- Offline demo capture (one example per class, 9 classes, 89–91% confidence) for
  presentation/video use without a live server

## 10. Model Details & Performance Metrics
*(measured on CRC-VAL-HE-7K; see Table 2 in `main.tex`)*

| Model | In-dist Accuracy | In-dist Macro-F1 |
|---|---|---|
| ResNet-50 (stain aug) | 94.6% | 91.6% |
| ViT-B/16 (stain aug) | **97.4%** | 96.5% |
| ViT-L/16 (stain aug) | 97.3% | 96.1% |
| Phikon + MLP head (stain aug) | 96.7% | 95.0% |

- ViT-B/16 and ViT-L/16 both match or exceed published SSL baselines (DINO 95.9%,
  CTransPath 96.5%) with simple ImageNet-pretrained fine-tuning.

## 11. Headline Result — Stain-Robustness Ablation
*(measured; see Table 3/4 in `main.tex`)*

| ViT-L/16 training | Clean Acc | OOD (stain-shifted) Acc | Drop |
|---|---|---|---|
| Without stain aug | 97.9% | 53.9% | **−44.0 pts** |
| With stain aug | 97.3% | 84.4% | **−12.9 pts** |

- **Stain augmentation cuts the OOD accuracy drop by 3.4×** (44.0 → 12.9 points) for a
  cost of only 0.6 points of in-distribution accuracy.
- Across all 4 stain-augmented architectures, ViT-L/16 has the smallest OOD drop
  (12.9 pts), despite ViT-B/16 having marginally higher clean accuracy — robustness and
  peak accuracy are different axes.
- Suggested visual: `outputs/vit_l_stain_confusion.png` or a simple before/after bar
  chart of the table above.

## 12. Explainability — Grad-CAM
- Per-class Grad-CAM overlays (`outputs/vit_l_stain_gradcam.png`) show activations
  concentrating on tumour epithelium and stromal boundaries — consistent with how a
  pathologist would read the slide, not spurious background cues.
- Available live in the Gradio demo for any uploaded patch.

## 13. Systems / Performance on MI300X
- Native `bfloat16` training — no gradient scaler, no NaN-debugging loop common with
  `float16`
- `torch.compile` (reduce-overhead) for ViTs; eager fallback for BatchNorm models
- 192 GB HBM3 enables **full fine-tuning of a 307M-parameter ViT-L/16 at batch size up to
  512** with no gradient checkpointing or layer freezing — would require checkpointing or
  reduced batch size on an 80GB-class accelerator
- All 5 model configs (12 epochs each) trained and evaluated end-to-end during the
  hackathon window on a single GPU

## 14. Expected Impact
- A reproducible, quantified recipe for making pathology classifiers robust to the #1
  real-world deployment failure mode (stain variation) — applicable to any HE-stained
  tissue classification task, not just colorectal.
- Lowers the barrier for smaller labs/hospitals to deploy models trained elsewhere,
  without needing site-specific stain normalization pipelines.

## 15. Innovation / Differentiators
- Single unified model factory spanning full fine-tuning *and* frozen-foundation-model
  regimes — apples-to-apples comparison.
- Controlled, reproducible OOD stain-shift protocol with a direct before/after
  measurement (not just a claim).
- Demonstrates that a single MI300X removes the memory constraint that normally forces
  ViT-L fine-tuning into checkpointing/freezing tradeoffs.
- Grad-CAM auditing built into both the eval pipeline and the live demo.

## 16. Future Work
- Multi-centre / real (non-synthetic) stain-shift validation
- Whole-slide-image multiple-instance learning for slide-level diagnosis
- Evaluate gated pathology foundation models (UNI, Prov-GigaPath) in the same harness
- Grounded natural-language report generation: a small LLM layer that *phrases* the
  classifier's own outputs (class, confidence, Grad-CAM region, OOD-robustness flag) as
  a sentence — without introducing unverified diagnostic claims

## 17. Demo Video + Code
- Demo video: *(insert link once uploaded — see `DEMO_VIDEO_SCRIPT.md`)*
- Code repository / submission tarball: *(insert link — see code submission package)*
