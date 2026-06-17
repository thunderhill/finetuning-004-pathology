"""One-off: MI300X systems micro-benchmark + second-mechanism OOD eval for ViT-L/16.
Reuses the exact model factory / datasets / augs from shared.ipynb (exec'd, magics stripped).
Outputs JSON to /workspace/shared/ft004/outputs/.
"""
import json, time, os, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

# ---- load shared.ipynb code (same defs used in training) ----
g = {}
nb = json.loads(Path("shared.ipynb").read_text())
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "\n".join(l for l in "".join(c["source"]).splitlines()
                    if not l.strip().startswith(("%", "!", "get_ipython")))
    if src.strip():
        exec(compile(src, "<shared>", "exec"), g)

build_model      = g["build_model"]
get_param_groups = g["get_param_groups"]
PathologyPatchDataset = g["PathologyPatchDataset"]
build_eval_aug   = g["build_eval_aug"]
compute_metrics  = g["compute_metrics"]
run_inference    = g["run_inference"]
load_checkpoint  = g["load_checkpoint"]
raw_module       = g["raw_module"]
CLASSES          = g["CLASSES"]
IMAGENET_MEAN, IMAGENET_STD = g["IMAGENET_MEAN"], g["IMAGENET_STD"]
device = torch.device("cuda")
g["set_seed"](42)

OUT = Path("/workspace/shared/ft004/outputs")
PERSIST = Path("/workspace/shared/ft004")
VAL_DIR = PERSIST / "data" / "CRC-VAL-HE-7K" / "CRC-VAL-HE-7K"
CKPT = Path("/checkpoints1/vit_l_stain/best.pth")

results = {}

# ============================================================
# PART 1 — MI300X systems micro-benchmark (ViT-L/16 training)
# ============================================================
print("\n=== PART 1: MI300X benchmark (ViT-L/16) ===", flush=True)
props = torch.cuda.get_device_properties(0)
dev_name = props.name or os.popen("rocm-smi --showproductname 2>/dev/null | grep -m1 -i 'Card series\\|Series\\|Product' ").read().strip()
total_hbm_gb = props.total_memory / 1024**3
print(f"device={dev_name!r} total_HBM={total_hbm_gb:.1f} GB", flush=True)

def bench(batch, compile_mode=None, warmup=6, iters=20):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = build_model({"model": "vit_l"}, num_classes=9)
    model.to(device).train()
    if compile_mode:
        model = torch.compile(model, mode=compile_mode)
    opt = torch.optim.AdamW(get_param_groups(model, {"model":"vit_l","lr":1e-4,"weight_decay":0.05,"backbone_lr_mult":0.1}))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    x = torch.randn(batch, 3, 224, 224, device=device)
    y = torch.randint(0, 9, (batch,), device=device)
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x); loss = lossf(out, y)
        loss.backward(); opt.step()
    for _ in range(warmup): step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak_alloc = torch.cuda.max_memory_allocated()/1024**3
    peak_resv  = torch.cuda.max_memory_reserved()/1024**3
    ips = batch * iters / dt
    nparams = sum(p.numel() for p in raw_module(model).parameters())/1e6
    del model, opt; torch.cuda.empty_cache()
    return dict(batch=batch, compile=compile_mode or "eager", img_per_s=round(ips,1),
                ms_per_step=round(1000*dt/iters,1), peak_alloc_gb=round(peak_alloc,1),
                peak_reserved_gb=round(peak_resv,1), params_M=round(nparams,1))

bench_rows = []
# headline: batch 512, eager vs compiled
for mode in ["eager", "reduce-overhead"]:
    cm = None if mode=="eager" else mode
    try:
        r = bench(512, cm)
        print(r, flush=True); bench_rows.append(r)
    except RuntimeError as e:
        print(f"batch=512 {mode}: {str(e)[:120]}", flush=True)
        bench_rows.append(dict(batch=512, compile=mode, error=str(e)[:120]))

compiled = next((r for r in bench_rows if r.get("compile")=="reduce-overhead" and "img_per_s" in r), None)
eager    = next((r for r in bench_rows if r.get("compile")=="eager" and "img_per_s" in r), None)
speedup  = round(compiled["img_per_s"]/eager["img_per_s"], 2) if (compiled and eager) else None

results["benchmark"] = dict(
    device=dev_name, total_hbm_gb=round(total_hbm_gb,1),
    runs=bench_rows, compile_speedup=speedup,
    fits_512=bool(compiled or eager),
    headline=dict(
        params_M=(eager or compiled or {}).get("params_M"),
        batch=512,
        img_per_s_compiled=(compiled or {}).get("img_per_s"),
        peak_reserved_gb=(compiled or eager or {}).get("peak_reserved_gb"),
        exceeds_80gb=( (compiled or eager or {}).get("peak_reserved_gb",0) > 80 ),
    ),
)
(OUT/"mi300x_benchmark.json").write_text(json.dumps(results["benchmark"], indent=2))
print("compile_speedup=", speedup, flush=True)

# ============================================================
# PART 2 — second-mechanism OOD (scanner-sim, NOT HED jitter)
# ============================================================
print("\n=== PART 2: second-mechanism OOD eval (vit_l_stain) ===", flush=True)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def build_scanner_ood(img_size=224):
    """Mechanistically distinct from training HED jitter: optical/sensor degradations
    + plain-RGB colour cast, NO HED decomposition."""
    return A.Compose([
        A.ImageCompression(quality_range=(30, 50), p=1.0),       # JPEG sensor artefacts
        A.GaussianBlur(blur_limit=(3, 7), p=0.8),                # focus/scanner blur
        A.Downscale(scale_range=(0.5, 0.75), p=0.7,
                    interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR}),
        A.RGBShift(r_shift_limit=30, g_shift_limit=20, b_shift_limit=30, p=1.0),  # RGB-space cast
        A.RandomGamma(gamma_limit=(70, 140), p=0.8),             # exposure/illumination
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

from torch.utils.data import DataLoader
g["set_seed"](42)
model = build_model({"model":"vit_l"}, num_classes=9).to(device)
ckpt = load_checkpoint(CKPT)
state = ckpt.get("model", ckpt.get("state_dict", ckpt))
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"loaded ckpt (missing={len(missing)} unexpected={len(unexpected)})", flush=True)

clean_ds = PathologyPatchDataset(VAL_DIR, classes=CLASSES, transform=build_eval_aug())
ood_ds   = PathologyPatchDataset(VAL_DIR, classes=CLASSES, transform=build_scanner_ood())
clean_ld = DataLoader(clean_ds, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
ood_ld   = DataLoader(ood_ds,   batch_size=512, shuffle=False, num_workers=8, pin_memory=True)

p, l = run_inference(model, clean_ld, device, torch.bfloat16)
m_clean = compute_metrics(l, p)
p, l = run_inference(model, ood_ld, device, torch.bfloat16)
m_ood = compute_metrics(l, p)

results["scanner_ood"] = dict(
    description="Second-mechanism OOD: JPEG + blur + downscale + RGB-space colour cast + gamma (no HED jitter).",
    clean=m_clean, scanner_ood=m_ood,
    clean_acc=round(m_clean["accuracy"],2), ood_acc=round(m_ood["accuracy"],2),
    drop=round(m_clean["accuracy"]-m_ood["accuracy"],2),
)
print("clean=%.2f scanner_ood=%.2f drop=%.2f" % (
    m_clean["accuracy"], m_ood["accuracy"], m_clean["accuracy"]-m_ood["accuracy"]), flush=True)
(OUT/"scanner_ood_metrics.json").write_text(json.dumps(results["scanner_ood"], indent=2))

print("\nDONE. Wrote mi300x_benchmark.json + scanner_ood_metrics.json")
