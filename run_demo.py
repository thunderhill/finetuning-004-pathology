"""
Persistent Gradio demo launcher for FINETUNING_004.
Run:  python run_demo.py
Then open http://localhost:7860 in your browser.
Ctrl+C to stop.
"""
import json
import os
import sys
from pathlib import Path

# ── 1. Execute shared.ipynb cells into this namespace ──────────────────────────
_repo = Path(__file__).parent
_shared = _repo / "shared.ipynb"
assert _shared.exists(), f"shared.ipynb not found at {_shared}"

_nb = json.loads(_shared.read_text())
for _cell in _nb["cells"]:
    if _cell["cell_type"] == "code":
        _src = "".join(_cell["source"])
        # skip IPython magic lines
        _src = "\n".join(
            l for l in _src.splitlines()
            if not l.strip().startswith(("%", "get_ipython"))
        )
        if _src.strip():
            exec(_src, globals())  # noqa: S102

# ── 2. Load the best checkpoint ────────────────────────────────────────────────
import numpy as np
from PIL import Image

PERSIST_DIR = Path(os.environ.get("PERSIST_DIR", "/workspace/shared/ft004"))
RUN_NAME = "vit_l_stain"

ck = load_checkpoint(PERSIST_DIR / "checkpoints" / RUN_NAME / "best.pth", map_location="cpu")
cfg, classes = ck["cfg"], ck["classes"]

device = get_device()
model = build_model(cfg, num_classes=len(classes)).to(device)
raw_module(model).load_state_dict(ck["model"])
model.eval()

cam = make_gradcam(raw_module(model), cfg["model"])
eval_aug = build_eval_aug(cfg["img_size"])
print(f"\nDemo model ready: {cfg['model']} on {device}\n")

# ── 3. Inference function ──────────────────────────────────────────────────────
import torch

def classify(image, use_macenko=False):
    img = np.array(image.convert("RGB"))
    if use_macenko:
        try:
            img = MacenkoNormalizer(target_img=img)(img)
        except Exception as e:
            logger.warning(f"Macenko skipped ({e})")
    x = eval_aug(image=img)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x).softmax(1)[0]
    top = int(probs.argmax())
    label_desc = f"{classes[top]} — {CLASS_DESCRIPTIONS[classes[top]]}"
    dist = {f"{c} ({CLASS_DESCRIPTIONS[c]})": float(probs[i]) for i, c in enumerate(classes)}
    if cam is not None:
        raw = np.array(Image.fromarray(img).resize((cfg["img_size"], cfg["img_size"])))
        heat, _ = cam(x, class_idx=top)
        overlay = Image.fromarray(overlay_heatmap(raw, heat))
    else:
        overlay = image
    return label_desc, dist, overlay

# ── 4. Build and launch Gradio UI (blocking until Ctrl+C) ─────────────────────
_ensure("gradio")
import gradio as gr

with gr.Blocks(title="FINETUNING_004 — Stain-Robust Pathology Classifier") as demo:
    gr.Markdown(
        "# FINETUNING_004 — Stain-Robust Pathology Classifier\n"
        "Upload a 224×224 H&E patch → tissue class + confidence + Grad-CAM overlay.  \n"
        f"Model: **{cfg['model']}** fine-tuned on NCT-CRC-HE-100K (AMD MI300X)."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="H&E patch (drag & drop or browse)")
            mac = gr.Checkbox(label="Macenko normalize upload", value=False)
            btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            out_label = gr.Markdown(label="Prediction")
            out_dist  = gr.Label(num_top_classes=9, label="Class probabilities")
            out_cam   = gr.Image(type="pil", label="Grad-CAM overlay")
    btn.click(classify, inputs=[inp, mac], outputs=[out_label, out_dist, out_cam])

import socket as _socket

def _free_port(candidates):
    for p in candidates:
        s = _socket.socket()
        try: s.bind(("0.0.0.0", p)); return p
        except OSError: pass
        finally: s.close()
    return candidates[-1]

POD_NAME = "jupyter-hack-team-3403-260615153045-ba231015"
port = _free_port([8501, 8502, 8503, 8504, 8505])
root_path = f"/{POD_NAME}/proxy/{port}"

print(f"\nPress Ctrl+C to stop.")
print(f"Open: https://notebooks.amd.com{root_path}/\n")
demo.launch(server_name="0.0.0.0", server_port=port, root_path=root_path, share=False)
