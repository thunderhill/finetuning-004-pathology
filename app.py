"""
FINETUNING_004 — Stain-Robust Pathology Classifier
Streamlit demo for AMD notebook proxy.

Launch:
    streamlit run app.py --server.port 8501 --server.headless true \
        --server.enableCORS false --server.enableXsrfProtection false

Access:
    https://notebooks.amd.com/jupyter-hack-team-3403-260615153045-ba231015/proxy/8501/
"""
import json, os, sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ── Load shared.ipynb into this namespace ──────────────────────────────────────
_repo = Path(__file__).parent
_shared = _repo / "shared.ipynb"
assert _shared.exists(), f"shared.ipynb not found at {_shared}"

_nb = json.loads(_shared.read_text())
for _cell in _nb["cells"]:
    if _cell["cell_type"] == "code":
        _src = "".join(_cell["source"])
        _src = "\n".join(
            l for l in _src.splitlines()
            if not l.strip().startswith(("%", "get_ipython"))
        )
        if _src.strip():
            exec(_src, globals())  # noqa: S102

# ── Model loading (cached across reruns) ──────────────────────────────────────
PERSIST_DIR = Path(os.environ.get("PERSIST_DIR", "/workspace/shared/ft004"))
RUN_NAME = "vit_l_stain"

@st.cache_resource(show_spinner="Loading ViT-L/16 checkpoint…")
def _load():
    import torch
    ck = load_checkpoint(PERSIST_DIR / "checkpoints" / RUN_NAME / "best.pth", map_location="cpu")
    cfg, classes = ck["cfg"], ck["classes"]
    device = get_device()
    model = build_model(cfg, num_classes=len(classes)).to(device)
    raw_module(model).load_state_dict(ck["model"])
    model.eval()
    cam = make_gradcam(raw_module(model), cfg["model"])
    aug = build_eval_aug(cfg["img_size"])
    return model, cam, aug, cfg, classes, device

model, cam, eval_aug, cfg, classes, device = _load()

# ── Inference ──────────────────────────────────────────────────────────────────
def classify(pil_image, use_macenko=False):
    import torch
    img = np.array(pil_image.convert("RGB"))
    if use_macenko:
        try:
            img = MacenkoNormalizer(target_img=img)(img)
        except Exception as e:
            st.warning(f"Macenko skipped: {e}")
    x = eval_aug(image=img)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x).softmax(1)[0]
    top = int(probs.argmax())
    pred_class = classes[top]
    confidence = float(probs[top])
    dist = {c: float(probs[i]) for i, c in enumerate(classes)}
    if cam is not None:
        raw = np.array(Image.fromarray(img).resize((cfg["img_size"], cfg["img_size"])))
        heat, _ = cam(x, class_idx=top)
        overlay = Image.fromarray(overlay_heatmap(raw, heat))
    else:
        overlay = pil_image
    return pred_class, confidence, dist, overlay

# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FINETUNING_004 — Pathology Classifier",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 FINETUNING_004 — Stain-Robust Pathology Classifier")
st.caption(
    f"Model: **{cfg['model']}** fine-tuned on NCT-CRC-HE-100K · AMD MI300X · "
    "9-class colorectal tissue classification"
)
st.divider()

col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.subheader("Upload")
    uploaded = st.file_uploader(
        "H&E patch (224×224 recommended)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )
    use_macenko = st.checkbox("Macenko stain-normalize upload", value=False,
                              help="Apply Macenko normalization for severely out-of-distribution inputs.")
    classify_btn = st.button("Classify", type="primary", use_container_width=True)

    if uploaded:
        st.image(uploaded, caption="Uploaded patch", use_container_width=True)

with col_right:
    if uploaded and classify_btn:
        pil_img = Image.open(uploaded)
        with st.spinner("Running ViT-L + Grad-CAM…"):
            pred_class, confidence, dist, overlay = classify(pil_img, use_macenko)

        desc = CLASS_DESCRIPTIONS[pred_class]
        st.subheader("Prediction")
        st.markdown(
            f"<div style='font-size:1.6rem; font-weight:700; color:#1f77b4'>"
            f"{pred_class}</div>"
            f"<div style='font-size:1.1rem; color:#555'>{desc}</div>"
            f"<div style='font-size:1rem; margin-top:4px'>Confidence: "
            f"<b>{confidence*100:.1f}%</b></div>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.subheader("Class probabilities")
        import pandas as pd
        df = pd.DataFrame(
            {"Probability": dist},
        ).sort_values("Probability", ascending=False)
        st.bar_chart(df, height=260)

        st.divider()
        st.subheader("Grad-CAM overlay")
        st.image(overlay, caption="Regions the model attended to", use_container_width=True)

    elif not uploaded:
        st.info("Upload an H&E patch on the left and click **Classify**.")
