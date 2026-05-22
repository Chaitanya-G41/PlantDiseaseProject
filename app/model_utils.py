"""
model_utils.py
--------------
Shared ViT model loader for GuavaScan.
Used by app3.py (via import) and app4.py.
Do NOT add any app logic here — loader only.
"""

import os
import sys
import torch
import streamlit as st

# ── Root path ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STAGE2_PTH = os.path.join(ROOT, "models", "stage2", "stage2_best.pth")

# ── Class registry ─────────────────────────────────────────────────────────────
GUAVA_CLASSES = [
    "Guava_anthracnose",
    "Guava_healthy",
    "Guava_insect_bite",
    "Guava_multiple",
    "Guava_scorch",
    "Guava_yld",
]

DISPLAY_NAMES = {
    "Guava_anthracnose": "Anthracnose",
    "Guava_healthy":     "Healthy",
    "Guava_insect_bite": "Insect Bite",
    "Guava_multiple":    "Multiple Diseases",
    "Guava_scorch":      "Leaf Scorch",
    "Guava_yld":         "Yellow Leaf Disease",
}

SEVERITY = {
    "Guava_anthracnose": ("High",   "#dc2626"),
    "Guava_healthy":     ("None",   "#16a34a"),
    "Guava_insect_bite": ("Medium", "#d97706"),
    "Guava_multiple":    ("High",   "#dc2626"),
    "Guava_scorch":      ("Medium", "#d97706"),
    "Guava_yld":         ("High",   "#dc2626"),
}


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the DeiT-tiny Stage 2 guava disease model.
    Cached via st.cache_resource — called only once per session.
    Returns: (model, device)
    """
    import timm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(STAGE2_PTH):
        st.error(f"Stage 2 checkpoint not found: {STAGE2_PTH}")
        st.info("Download stage2_best.pth from Drive → models/stage2/")
        st.stop()

    model = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=6)
    ckpt = torch.load(STAGE2_PTH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    val_acc = ckpt.get("val_acc", "N/A")
    print(f"[GuavaScan] Model loaded | val_acc: {val_acc:.2f}%" if isinstance(val_acc, float) else f"[GuavaScan] Model loaded")
    return model, device
