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

# ── Root path ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STAGE2_PTH = os.path.join(ROOT, "models", "stage2", "stage2_best.pth")

# ── Class registry ─────────────────────────────────────────────────────────────
# Must match exact order used during Stage 2A training (alphabetical / dataset order)
GUAVA_CLASSES = [
    "Guava_Canker",
    "Guava_healthy",
    "Guava_insect_bite",
    "Guava_Mummification",
    "Guava_multiple",
    "Guava_Rust",
    "Guava_scorch",
]

DISPLAY_NAMES = {
    "Guava_Canker":        "Canker",
    "Guava_healthy":       "Healthy",
    "Guava_insect_bite":   "Insect Bite",
    "Guava_Mummification": "Mummification",
    "Guava_multiple":      "Multiple Diseases",
    "Guava_Rust":          "Rust",
    "Guava_scorch":        "Leaf Scorch",
}

SEVERITY = {
    "Guava_Canker":        ("High",   "#dc2626"),
    "Guava_healthy":       ("None",   "#16a34a"),
    "Guava_insect_bite":   ("Medium", "#d97706"),
    "Guava_Mummification": ("High",   "#dc2626"),
    "Guava_multiple":      ("High",   "#dc2626"),
    "Guava_Rust":          ("High",   "#dc2626"),
    "Guava_scorch":        ("Medium", "#d97706"),
}

# Maps model class name → config.yaml class_query_map key
# Used by chain.py to build the retrieval query
RAG_CLASS_MAP = {
    "Guava_Canker":        "Guava_Canker",
    "Guava_healthy":       "Guava_healthy",
    "Guava_insect_bite":   "Guava_insect_bite",
    "Guava_Mummification": "Guava_Mummification",
    "Guava_multiple":      "Guava_multiple",
    "Guava_Rust":          "Guava_Rust",
    "Guava_scorch":        "Guava_scorch",
}


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the DeiT-tiny Stage 2A guava disease model.
    Cached via st.cache_resource — called only once per session.
    Returns: (model, device)
    """
    import timm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(STAGE2_PTH):
        st.error(f"Stage 2 checkpoint not found: {STAGE2_PTH}")
        st.info("Download stage2_best.pth from Drive → models/stage2/")
        st.stop()

    # num_classes=7 — must match Stage 2A training head exactly
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=7)
    ckpt = torch.load(STAGE2_PTH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    val_acc = ckpt.get("val_acc", "N/A")
    if isinstance(val_acc, float):
        print(f"[GuavaScan] Stage2A model loaded | val_acc: {val_acc:.4f}%")
    else:
        print(f"[GuavaScan] Stage2A model loaded | val_acc: {val_acc}")

    return model, device