"""
EcoNova — Leaf Disease Detection Dashboard
Streamlit app for Phase 2 presentation.

Architecture: DeiT-tiny Stage 2 model (6 guava classes)
Features:
  - Image upload + preprocessing
  - Top-3 predictions with confidence bars
  - Attention map heatmap overlay
  - RAG-style treatment advice

Run locally:
    streamlit run app/app.py

Run on Colab (after cloning repo and downloading model weights):
    !streamlit run app/app.py &
    from pyngrok import ngrok
    public_url = ngrok.connect(8501)
    print(public_url)
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import get_attention_maps
from src.preprocess import get_transforms, get_inverse_transform

# ── Model loader (shared with app4.py) ────────────────────────────────────────
from app.model_utils import load_model, GUAVA_CLASSES, DISPLAY_NAMES, SEVERITY

# ── Model weight paths ─────────────────────────────────────────────────────────
STAGE1_PTH = os.path.join(ROOT, "models", "stage1", "stage1_best.pth")
STAGE2_PTH = os.path.join(ROOT, "models", "stage2", "stage2_best.pth")

# ── Class definitions are imported from model_utils.py ───────────────────────

# ── Treatment advice ───────────────────────────────────────────────────────────
DISEASE_ADVICE = {
    "Guava_anthracnose": {
        "overview": (
            "Anthracnose is a fungal disease caused by Colletotrichum gloeosporioides. "
            "It produces dark, sunken lesions on leaves, stems, and fruits, and thrives "
            "in warm, humid conditions."
        ),
        "symptoms": [
            "Dark brown to black irregular spots on leaf surface",
            "Lesions with yellow halo in early stages",
            "Spots coalesce and cause premature leaf drop",
            "Sunken, dark lesions on fruits in severe cases",
        ],
        "treatment": [
            "Apply copper-based fungicide (Copper oxychloride 0.3%) every 10–14 days",
            "Use Mancozeb 0.2% or Carbendazim 0.1% as alternatives",
            "Remove and destroy all infected plant material immediately",
            "Avoid overhead irrigation — switch to drip irrigation",
        ],
        "prevention": [
            "Ensure good air circulation with proper canopy pruning",
            "Avoid wetting foliage during irrigation",
            "Apply preventive fungicide spray during monsoon season",
            "Use disease-free planting material",
        ],
        "note": "⚠️ Consult a local agricultural extension officer for region-specific guidance.",
    },
    "Guava_healthy": {
        "overview": (
            "The leaf appears healthy with no visible signs of disease or pest damage. "
            "Continue regular monitoring and preventive care."
        ),
        "symptoms": [
            "No lesions, spots, or discoloration observed",
            "Uniform green color throughout the leaf",
            "Normal leaf texture and shape",
        ],
        "treatment": [
            "No treatment required at this time",
            "Maintain regular watering schedule",
            "Apply balanced NPK fertilizer as per schedule",
        ],
        "prevention": [
            "Monitor plants weekly for early signs of disease",
            "Keep orchard floor clean — remove fallen leaves",
            "Maintain soil pH between 6.0 and 7.5",
            "Apply preventive neem oil spray once a month",
        ],
        "note": "✅ Plant appears healthy. Continue standard maintenance practices.",
    },
    "Guava_insect_bite": {
        "overview": (
            "Insect bite damage is caused by feeding insects such as fruit flies, "
            "thrips, or mealybugs. Irregular holes and necrotic patches are characteristic. "
            "Secondary fungal infections may follow."
        ),
        "symptoms": [
            "Irregular holes or torn edges on leaf surface",
            "Small necrotic (brown/black) spots at feeding sites",
            "Silvery or bronze discoloration from thrips feeding",
            "Sticky honeydew deposits indicating mealybug or aphid presence",
        ],
        "treatment": [
            "Spray Neem oil (5 ml/L water) — effective against soft-bodied insects",
            "Apply Imidacloprid 0.3 ml/L for severe infestations",
            "Use yellow sticky traps to monitor and reduce adult fly populations",
            "Introduce natural predators such as ladybugs if available",
        ],
        "prevention": [
            "Regularly inspect the undersides of leaves for egg clusters",
            "Remove and destroy heavily infested branches",
            "Avoid excessive nitrogen fertilization — promotes soft tissue",
            "Maintain orchard hygiene — remove fruit waste promptly",
        ],
        "note": "⚠️ Consult a local agricultural extension officer for region-specific guidance.",
    },
    "Guava_multiple": {
        "overview": (
            "Multiple concurrent diseases detected. This indicates a stressed plant with "
            "compromised immunity, likely under combined fungal and pest pressure. "
            "Immediate intervention is required."
        ),
        "symptoms": [
            "Multiple lesion types visible — fungal spots + insect damage",
            "Widespread leaf discoloration and necrosis",
            "Possible stem cankers or bark lesions",
            "Significant defoliation risk",
        ],
        "treatment": [
            "Apply broad-spectrum fungicide (Mancozeb 0.25%) immediately",
            "Follow with systemic insecticide (Chlorpyrifos 2 ml/L) after 3 days",
            "Remove all severely affected branches — sterilize pruning tools",
            "Foliar spray of micronutrients (Zinc + Boron) to boost immunity",
        ],
        "prevention": [
            "Conduct thorough disease scouting every 5–7 days",
            "Implement integrated pest management (IPM) program",
            "Improve drainage — waterlogged soil increases disease susceptibility",
            "Avoid plant stress — maintain consistent irrigation and nutrition",
        ],
        "note": "🚨 Multiple conditions detected. Consult a local agricultural extension officer for a combined treatment plan.",
    },
    "Guava_scorch": {
        "overview": (
            "Leaf scorch appears as browning of leaf margins and tips, caused by water "
            "stress, nutrient deficiency (especially potassium), or root damage. "
            "It is not infectious but indicates physiological stress."
        ),
        "symptoms": [
            "Brown, dry margins and leaf tips",
            "Yellowing of leaf tissue between veins (interveinal chlorosis)",
            "Brittle, papery texture at affected margins",
            "Symptoms progress from older leaves to younger leaves",
        ],
        "treatment": [
            "Improve irrigation — ensure consistent moisture, avoid drought stress",
            "Apply potassium sulfate (K2SO4) 2–3 kg per tree as soil drench",
            "Foliar spray of 0.5% potassium nitrate (KNO3) for quick uptake",
            "Check and treat root health — address compaction or root rot",
        ],
        "prevention": [
            "Mulch around the base to retain soil moisture",
            "Conduct soil test annually — maintain adequate K and Ca levels",
            "Avoid over-fertilization with nitrogen — causes K imbalance",
            "Provide shade during extreme heat if possible",
        ],
        "note": "⚠️ Consult a local agricultural extension officer for region-specific guidance.",
    },
    "Guava_yld": {
        "overview": (
            "Yellow Leaf Disease in guava is associated with phytoplasma infection or "
            "severe micronutrient deficiency (iron, magnesium, zinc). It causes progressive "
            "yellowing and is a significant yield threat if untreated."
        ),
        "symptoms": [
            "Uniform yellowing starting from leaf margins inward",
            "Veins may remain green initially (vein clearing)",
            "Stunted new growth and reduced leaf size",
            "Premature defoliation in advanced stages",
        ],
        "treatment": [
            "Soil application of ferrous sulfate (FeSO4) 50g per tree",
            "Foliar spray of 0.5% magnesium sulfate (MgSO4) every 2 weeks",
            "Apply chelated micronutrient mix (Fe + Zn + Mn) as foliar spray",
            "If phytoplasma suspected: apply Tetracycline antibiotic (consult specialist)",
        ],
        "prevention": [
            "Test soil pH — iron unavailability increases in alkaline soils (pH > 7.5)",
            "Apply organic compost annually to improve micronutrient availability",
            "Control leafhoppers — primary vector of phytoplasma",
            "Remove and destroy severely infected trees to prevent spread",
        ],
        "note": "🚨 If symptoms spread rapidly across the orchard, phytoplasma infection is suspected. Contact your local agriculture department.",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

# @st.cache_resource(show_spinner=False)
# def load_model():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if not os.path.exists(STAGE2_PTH):
#         st.error(f"Stage 2 checkpoint not found: {STAGE2_PTH}")
#         st.info("Download stage2_best.pth from Drive → models/stage2/")
#         st.stop()
#     import timm
#     model = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=6)
#     ckpt = torch.load(STAGE2_PTH, map_location=device)
#     model.load_state_dict(ckpt["model_state_dict"], strict=True)
#     model.to(device)
#     model.eval()
#     return model, device
# load_model() is imported from app.model_utils — see import above


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(pil_image: Image.Image) -> torch.Tensor:
    transform = get_transforms("val")
    tensor = transform(pil_image.convert("RGB"))
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict(model, tensor: torch.Tensor, device):
    tensor = tensor.to(device)
    outputs = model(tensor)
    probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(GUAVA_CLASSES[i], float(probs[i])) for i in top3_idx]
    return top3, probs


# ══════════════════════════════════════════════════════════════════════════════
# ATTENTION MAP
# ══════════════════════════════════════════════════════════════════════════════

def generate_attention_overlay(model, tensor: torch.Tensor, device, pil_image: Image.Image) -> Image.Image:
    tensor = tensor.to(device)
    attn_maps = get_attention_maps(model, tensor)
    last_attn = attn_maps[-1]
    cls_attn = last_attn[0, :, 0, 1:]
    cls_attn = cls_attn.mean(dim=0).cpu().numpy()
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
    attn_map = cls_attn.reshape(14, 14)
    attn_pil = Image.fromarray((attn_map * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    attn_array = np.array(attn_pil) / 255.0
    colormap = cm.get_cmap("jet")
    heatmap_rgba = colormap(attn_array)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb)
    orig_resized = pil_image.convert("RGB").resize((224, 224), Image.LANCZOS)
    return Image.blend(orig_resized, heatmap_pil, alpha=0.45)


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def confidence_bar_html(label: str, confidence: float, color: str, is_top: bool = False) -> str:
    pct = confidence * 100
    bar_color = color if is_top else "#94a3b8"
    bg = f"{color}0d" if is_top else "rgba(0,0,0,0.02)"
    border = f"2px solid {color}55" if is_top else "1px solid #e2e8f0"
    label_color = "#0f172a" if is_top else "#334155"

    return f"""
    <div style="
        background: {bg};
        border: {border};
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
    ">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span style="font-family:'Lora',serif; font-size:18px;
                         color:{label_color}; font-weight:800;">{label}</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:18px;
                         color:{color}; font-weight:800;">{pct:.1f}%</span>
        </div>
        <div style="background:#e2e8f0; border-radius:6px; height:10px; overflow:hidden;">
            <div style="width:{pct:.1f}%; height:100%; background:{bar_color};
                        border-radius:6px; transition:width 0.6s ease;"></div>
        </div>
    </div>
    """


def advice_card_html(icon: str, title: str, items: list, accent: str) -> str:
    items_html = "".join(
        f'<li style="margin-bottom:10px; color:#1e293b; font-size:17px; font-weight:800; line-height:1.7;">{item}</li>'
        for item in items
    )
    return f"""
    <div style="
        background: #ffffff;
        border: 1px solid {accent}35;
        border-left: 5px solid {accent};
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    ">
        <div style="font-family:'Lora',serif; font-size:20px;
                    font-weight:800; color:{accent}; margin-bottom:16px;">
            {icon} {title}
        </div>
        <ul style="margin:0; padding-left:20px;">
            {items_html}
        </ul>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE THRESHOLD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_confidence_zone(confidence: float) -> dict:
    """
    Returns zone metadata based on top-1 confidence score.
      Green Zone  : >= 0.85  → high confidence, show full results + treatment
      Gray Zone   : 0.60–0.85 → uncertain, show limited results + caution banner
      Red Zone    : < 0.60   → low confidence, block results, show expert warning
    """
    if confidence >= 0.85:
        return {
            "zone": "green",
            "label": "High Confidence",
            "icon": "✅",
            "color": "#16a34a",
            "bg": "#f0fdf4",
            "border": "#bbf7d0",
            "text_color": "#14532d",
            "badge_bg": "#dcfce7",
            "message": "The model is confident in this diagnosis. You may proceed with the recommended actions below.",
            "show_results": True,
            "show_treatment": True,
        }
    elif confidence >= 0.60:
        return {
            "zone": "gray",
            "label": "Uncertain — Cross-Check Advised",
            "icon": "⚠️",
            "color": "#64748b",
            "bg": "#f8fafc",
            "border": "#cbd5e1",
            "text_color": "#1e293b",
            "badge_bg": "#e2e8f0",
            "message": (
                "The model is not fully confident in this result. "
                "Try retaking the photo in better lighting, ensure the leaf fills the frame, "
                "or cross-check symptoms against a second source before taking any action."
            ),
            "show_results": True,
            "show_treatment": False,
        }
    else:
        return {
            "zone": "red",
            "label": "Low Confidence — Diagnosis Unreliable",
            "icon": "🚨",
            "color": "#dc2626",
            "bg": "#fff1f2",
            "border": "#fecdd3",
            "text_color": "#7f1d1d",
            "badge_bg": "#ffe4e6",
            "message": (
                "The model cannot make a reliable diagnosis from this image. "
                "Please upload a clearer, well-lit photo of a single leaf, "
                "or consult a local agricultural expert directly."
            ),
            "show_results": False,
            "show_treatment": False,
        }


def zone_banner_html(zone: dict, confidence: float) -> str:
    pct = confidence * 100
    return f"""
    <div style="
        background: {zone['bg']};
        border: 2px solid {zone['border']};
        border-left: 6px solid {zone['color']};
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    ">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px; flex-wrap:wrap;">
            <span style="font-size:22px;">{zone['icon']}</span>
            <span style="
                font-family:'DM Sans',sans-serif; font-size:16px; font-weight:800;
                color:{zone['color']}; letter-spacing:0.5px;
            ">{zone['label']}</span>
            <span style="
                margin-left:auto;
                background:{zone['badge_bg']};
                border:1px solid {zone['border']};
                border-radius:8px;
                padding:4px 14px;
                font-family:'JetBrains Mono',monospace;
                font-size:15px;
                font-weight:800;
                color:{zone['color']};
            ">{pct:.1f}%</span>
        </div>
        <p style="
            font-family:'DM Sans',sans-serif; font-size:15px; font-weight:800;
            color:{zone['text_color']}; margin:0; line-height:1.7;
        ">{zone['message']}</p>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EcoNova — Leaf Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600;700;800&family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 17px;
    font-weight: 700;
}
.stApp {
    background: #f0f7f0;
}
.main .block-container {
    padding: 1.5rem 2.5rem 4rem 2.5rem;
    max-width: 1440px;
}

/* ── Responsive layout ────────────────────────────── */
@media (max-width: 900px) {
    .main .block-container {
        padding: 1rem 1rem 3rem 1rem !important;
    }
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
}

/* ── Hide Streamlit chrome ────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Upload widget ────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #2d6a4f !important;
    border: 2px dashed rgba(255, 255, 255, 0.5) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(255, 255, 255, 0.9) !important;
}

/* Drag and drop text */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFileData"] {
    color: rgba(255, 255, 255, 0.8) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 17px !important;
    font-weight: 800 !important;
}

/* Icons inside uploader */
[data-testid="stFileUploader"] section svg,
[data-testid="stFileUploader"] section svg path {
    fill: rgba(255, 255, 255, 0.8) !important;
}

/* File name and size after upload */
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p,
[data-testid="stFileUploader"] div[class*="uploadedFile"] span,
[data-testid="stFileUploader"] div[class*="uploadedFile"] p,
[data-testid="stFileUploader"] div[class*="uploadedFile"] small,
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] section p {
    color: rgba(255, 255, 255, 0.8) !important;
    font-weight: 800 !important;
}

/* Browse files button */
[data-testid="stFileUploader"] button {
    background: rgba(255, 255, 255, 0.15) !important;
    color: rgba(255, 255, 255, 0.8) !important;
    border: 2px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    transition: background 0.2s, border-color 0.2s;
}
[data-testid="stFileUploader"] button:hover {
    background: rgba(255, 255, 255, 0.25) !important;
    border-color: rgba(255, 255, 255, 0.9) !important;
    color: #ffffff !important;
}

/* ── Checkbox ─────────────────────────────────────── */
[data-testid="stCheckbox"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 17px !important;
    font-weight: 800 !important;
    color: #334155 !important;
}

/* ── Spinner ──────────────────────────────────────── */
.stSpinner > div {
    border-top-color: #16a34a !important;
}

/* ── Image caption ────────────────────────────────── */
[data-testid="stImage"] p {
    font-size: 16px !important;
    font-weight: 800 !important;
    color: #64748b !important;
}

/* ── Global bold enforcement ──────────────────────── */
p, li, span, div, label {
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER — fully centered, updated subtitle, no badges
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="
    background: linear-gradient(135deg, #dcfce7 0%, #f0fdf4 60%, #ecfdf5 100%);
    border: 1px solid #bbf7d0;
    border-radius: 20px;
    padding: 22px 36px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(22,163,74,0.10);
">
    <div style="
        position: absolute; top: -50px; right: -50px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(22,163,74,0.13) 0%, transparent 70%);
        border-radius: 50%;
    "></div>
    <div style="
        position: absolute; bottom: -40px; left: -40px;
        width: 180px; height: 180px;
        background: radial-gradient(circle, rgba(22,163,74,0.08) 0%, transparent 70%);
        border-radius: 50%;
    "></div>
    <div style="display: flex; align-items: center; gap: 18px;">
        <div style="font-size: 52px; line-height:1; flex-shrink:0;">🌿</div>
        <div>
            <h1 style="
                font-family: 'Lora', serif;
                font-size: clamp(2rem, 4vw, 3rem);
                color: #14532d;
                margin: 0 0 4px 0;
                font-weight: 800;
                letter-spacing: -1px;
                line-height: 1;
            ">EcoNova</h1>
            <p style="
                font-family: 'DM Sans', sans-serif;
                font-size: clamp(0.95rem, 2vw, 1.2rem);
                color: #166534;
                font-weight: 800;
                margin: 0;
                letter-spacing: 0.5px;
            ">Leaf Disease Detection using ViT</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL (cached)
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Loading EcoNova model..."):
    model, device = load_model()

device_label = "GPU (CUDA)" if str(device) == "cuda" else "CPU"
st.markdown(f"""
<div style="text-align:right; font-family:'JetBrains Mono',monospace;
            font-size:14px; color:#16a34a; font-weight:800;
            margin-top:-14px; margin-bottom:18px;">
    ● Model loaded · {device_label}
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT — two columns
# ══════════════════════════════════════════════════════════════════════════════

col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown("""
    <div style="font-family:'DM Sans',sans-serif; font-size:14px; letter-spacing:2px;
                color:#16a34a; font-weight:800; margin-bottom:14px; text-transform:uppercase;">
        📤 Upload Leaf Image
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a guava leaf photo here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    pil_image = None

    if uploaded is not None:
        pil_image = Image.open(uploaded).convert("RGB")

        # Constrained image — 65% width, centered
        _, img_center, _ = st.columns([0.175, 0.65, 0.175])
        with img_center:
            st.image(pil_image, use_container_width=True, caption="Uploaded leaf image")

        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        show_attention = st.checkbox("🔥 Show Attention Heatmap", value=False)

        if show_attention:
            with st.spinner("Generating attention map..."):
                tensor_att = preprocess(pil_image)
                overlay = generate_attention_overlay(model, tensor_att, device, pil_image)
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.markdown('<p style="font-size:16px; font-weight:800; color:#475569; text-align:center; margin-bottom:6px; font-family:\'DM Sans\',sans-serif;">Original</p>', unsafe_allow_html=True)
                st.image(pil_image.resize((224, 224)), use_container_width=True)
            with img_col2:
                st.markdown('<p style="font-size:16px; font-weight:800; color:#475569; text-align:center; margin-bottom:6px; font-family:\'DM Sans\',sans-serif;">Attention Heatmap</p>', unsafe_allow_html=True)
                st.image(overlay, use_container_width=True)
            st.markdown("""
            <div style="background:#f0fdf4; border:1px solid #bbf7d0;
                        border-radius:10px; padding:14px 18px; margin-top:12px;">
                <p style="font-size:16px; font-weight:800; color:#15803d; font-family:'DM Sans',sans-serif; margin:0;">
                    🔬 <strong>Explainability:</strong> Red/warm regions show where the model focused its attention for the prediction.
                </p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            background: #ffffff;
            border: 2px dashed #bbf7d0;
            border-radius: 16px;
            height: 290px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 14px;
            box-shadow: 0 1px 6px rgba(0,0,0,0.04);
        ">
            <div style="font-size: 54px; opacity:0.22;">🍃</div>
            <p style="color:#64748b; font-size:20px; font-weight:800; margin:0; font-family:'DM Sans',sans-serif;">
                Upload a leaf photo to begin diagnosis
            </p>
            <p style="color:#94a3b8; font-size:15px; font-weight:800; margin:0; font-family:'JetBrains Mono',monospace;">
                JPG · PNG · WEBP · max 200MB
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Results (auto-runs on upload, no button needed)
# ══════════════════════════════════════════════════════════════════════════════

with col_right:
    st.markdown("""
    <div style="font-family:'DM Sans',sans-serif; font-size:14px; letter-spacing:2px;
                color:#16a34a; font-weight:800; margin-bottom:14px; text-transform:uppercase;">
        📊 Disease Analysis
    </div>
    """, unsafe_allow_html=True)

    if pil_image is None:
        st.markdown("""
        <div style="
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 64px 24px;
            text-align: center;
            box-shadow: 0 1px 6px rgba(0,0,0,0.04);
        ">
            <div style="font-size:46px; opacity:0.16; margin-bottom:16px;">🔬</div>
            <p style="color:#94a3b8; font-size:20px; font-weight:800; font-family:'DM Sans',sans-serif; margin:0;">
                Results will appear here after upload
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Analysing leaf..."):
            tensor = preprocess(pil_image)
            top3, all_probs = predict(model, tensor, device)

        top_class, top_conf = top3[0]
        display_name = DISPLAY_NAMES[top_class]
        severity_label, severity_color = SEVERITY[top_class]
        is_healthy = top_class == "Guava_healthy"
        card_accent = "#16a34a" if is_healthy else severity_color

        # ── Confidence zone ───────────────────────────────────
        zone = get_confidence_zone(top_conf)
        st.markdown(zone_banner_html(zone, top_conf), unsafe_allow_html=True)

        if not zone["show_results"]:
            st.markdown("""
            <div style="
                background: #ffffff;
                border: 2px dashed #fecdd3;
                border-radius: 14px;
                padding: 44px 24px;
                text-align: center;
            ">
                <div style="font-size:48px; margin-bottom:16px;">🔴</div>
                <p style="font-family:'Lora',serif; font-size:22px; font-weight:800;
                           color:#7f1d1d; margin:0 0 12px 0;">Diagnosis Withheld</p>
                <p style="font-family:'DM Sans',sans-serif; font-size:16px; font-weight:800;
                           color:#991b1b; margin:0; line-height:1.7;">
                    Upload a clearer image or contact a local agricultural expert for a reliable diagnosis.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {card_accent}14 0%, {card_accent}06 100%);
                border: 2px solid {card_accent}45;
                border-radius: 16px;
                padding: 28px 32px;
                margin-bottom: 22px;
                box-shadow: 0 3px 14px rgba(0,0,0,0.08);
            ">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap;">
                    <div>
                        <div style="font-family:'DM Sans',sans-serif; font-size:14px;
                                    letter-spacing:2px; color:#64748b; margin-bottom:10px; font-weight:800;
                                    text-transform:uppercase;">Detected Disease</div>
                        <div style="font-family:'Lora',serif; font-size:clamp(1.8rem, 3vw, 2.5rem);
                                    color:#0f172a; line-height:1.15; font-weight:800;">{display_name}</div>
                    </div>
                    <div style="
                        background: {severity_color}1a;
                        border: 2px solid {severity_color}65;
                        border-radius: 12px;
                        padding: 12px 22px;
                        text-align: center;
                        flex-shrink: 0;
                    ">
                        <div style="font-size:13px; color:{severity_color}; letter-spacing:2px;
                                    font-family:'JetBrains Mono',monospace; font-weight:800;
                                    text-transform:uppercase;">Severity</div>
                        <div style="font-size:22px; color:{severity_color}; font-weight:800;
                                    font-family:'DM Sans',sans-serif; margin-top:4px;">{severity_label}</div>
                    </div>
                </div>
                <div style="margin-top:22px;">
                    <div style="font-size:14px; color:#64748b; margin-bottom:10px;
                                font-family:'JetBrains Mono',monospace; font-weight:800;
                                letter-spacing:2px; text-transform:uppercase;">Confidence</div>
                    <div style="display:flex; align-items:center; gap:16px;">
                        <div style="flex:1; background:#e2e8f0; border-radius:8px; height:16px; overflow:hidden;">
                            <div style="width:{top_conf*100:.1f}%; height:100%;
                                        background: linear-gradient(90deg, {card_accent}, {card_accent}cc);
                                        border-radius:8px;"></div>
                        </div>
                        <span style="font-family:'JetBrains Mono',monospace; font-size:26px;
                                     color:{card_accent}; font-weight:800; min-width:80px;">
                            {top_conf*100:.1f}%
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="font-family:'DM Sans',sans-serif; font-size:14px; letter-spacing:2px;
                        color:#64748b; font-weight:800; margin-bottom:14px; text-transform:uppercase;">
                Top 3 Predictions
            </div>
            """, unsafe_allow_html=True)

            colors_top3 = [card_accent, "#2563eb", "#7c3aed"]
            for i, (cls, conf) in enumerate(top3):
                st.markdown(
                    confidence_bar_html(DISPLAY_NAMES[cls], conf, colors_top3[i], is_top=(i == 0)),
                    unsafe_allow_html=True
                )

            st.markdown(f"""
            <div style="
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 16px 20px;
                margin-top: 12px;
                display: flex;
                gap: 28px;
                flex-wrap: wrap;
            ">
                <p style="font-size:15px; font-weight:800; color:#64748b; font-family:'JetBrains Mono',monospace; margin:0;">
                    Model: DeiT-tiny Stage 2
                </p>
                <p style="font-size:15px; font-weight:800; color:#64748b; font-family:'JetBrains Mono',monospace; margin:0;">
                    Device: {device_label}
                </p>
                <p style="font-size:15px; font-weight:800; color:#64748b; font-family:'JetBrains Mono',monospace; margin:0;">
                    Classes: 6
                </p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TREATMENT ADVICE — full width below, zone-aware
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<hr style='border-color:#d1fae5; margin: 2rem 0;'>", unsafe_allow_html=True)

st.markdown("""
<div style="font-family:'DM Sans',sans-serif; font-size:14px; letter-spacing:2px;
            color:#854d0e; font-weight:800; margin-bottom:22px; text-transform:uppercase;">
    🤖 AI Treatment Advice
</div>
""", unsafe_allow_html=True)

if pil_image is None:
    st.markdown("""
    <div style="
        background: #ffffff;
        border: 2px dashed #e2e8f0;
        border-radius: 14px;
        padding: 44px;
        text-align: center;
        color: #94a3b8;
        font-family: 'DM Sans', sans-serif;
        font-size: 20px;
        font-weight: 800;
    ">
        Upload a leaf image to see treatment recommendations
    </div>
    """, unsafe_allow_html=True)

else:
    if not zone["show_treatment"]:
        # Gray zone: symptoms only, no treatment/prevention
        if zone["show_results"]:
            advice = DISEASE_ADVICE[top_class]
            st.markdown(f"""
            <div style="
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 24px 30px;
                margin-bottom: 22px;
                font-family: 'DM Sans', sans-serif;
                font-size: 18px;
                font-weight: 800;
                color: #1e293b;
                line-height: 1.8;
                box-shadow: 0 1px 6px rgba(0,0,0,0.04);
            ">
                {advice['overview']}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(advice_card_html(
                "🔍", "Symptoms to Watch — Cross-check these manually",
                advice["symptoms"], "#64748b"
            ), unsafe_allow_html=True)
            st.markdown("""
            <div style="
                background: #f8fafc;
                border: 2px solid #cbd5e1;
                border-left: 6px solid #64748b;
                border-radius: 12px;
                padding: 20px 24px;
                margin-top: 8px;
            ">
                <p style="font-family:'DM Sans',sans-serif; font-size:16px; font-weight:800;
                           color:#334155; margin:0 0 8px 0;">
                    ⚠️ <strong>Treatment withheld — confidence too low for a reliable recommendation.</strong>
                </p>
                <p style="font-family:'DM Sans',sans-serif; font-size:15px; font-weight:800;
                           color:#475569; margin:0; line-height:1.7;">
                    Retake the photo in natural daylight with the leaf fully in frame, then re-upload.
                    Alternatively, consult a local agricultural expert before applying any treatment.
                </p>
            </div>
            """, unsafe_allow_html=True)
        # Red zone: full block
        else:
            st.markdown("""
            <div style="
                background: #fff1f2;
                border: 2px dashed #fecdd3;
                border-radius: 14px;
                padding: 44px;
                text-align: center;
            ">
                <p style="font-family:'DM Sans',sans-serif; font-size:18px; font-weight:800;
                           color:#991b1b; margin:0;">
                    🚨 No treatment advice available — diagnosis confidence is too low.
                    Please upload a clearer image or contact an agricultural expert.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Green zone: full treatment
    else:
        advice = DISEASE_ADVICE[top_class]
        st.markdown(f"""
        <div style="
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 24px 30px;
            margin-bottom: 22px;
            font-family: 'DM Sans', sans-serif;
            font-size: 18px;
            font-weight: 800;
            color: #1e293b;
            line-height: 1.8;
            box-shadow: 0 1px 6px rgba(0,0,0,0.04);
        ">
            {advice['overview']}
        </div>
        """, unsafe_allow_html=True)

        adv_col1, adv_col2, adv_col3 = st.columns(3)
        with adv_col1:
            st.markdown(advice_card_html("🔍", "Symptoms to Watch", advice["symptoms"], "#dc2626"), unsafe_allow_html=True)
        with adv_col2:
            st.markdown(advice_card_html("💊", "Treatment Steps", advice["treatment"], "#16a34a"), unsafe_allow_html=True)
        with adv_col3:
            st.markdown(advice_card_html("🛡️", "Prevention Measures", advice["prevention"], "#2563eb"), unsafe_allow_html=True)

        st.markdown(f"""
        <div style="
            background: #fefce8;
            border: 2px solid #fde68a;
            border-radius: 10px;
            padding: 18px 24px;
            font-family: 'DM Sans', sans-serif;
            font-size: 17px;
            font-weight: 800;
            color: #854d0e;
            margin-top: 6px;
        ">
            {advice['note']}
        </div>
        """, unsafe_allow_html=True)

# FOOTER — clean and minimal, no accuracy spans
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="
    text-align: center;
    margin-top: 56px;
    padding-top: 24px;
    border-top: 1px solid #d1fae5;
    font-family: 'DM Sans', sans-serif;
    font-size: 16px;
    font-weight: 800;
    color: #64748b;
">
    EcoNova &nbsp;·&nbsp; Leaf Disease Detection using ViT &nbsp;·&nbsp; Phase 2
</div>
""", unsafe_allow_html=True)