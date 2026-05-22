"""
GuavaScan — RAG-Integrated Leaf Disease Detection Dashboard
app/app4.py

New dashboard built on app3.py's visual design, with full RAG integration.
- ViT model loaded from app/model_utils.py (shared, cached)
- RAG chain from rag/chain.py (stateless, cached)
- Confidence gating: >= 0.60 -> full RAG | 0.40-0.60 -> general | < 0.40 -> skip

Run:
    streamlit run app/app4.py
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
import streamlit as st
import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import get_attention_maps
from src.preprocess import get_transforms, get_inverse_transform
from app.model_utils import load_model, GUAVA_CLASSES, DISPLAY_NAMES, SEVERITY

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_config():
    cfg_path = os.path.join(ROOT, "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

# ══════════════════════════════════════════════════════════════════════════════
# RAG CHAIN (cached — initialised once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_rag_components():
    """Load retriever and config once. Chain is lightweight and built per-call."""
    from rag.retriever import get_retriever
    cfg = load_config()
    retriever = get_retriever(cfg)
    return cfg, retriever

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
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

def generate_attention_overlay(model, tensor, device, pil_image):
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

def confidence_bar_html(label, confidence, color, is_top=False):
    pct = confidence * 100
    bar_color = color if is_top else "#94a3b8"
    bg = f"{color}0d" if is_top else "rgba(0,0,0,0.02)"
    border = f"2px solid {color}55" if is_top else "1px solid #e2e8f0"
    label_color = "#0f172a" if is_top else "#334155"
    return f"""
    <div style="background:{bg}; border:{border}; border-radius:12px;
                padding:14px 18px; margin-bottom:12px;">
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

def section_card_html(icon, title, content_md, accent):
    """Render an advisory section as a styled HTML card."""
    return f"""
    <div style="background:#ffffff; border:1px solid {accent}35;
                border-left:5px solid {accent}; border-radius:12px;
                padding:20px 24px; margin-bottom:14px;
                box-shadow:0 2px 8px rgba(0,0,0,0.06);">
        <div style="font-family:'Lora',serif; font-size:18px;
                    font-weight:800; color:{accent}; margin-bottom:12px;">
            {icon} {title}
        </div>
        <div style="font-family:'DM Sans',sans-serif; font-size:16px;
                    font-weight:700; color:#1e293b; line-height:1.8;
                    white-space:pre-wrap;">{content_md}</div>
    </div>
    """

def zone_banner_html(zone, confidence):
    pct = confidence * 100
    return f"""
    <div style="background:{zone['bg']}; border:2px solid {zone['border']};
                border-left:6px solid {zone['color']}; border-radius:14px;
                padding:18px 22px; margin-bottom:20px;
                box-shadow:0 2px 8px rgba(0,0,0,0.05);">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px; flex-wrap:wrap;">
            <span style="font-size:22px;">{zone['icon']}</span>
            <span style="font-family:'DM Sans',sans-serif; font-size:16px; font-weight:800;
                color:{zone['color']};">{zone['label']}</span>
            <span style="margin-left:auto; background:{zone['badge_bg']};
                border:1px solid {zone['border']}; border-radius:8px;
                padding:4px 14px; font-family:'JetBrains Mono',monospace;
                font-size:15px; font-weight:800; color:{zone['color']};">{pct:.1f}%</span>
        </div>
        <p style="font-family:'DM Sans',sans-serif; font-size:15px; font-weight:800;
            color:{zone['text_color']}; margin:0; line-height:1.7;">{zone['message']}</p>
    </div>
    """

def get_confidence_zone(confidence, cfg):
    rag_thresh = cfg["confidence_gate"]["rag_threshold"]
    fall_thresh = cfg["confidence_gate"]["fallback_threshold"]
    if confidence >= rag_thresh:
        return {
            "zone": "green", "label": "High Confidence",
            "icon": "✅", "color": "#16a34a", "bg": "#f0fdf4",
            "border": "#bbf7d0", "text_color": "#14532d", "badge_bg": "#dcfce7",
            "message": "The model is confident in this diagnosis. Full RAG advisory loaded below.",
            "rag_mode": "full",
        }
    elif confidence >= fall_thresh:
        return {
            "zone": "amber", "label": "Uncertain — General Guidance Shown",
            "icon": "⚠️", "color": "#d97706", "bg": "#fffbeb",
            "border": "#fde68a", "text_color": "#78350f", "badge_bg": "#fef3c7",
            "message": (
                "Low confidence prediction — showing general guava care guidance. "
                "Retake the photo in better lighting for a disease-specific advisory."
            ),
            "rag_mode": "fallback",
        }
    else:
        return {
            "zone": "red", "label": "Confidence Too Low",
            "icon": "🚨", "color": "#dc2626", "bg": "#fff1f2",
            "border": "#fecdd3", "text_color": "#7f1d1d", "badge_bg": "#ffe4e6",
            "message": "Confidence too low for reliable advisory. Please upload a clearer image.",
            "rag_mode": "skip",
        }

def render_rag_panel(rag_result: dict):
    """Parse and render the LLM advisory into structured st.tabs."""
    answer = rag_result["answer"]
    sources = rag_result["sources"]

    # Render as markdown directly — LLM output is already structured
    st.markdown(answer)

    # Source attribution
    with st.expander("📄 Knowledge Base Sources", expanded=False):
        if sources:
            for s in sources:
                st.markdown(
                    f"- **`{s['file']}`** — *{s['section']}*"
                )
        else:
            st.info("No source metadata available.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="GuavaScan — RAG Disease Advisory",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;700;800&family=DM+Sans:wght@400;500;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 17px;
    font-weight: 700;
}
.stApp { background: #f0f7f0; }
.main .block-container {
    padding: 1.5rem 2.5rem 4rem 2.5rem;
    max-width: 1440px;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

[data-testid="stFileUploader"] {
    background: #2d6a4f !important;
    border: 2px dashed rgba(255,255,255,0.5) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: rgba(255,255,255,0.85) !important;
    font-weight: 800 !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(255,255,255,0.15) !important;
    color: rgba(255,255,255,0.85) !important;
    border: 2px solid rgba(255,255,255,0.5) !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
}
.stSpinner > div { border-top-color: #16a34a !important; }
p, li, span, div, label { font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="
    background: linear-gradient(135deg, #dcfce7 0%, #f0fdf4 60%, #ecfdf5 100%);
    border: 1px solid #bbf7d0; border-radius: 20px;
    padding: 22px 36px; margin-bottom: 30px;
    box-shadow: 0 2px 16px rgba(22,163,74,0.10);
    position: relative; overflow: hidden;
">
    <div style="display:flex; align-items:center; gap:18px;">
        <div style="font-size:52px; line-height:1; flex-shrink:0;">🌿</div>
        <div>
            <h1 style="font-family:'Lora',serif; font-size:clamp(2rem,4vw,3rem);
                color:#14532d; margin:0 0 4px 0; font-weight:800;
                letter-spacing:-1px; line-height:1;">GuavaScan</h1>
            <p style="font-family:'DM Sans',sans-serif; font-size:clamp(0.95rem,2vw,1.1rem);
                color:#166534; font-weight:800; margin:0; letter-spacing:0.5px;">
                Guava Leaf Disease Detection · ViT + RAG Advisory Pipeline
            </p>
        </div>
        <div style="margin-left:auto; text-align:right;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:12px;
                color:#16a34a; font-weight:800; background:#dcfce7;
                padding:4px 12px; border-radius:8px; border:1px solid #bbf7d0;">
                RAG · Gemini 1.5 Flash
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ══════════════════════════════════════════════════════════════════════════════

cfg = load_config()

with st.spinner("Loading GuavaScan model..."):
    model, device = load_model()

with st.spinner("Initialising RAG pipeline..."):
    try:
        rag_cfg, retriever = load_rag_components()
        rag_ready = True
    except Exception as e:
        rag_ready = False
        rag_error = str(e)

device_label = "GPU (CUDA)" if str(device) == "cuda" else "CPU"
rag_status = "✅ RAG Ready" if rag_ready else "⚠️ RAG Offline"
rag_color  = "#16a34a" if rag_ready else "#d97706"

st.markdown(f"""
<div style="display:flex; gap:20px; justify-content:flex-end;
            margin-top:-14px; margin-bottom:18px; flex-wrap:wrap;">
    <span style="font-family:'JetBrains Mono',monospace; font-size:14px;
        color:#16a34a; font-weight:800;">● Model · {device_label}</span>
    <span style="font-family:'JetBrains Mono',monospace; font-size:14px;
        color:{rag_color}; font-weight:800;">{rag_status}</span>
</div>
""", unsafe_allow_html=True)

if not rag_ready:
    st.warning(f"RAG pipeline could not initialise: {rag_error}. Run `python rag/ingest.py` first.")

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — prevent re-querying identical predictions
# ══════════════════════════════════════════════════════════════════════════════

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None   # (predicted_class, confidence)
if "last_rag_result" not in st.session_state:
    st.session_state.last_rag_result = None

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
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
                st.markdown('<p style="font-size:16px; font-weight:800; color:#475569; text-align:center; font-family:\'DM Sans\',sans-serif;">Original</p>', unsafe_allow_html=True)
                st.image(pil_image.resize((224, 224)), use_container_width=True)
            with img_col2:
                st.markdown('<p style="font-size:16px; font-weight:800; color:#475569; text-align:center; font-family:\'DM Sans\',sans-serif;">Attention Heatmap</p>', unsafe_allow_html=True)
                st.image(overlay, use_container_width=True)
            st.markdown("""
            <div style="background:#f0fdf4; border:1px solid #bbf7d0;
                        border-radius:10px; padding:14px 18px; margin-top:12px;">
                <p style="font-size:15px; font-weight:800; color:#15803d;
                    font-family:'DM Sans',sans-serif; margin:0;">
                    🔬 <strong>Explainability:</strong> Warm regions show where the ViT model focused its attention.
                </p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#ffffff; border:2px dashed #bbf7d0;
                    border-radius:16px; height:290px; display:flex;
                    flex-direction:column; align-items:center;
                    justify-content:center; gap:14px;
                    box-shadow:0 1px 6px rgba(0,0,0,0.04);">
            <div style="font-size:54px; opacity:0.22;">🍃</div>
            <p style="color:#64748b; font-size:20px; font-weight:800; margin:0;
                font-family:'DM Sans',sans-serif;">Upload a leaf photo to begin</p>
            <p style="color:#94a3b8; font-size:15px; font-weight:800; margin:0;
                font-family:'JetBrains Mono',monospace;">JPG · PNG · WEBP</p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — ViT Prediction
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
        <div style="background:#ffffff; border:1px solid #e2e8f0;
                    border-radius:16px; padding:64px 24px; text-align:center;
                    box-shadow:0 1px 6px rgba(0,0,0,0.04);">
            <div style="font-size:46px; opacity:0.16; margin-bottom:16px;">🔬</div>
            <p style="color:#94a3b8; font-size:20px; font-weight:800;
                font-family:'DM Sans',sans-serif; margin:0;">
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
        card_accent = "#16a34a" if top_class == cfg.get("healthy_class") else severity_color

        zone = get_confidence_zone(top_conf, cfg)
        st.markdown(zone_banner_html(zone, top_conf), unsafe_allow_html=True)

        if zone["rag_mode"] != "skip":
            # Disease prediction card
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{card_accent}14 0%,{card_accent}06 100%);
                border:2px solid {card_accent}45; border-radius:16px;
                padding:28px 32px; margin-bottom:22px;
                box-shadow:0 3px 14px rgba(0,0,0,0.08);">
                <div style="display:flex; justify-content:space-between;
                    align-items:flex-start; gap:12px; flex-wrap:wrap;">
                    <div>
                        <div style="font-family:'DM Sans',sans-serif; font-size:13px;
                            letter-spacing:2px; color:#64748b; margin-bottom:10px;
                            font-weight:800; text-transform:uppercase;">Detected Condition</div>
                        <div style="font-family:'Lora',serif;
                            font-size:clamp(1.8rem,3vw,2.5rem);
                            color:#0f172a; line-height:1.15; font-weight:800;">{display_name}</div>
                    </div>
                    <div style="background:{severity_color}1a;
                        border:2px solid {severity_color}65; border-radius:12px;
                        padding:12px 22px; text-align:center; flex-shrink:0;">
                        <div style="font-size:12px; color:{severity_color};
                            letter-spacing:2px; font-family:'JetBrains Mono',monospace;
                            font-weight:800; text-transform:uppercase;">Severity</div>
                        <div style="font-size:22px; color:{severity_color};
                            font-weight:800; font-family:'DM Sans',sans-serif;
                            margin-top:4px;">{severity_label}</div>
                    </div>
                </div>
                <div style="margin-top:22px;">
                    <div style="font-size:13px; color:#64748b; margin-bottom:10px;
                        font-family:'JetBrains Mono',monospace; font-weight:800;
                        letter-spacing:2px; text-transform:uppercase;">Confidence</div>
                    <div style="display:flex; align-items:center; gap:16px;">
                        <div style="flex:1; background:#e2e8f0; border-radius:8px;
                            height:16px; overflow:hidden;">
                            <div style="width:{top_conf*100:.1f}%; height:100%;
                                background:linear-gradient(90deg,{card_accent},{card_accent}cc);
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

            # Top-3 predictions
            st.markdown("""
            <div style="font-family:'DM Sans',sans-serif; font-size:13px; letter-spacing:2px;
                color:#64748b; font-weight:800; margin-bottom:14px; text-transform:uppercase;">
                Top 3 Predictions
            </div>
            """, unsafe_allow_html=True)
            colors_top3 = [card_accent, "#2563eb", "#7c3aed"]
            for i, (cls, conf) in enumerate(top3):
                st.markdown(
                    confidence_bar_html(DISPLAY_NAMES[cls], conf, colors_top3[i], is_top=(i == 0)),
                    unsafe_allow_html=True,
                )

        else:
            # Red zone — block output
            st.markdown("""
            <div style="background:#fff1f2; border:2px dashed #fecdd3;
                        border-radius:14px; padding:44px 24px; text-align:center;">
                <div style="font-size:48px; margin-bottom:16px;">🔴</div>
                <p style="font-family:'Lora',serif; font-size:22px; font-weight:800;
                    color:#7f1d1d; margin:0 0 12px 0;">Diagnosis Withheld</p>
                <p style="font-family:'DM Sans',sans-serif; font-size:16px; font-weight:800;
                    color:#991b1b; margin:0; line-height:1.7;">
                    Upload a clearer, well-lit image of a single guava leaf.
                </p>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RAG ADVISORY PANEL — full width
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<hr style='border-color:#d1fae5; margin:2rem 0;'>", unsafe_allow_html=True)

st.markdown("""
<div style="font-family:'DM Sans',sans-serif; font-size:14px; letter-spacing:2px;
            color:#854d0e; font-weight:800; margin-bottom:22px; text-transform:uppercase;">
    🤖 RAG-Powered Agronomic Advisory
</div>
""", unsafe_allow_html=True)

if pil_image is None:
    st.markdown("""
    <div style="background:#ffffff; border:2px dashed #e2e8f0; border-radius:14px;
                padding:44px; text-align:center; color:#94a3b8;
                font-family:'DM Sans',sans-serif; font-size:20px; font-weight:800;">
        Upload a leaf image to see the RAG advisory
    </div>
    """, unsafe_allow_html=True)

elif zone["rag_mode"] == "skip":
    st.markdown("""
    <div style="background:#fff1f2; border:2px dashed #fecdd3; border-radius:14px;
                padding:44px; text-align:center;">
        <p style="font-family:'DM Sans',sans-serif; font-size:18px; font-weight:800;
                  color:#991b1b; margin:0;">
            🚨 No advisory available — confidence too low. Please upload a clearer image.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif not rag_ready:
    st.error("RAG pipeline not initialised. Run `python rag/ingest.py` then restart the app.")

else:
    # Determine what to query
    if zone["rag_mode"] == "full":
        query_class = top_class
        query_conf  = top_conf
        mode_label  = f"Advisory for: **{display_name}**"
    else:
        # Fallback — general agronomy query
        query_class = "__fallback__"
        query_conf  = top_conf
        mode_label  = "⚠️ Low confidence — showing general guava care guidance"
        st.warning(mode_label)

    # ── Session state guard — avoid re-querying same prediction ───────────────
    current_pred = (query_class, round(query_conf, 3))

    if st.session_state.last_prediction == current_pred and st.session_state.last_rag_result:
        rag_result = st.session_state.last_rag_result
        st.caption("📌 Advisory loaded from session cache — same prediction as before.")
    else:
        with st.spinner("🔍 Retrieving from knowledge base and generating advisory..."):
            try:
                from rag.chain import run_rag_chain

                if zone["rag_mode"] == "full":
                    rag_result = run_rag_chain(top_class, top_conf, rag_cfg, retriever)
                else:
                    # Fallback: use general agronomy as context
                    fallback_query = rag_cfg.get("fallback_query", "guava orchard management prevention")
                    # Manually build a fallback result using retriever
                    chunks = retriever.retrieve(fallback_query)
                    from rag.chain import _format_context
                    from rag.prompts import HEALTHY_PROMPT
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    from langchain_core.output_parsers import StrOutputParser
                    from langchain_core.messages import HumanMessage
                    import os
                    from dotenv import load_dotenv
                    load_dotenv(os.path.join(ROOT, ".env"))
                    api_key = os.getenv("GEMINI_API_KEY")
                    llm = ChatGoogleGenerativeAI(
                        model=rag_cfg["llm_model"],
                        google_api_key=api_key,
                        temperature=0.2,
                        max_output_tokens=800,
                    )
                    context = _format_context(chunks)
                    prompt_val = HEALTHY_PROMPT.format(
                        confidence=query_conf * 100,
                        context=context,
                    )
                    response = llm.invoke([HumanMessage(content=prompt_val)])
                    answer = StrOutputParser().invoke(response)
                    sources = [{"file": c["source"], "section": c["section"]} for c in chunks]
                    rag_result = {"answer": answer, "sources": sources}

                st.session_state.last_prediction = current_pred
                st.session_state.last_rag_result  = rag_result

            except Exception as e:
                st.error(f"RAG chain error: {e}")
                st.info("Ensure GEMINI_API_KEY is set in .env and the knowledge base is indexed.")
                rag_result = None

    if rag_result:
        render_rag_panel(rag_result)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align:center; margin-top:56px; padding-top:24px;
            border-top:1px solid #d1fae5; font-family:'DM Sans',sans-serif;
            font-size:15px; font-weight:800; color:#64748b;">
    GuavaScan &nbsp;·&nbsp; ViT (DeiT-tiny) + RAG Advisory &nbsp;·&nbsp;
    Gemini 1.5 Flash &nbsp;·&nbsp; ChromaDB · BM25 · sentence-transformers
</div>
""", unsafe_allow_html=True)
