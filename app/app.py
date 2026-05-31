"""
GuavaScan — RAG-Integrated Leaf Disease Detection Dashboard
app/app4.py  (v3)

Changes:
- Removed attention heatmap feature entirely
- Removed emoji from all UI text
- Removed model/RAG status pills from header area
- Increased font size of RAG section header
- Renamed "AI-Generated Advisory" to "AGRONOMY ADVISORY"
- Live generation progress pills (real-time per section)
- Fallback max_output_tokens raised to 2048
- RAG_SECTIONS updated to 4-section structure
"""

import os
import sys
import re
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import streamlit as st
import yaml
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv(os.path.join(ROOT, ".env"))

from src.preprocess import get_transforms
from model_utils import load_model, GUAVA_CLASSES, DISPLAY_NAMES, SEVERITY

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_config():
    cfg_path = os.path.join(ROOT, "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

# ══════════════════════════════════════════════════════════════════════════════
# RAG COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_rag_components():
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

# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def confidence_bar_html(label, confidence, color, is_top=False):
    pct = confidence * 100
    bar_color = color if is_top else "#94a3b8"
    bg        = f"{color}12" if is_top else "#f8fafc"
    border    = f"2px solid {color}55" if is_top else "1px solid #e2e8f0"
    return f"""
    <div style="background:{bg}; border:{border}; border-radius:12px;
                padding:14px 18px; margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span style="font-family:'Lora',serif; font-size:18px;
                         color:#0f172a; font-weight:800;">{label}</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:18px;
                         color:{color}; font-weight:800;">{pct:.1f}%</span>
        </div>
        <div style="background:#e2e8f0; border-radius:6px; height:10px; overflow:hidden;">
            <div style="width:{pct:.1f}%; height:100%; background:{bar_color};
                        border-radius:6px;"></div>
        </div>
    </div>
    """

def zone_banner_html(zone, confidence):
    pct = confidence * 100
    return f"""
    <div style="background:{zone['bg']}; border:2px solid {zone['border']};
                border-left:6px solid {zone['color']}; border-radius:14px;
                padding:18px 22px; margin-bottom:20px;">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px; flex-wrap:wrap;">
            <span style="font-family:'DM Sans',sans-serif; font-size:16px; font-weight:800;
                color:{zone['color']};">{zone['label']}</span>
            <span style="margin-left:auto; background:{zone['badge_bg']};
                border:1px solid {zone['border']}; border-radius:8px;
                padding:4px 14px; font-family:'JetBrains Mono',monospace;
                font-size:15px; font-weight:800; color:{zone['color']};">{pct:.1f}%</span>
        </div>
        <p style="font-family:'DM Sans',sans-serif; font-size:15px; font-weight:700;
            color:{zone['text_color']}; margin:0; line-height:1.7;">{zone['message']}</p>
    </div>
    """

def get_confidence_zone(confidence, cfg):
    rag_thresh  = cfg["confidence_gate"]["rag_threshold"]
    fall_thresh = cfg["confidence_gate"]["fallback_threshold"]
    if confidence >= rag_thresh:
        return {
            "zone": "green", "label": "High Confidence — Full Advisory",
            "color": "#16a34a", "bg": "#f0fdf4",
            "border": "#bbf7d0", "text_color": "#14532d", "badge_bg": "#dcfce7",
            "message": "Model is confident in this diagnosis. Full advisory loaded below.",
            "rag_mode": "full",
        }
    elif confidence >= fall_thresh:
        return {
            "zone": "amber", "label": "Uncertain — General Guidance",
            "color": "#d97706", "bg": "#fffbeb",
            "border": "#fde68a", "text_color": "#78350f", "badge_bg": "#fef3c7",
            "message": "Low confidence — showing general guava care guidance. Retake in better lighting.",
            "rag_mode": "fallback",
        }
    else:
        return {
            "zone": "red", "label": "Confidence Too Low",
            "color": "#dc2626", "bg": "#fff1f2",
            "border": "#fecdd3", "text_color": "#7f1d1d", "badge_bg": "#ffe4e6",
            "message": "Confidence too low for reliable advisory. Please upload a clearer image.",
            "rag_mode": "skip",
        }

# ── RAG section definitions — 4-section structure ────────────────────────────
RAG_SECTIONS = [
    ("Diagnosis Summary",         "🔬", "#0e7490"),
    ("Key Symptoms",              "🩺", "#7c3aed"),
    ("Treatment Recommendations", "⚡", "#dc2626"),
    ("Prevention Tips",           "🛡️", "#1d4ed8"),
    # Healthy sections
    ("Plant Health Status",       "✅", "#16a34a"),
    ("Monitoring",                "👁️", "#0891b2"),
    ("Nutritional",               "🌱", "#15803d"),
]

# Section titles shown in progress pills — must match SECTION_PROMPTS keys
SECTION_TITLES = [
    "Diagnosis Summary",
    "Key Symptoms",
    "Treatment Recommendations",
    "Prevention Tips",
]

def _render_progress_pills(completed: list, active: str | None, placeholder):
    """Render live generation progress as styled pills."""
    pills_html = ""
    for title in SECTION_TITLES:
        if title in completed:
            style = ("background:#dcfce7; color:#14532d; border:1px solid #bbf7d0;")
            marker = "+"
        elif title == active:
            style = ("background:#fef9c3; color:#854d0e; border:1px solid #fde047;")
            marker = "..."
        else:
            style = ("background:#f1f5f9; color:#94a3b8; border:1px solid #e2e8f0;")
            marker = "o"
        pills_html += f"""
        <span style="{style} border-radius:20px; padding:5px 14px; margin:3px;
            font-family:'DM Sans',sans-serif; font-size:13px; font-weight:700;
            display:inline-block; letter-spacing:0.3px;">
            {marker}  {title}
        </span>"""

    placeholder.markdown(f"""
    <div style="margin-bottom:18px;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
            color:#64748b; font-weight:700; letter-spacing:2px;
            text-transform:uppercase; margin-bottom:10px;">
            Generating Advisory
        </div>
        <div style="line-height:2.4;">{pills_html}</div>
    </div>
    """, unsafe_allow_html=True)

def _strip_markdown_heading(text: str) -> str:
    text = re.sub(r"^#{1,4}\s*", "", text.strip())
    text = re.sub(r"^\*{1,2}(.*?)\*{1,2}$", r"\1", text.strip())
    return text.strip()

def _parse_rag_answer(answer: str) -> list[dict]:
    """
    Split only on ## headings — bold lines (**...**) are sub-labels
    inside Treatment Recommendations, NOT separate sections.
    """
    parts = re.split(r"\n(?=#{1,4}\s)", answer)
    if len(parts) <= 1:
        parts = re.split(r"(?=\n##\s)", answer)

    sections = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        if not lines:
            continue

        heading_raw = lines[0]
        heading     = _strip_markdown_heading(heading_raw)
        body_lines  = lines[1:]

        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        body = "\n".join(body_lines).strip()

        if not body:
            continue

        # Skip whole section if LLM marked it SKIP_SECTION
        if body.strip().upper().startswith("SKIP_SECTION"):
            continue

        icon, color = "•", "#475569"
        for title_kw, sec_icon, sec_color in RAG_SECTIONS:
            if any(kw in heading.lower() for kw in title_kw.lower().split()):
                icon  = sec_icon
                color = sec_color
                break

        sections.append({
            "title":   heading if heading else "Advisory",
            "icon":    icon,
            "color":   color,
            "content": body,
        })

    if not sections:
        sections = [{"title": "Advisory", "icon": "•", "color": "#475569",
                     "content": answer.strip()}]
    return sections

def _render_body_line(line: str) -> str:
    line = line.strip()
    if not line:
        return "<div style='height:6px;'></div>"

    # Silently hide SKIP_SECTION anywhere it appears
    if "SKIP_SECTION" in line.upper():
        return ""

    # Bullet points
    if line.startswith(("- ", "* ", "• ")):
        content = re.sub(r"^[-*•]\s+", "", line)
        if "SKIP_SECTION" in content.upper():
            return ""
        content = re.sub(r"\*\*(.*?)\*\*",
                         r"<strong style='color:#0f172a;'>\1</strong>", content)
        return f"""
        <div style="display:flex; gap:10px; align-items:flex-start; margin-bottom:8px;">
            <span style="color:#16a34a; font-size:14px; margin-top:3px;
                flex-shrink:0; font-weight:800;">▸</span>
            <span style="font-family:'DM Sans',sans-serif; font-size:17px;
                font-weight:600; color:#1e293b; line-height:1.7;">{content}</span>
        </div>"""

    # Numbered list
    m = re.match(r"^(\d+)\.\s+(.*)", line)
    if m:
        num, content = m.group(1), m.group(2)
        if "SKIP_SECTION" in content.upper():
            return ""
        content = re.sub(r"\*\*(.*?)\*\*",
                         r"<strong style='color:#0f172a;'>\1</strong>", content)
        return f"""
        <div style="display:flex; gap:12px; align-items:flex-start; margin-bottom:8px;">
            <span style="background:#e0f2fe; color:#0369a1; font-size:13px;
                font-weight:800; min-width:24px; height:24px; border-radius:50%;
                display:flex; align-items:center; justify-content:center;
                flex-shrink:0; margin-top:2px;">{num}</span>
            <span style="font-family:'DM Sans',sans-serif; font-size:17px;
                font-weight:600; color:#1e293b; line-height:1.7;">{content}</span>
        </div>"""

    # Bold sub-heading (e.g. **Immediate Actions (24-72 hrs)**)
    if re.match(r"^\*\*.*\*\*$", line):
        content = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        if "SKIP_SECTION" in content.upper():
            return ""
        return f"""
        <div style="font-family:'DM Sans',sans-serif; font-size:15px;
            font-weight:800; color:#334155; margin:14px 0 6px 0;
            padding-top:10px; border-top:1px solid #f1f5f9;">{content}</div>"""

    # Plain paragraph
    line = re.sub(r"\*\*(.*?)\*\*",
                  r"<strong style='color:#0f172a;'>\1</strong>", line)
    return f"""<p style="font-family:'DM Sans',sans-serif; font-size:17px;
        font-weight:600; color:#1e293b; line-height:1.8; margin:0 0 6px 0;">{line}</p>"""

def render_section_card(title: str, icon: str, color: str, content: str):
    body_html = "".join(_render_body_line(ln) for ln in content.splitlines())
    st.markdown(f"""
    <div style="background:#ffffff; border:1px solid {color}30;
                border-left:5px solid {color}; border-radius:14px;
                padding:22px 26px; margin-bottom:16px;
                box-shadow:0 2px 10px rgba(0,0,0,0.05);">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;
                    padding-bottom:12px; border-bottom:1px solid {color}20;">
            <span style="font-family:'Lora',serif; font-size:20px;
                font-weight:800; color:{color};">{title}</span>
        </div>
        <div>{body_html}</div>
    </div>
    """, unsafe_allow_html=True)

def render_rag_panel(rag_result: dict, disease_display: str):
    answer  = rag_result["answer"]
    sources = rag_result["sources"]
    sections = _parse_rag_answer(answer)

    # Advisory header — no emoji, renamed to AGRONOMY ADVISORY
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#f0fdf4,#ecfdf5);
                border:1px solid #bbf7d0; border-radius:14px;
                padding:18px 24px; margin-bottom:20px;">
        <div style="font-family:'Lora',serif; font-size:22px;
            font-weight:800; color:#14532d; margin-bottom:4px;
            letter-spacing:0.5px; text-transform:uppercase;">
            Agronomy Advisory
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:13px;
            color:#16a34a; font-weight:700;">
            {disease_display} &nbsp;|&nbsp; GuavaScan Knowledge Base &nbsp;|&nbsp; Gemini 2.5 Flash
        </div>
    </div>
    """, unsafe_allow_html=True)

    for sec in sections:
        render_section_card(sec["title"], sec["icon"], sec["color"], sec["content"])

    # Source attribution
    with st.expander("Knowledge Base Sources Retrieved", expanded=False):
        if sources:
            cols = st.columns(min(len(sources), 3))
            for i, s in enumerate(sources):
                with cols[i % len(cols)]:
                    st.markdown(f"""
                    <div style="background:#f8fafc; border:1px solid #e2e8f0;
                                border-radius:10px; padding:12px 14px; margin-bottom:8px;">
                        <div style="font-family:'JetBrains Mono',monospace; font-size:12px;
                            color:#16a34a; font-weight:700; margin-bottom:4px;">
                            {s['file']}
                        </div>
                        <div style="font-family:'DM Sans',sans-serif; font-size:13px;
                            color:#475569; font-weight:700;">
                            {s['section']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No source metadata available.")

    st.markdown("""
    <div style="background:#fefce8; border:1px solid #fde047; border-radius:10px;
                padding:12px 16px; margin-top:8px;">
        <p style="font-family:'DM Sans',sans-serif; font-size:14px; font-weight:700;
            color:#713f12; margin:0; line-height:1.6;">
            <strong>Disclaimer:</strong> This advisory is AI-generated from the GuavaScan
            knowledge base. Always verify with a certified agronomist before applying treatments.
        </p>
    </div>
    """, unsafe_allow_html=True)

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
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;700;800&family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #1e293b !important;
}
.stApp { background: #f0f7f0; }
.main .block-container {
    padding: 1.5rem 2.5rem 4rem 2.5rem;
    max-width: 1440px;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
p, li, span, div, label, h1, h2, h3 { color: #1e293b; }
.stMarkdown, .stMarkdown p, .stMarkdown li {
    color: #1e293b !important;
    font-size: 16px;
    font-weight: 600;
}
.streamlit-expanderHeader {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 800 !important;
    color: #1e293b !important;
    font-size: 15px !important;
}
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
.stCheckbox label { color: #1e293b !important; font-weight: 700 !important; }
.stCaption, [data-testid="stCaptionContainer"] {
    color: #64748b !important;
    font-weight: 600 !important;
}
.stAlert p { color: inherit !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="background:linear-gradient(135deg,#dcfce7 0%,#f0fdf4 60%,#ecfdf5 100%);
    border:1px solid #bbf7d0; border-radius:20px;
    padding:22px 36px; margin-bottom:30px;
    box-shadow:0 2px 16px rgba(22,163,74,0.10);">
    <div style="display:flex; align-items:center; gap:18px;">
        <div style="font-size:52px; line-height:1; flex-shrink:0;">🌿</div>
        <div>
            <h1 style="font-family:'Lora',serif; font-size:clamp(2rem,4vw,3rem);
                color:#14532d; margin:0 0 4px 0; font-weight:800;
                letter-spacing:-1px; line-height:1;">GuavaScan</h1>
            <p style="font-family:'DM Sans',sans-serif; font-size:clamp(0.95rem,2vw,1.1rem);
                color:#166534; font-weight:800; margin:0; letter-spacing:0.5px;">
                Guava Leaf Disease Detection &nbsp;&middot;&nbsp; ViT (DeiT-tiny) + RAG Advisory Pipeline
            </p>
        </div>
        <div style="margin-left:auto; text-align:right;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:12px;
                color:#16a34a; font-weight:800; background:#dcfce7;
                padding:4px 12px; border-radius:8px; border:1px solid #bbf7d0;
                white-space:nowrap;">RAG · Gemini 2.5 Flash</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD RESOURCES — silent, no status pills shown in UI
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

if not rag_ready:
    st.warning(f"RAG pipeline could not initialise: {rag_error}. Run `python rag/ingest.py` first.")

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_rag_result" not in st.session_state:
    st.session_state.last_rag_result = None

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

col_left, col_right = st.columns([1.1, 1], gap="large")

zone         = None
pil_image    = None
top_class    = None
top_conf     = None
display_name = None

with col_left:
    st.markdown("""
    <div style="font-family:'DM Sans',sans-serif; font-size:14px; letter-spacing:2px;
                color:#16a34a; font-weight:800; margin-bottom:14px; text-transform:uppercase;">
        Upload Leaf Image
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a guava leaf photo here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        pil_image = Image.open(uploaded).convert("RGB")
        _, img_center, _ = st.columns([0.175, 0.65, 0.175])
        with img_center:
            st.image(pil_image, use_container_width=True, caption="Uploaded leaf image")
    else:
        st.markdown("""
        <div style="background:#ffffff; border:2px dashed #bbf7d0;
                    border-radius:16px; height:290px; display:flex;
                    flex-direction:column; align-items:center;
                    justify-content:center; gap:14px;">
            <div style="font-size:54px; opacity:0.22;">🍃</div>
            <p style="color:#64748b; font-size:20px; font-weight:800; margin:0;">
                Upload a leaf photo to begin
            </p>
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
        Disease Analysis
    </div>
    """, unsafe_allow_html=True)

    if pil_image is None:
        st.markdown("""
        <div style="background:#ffffff; border:1px solid #e2e8f0;
                    border-radius:16px; padding:64px 24px; text-align:center;">
            <div style="font-size:46px; opacity:0.16; margin-bottom:16px;">🔬</div>
            <p style="color:#94a3b8; font-size:20px; font-weight:800; margin:0;">
                Results will appear here after upload
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Analysing leaf..."):
            tensor          = preprocess(pil_image)
            top3, all_probs = predict(model, tensor, device)

        top_class, top_conf     = top3[0]
        display_name            = DISPLAY_NAMES[top_class]
        severity_label, severity_color = SEVERITY[top_class]
        card_accent = "#16a34a" if top_class == cfg.get("healthy_class") else severity_color

        zone = get_confidence_zone(top_conf, cfg)
        st.markdown(zone_banner_html(zone, top_conf), unsafe_allow_html=True)

        if zone["rag_mode"] != "skip":
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
                            font-weight:800; margin-top:4px;">{severity_label}</div>
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
            st.markdown("""
            <div style="background:#fff1f2; border:2px dashed #fecdd3;
                        border-radius:14px; padding:44px 24px; text-align:center;">
                <p style="font-family:'Lora',serif; font-size:22px; font-weight:800;
                    color:#7f1d1d; margin:0 0 12px 0;">Diagnosis Withheld</p>
                <p style="font-family:'DM Sans',sans-serif; font-size:16px; font-weight:700;
                    color:#991b1b; margin:0; line-height:1.7;">
                    Upload a clearer, well-lit image of a single guava leaf.
                </p>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RAG ADVISORY PANEL
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<hr style='border-color:#d1fae5; margin:2rem 0;'>", unsafe_allow_html=True)

st.markdown("""
<div style="font-family:'Lora',serif; font-size:26px; font-weight:800;
            color:#854d0e; margin-bottom:22px; letter-spacing:0.3px;">
    RAG-Powered Agronomic Advisory
</div>
""", unsafe_allow_html=True)

if pil_image is None:
    st.markdown("""
    <div style="background:#ffffff; border:2px dashed #e2e8f0; border-radius:14px;
                padding:44px; text-align:center;">
        <p style="color:#94a3b8; font-size:20px; font-weight:800; margin:0;">
            Upload a leaf image to see the advisory
        </p>
    </div>
    """, unsafe_allow_html=True)

elif zone is not None and zone["rag_mode"] == "skip":
    st.markdown("""
    <div style="background:#fff1f2; border:2px dashed #fecdd3; border-radius:14px;
                padding:44px; text-align:center;">
        <p style="font-size:18px; font-weight:800; color:#991b1b; margin:0;">
            Confidence too low for advisory. Please upload a clearer image.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif not rag_ready:
    st.error("RAG pipeline not initialised. Run `python rag/ingest.py` then restart the app.")

elif zone is not None:
    if zone["rag_mode"] == "full":
        query_class    = top_class
        query_conf     = top_conf
        advisory_label = display_name
    else:
        query_class    = "__fallback__"
        query_conf     = top_conf
        advisory_label = "General Guava Care"
        st.warning("Low confidence — showing general guava care guidance.")

    current_pred = (query_class, round(query_conf, 3))

    if st.session_state.last_prediction == current_pred and st.session_state.last_rag_result:
        rag_result = st.session_state.last_rag_result
        st.caption("Advisory loaded from session cache.")
    else:
        # ── Live progress pills placeholder ───────────────────────────────────
        progress_placeholder = st.empty()
        _render_progress_pills([], None, progress_placeholder)

        try:
            if zone["rag_mode"] == "full":
                # Patch chain to report progress back via callback
                from rag.chain import (
                    _retrieve_comprehensive, _build_llm, _generate_section,
                    _format_context, load_config as chain_load_config
                )
                from rag.prompts import SECTION_PROMPTS
                import os as _os

                api_key = _os.getenv("GEMINI_API_KEY")
                chunks  = _retrieve_comprehensive(top_class, rag_cfg, retriever)
                llm     = _build_llm(rag_cfg, api_key)
                conf_str = f"{top_conf * 100:.1f}"
                display  = top_class.replace("Guava_", "").replace("_", " ").title()

                section_bodies = []
                completed      = []

                for section_title, prompt_template in SECTION_PROMPTS.items():
                    # Update pills — mark current as active
                    _render_progress_pills(completed, section_title, progress_placeholder)

                    body = _generate_section(
                        section_title=section_title,
                        section_prompt_template=prompt_template,
                        disease_name=display,
                        confidence=conf_str,
                        chunks=chunks,
                        llm=llm,
                    )
                    section_bodies.append(body)
                    completed.append(section_title)
                    # Update pills — mark completed
                    _render_progress_pills(completed, None, progress_placeholder)

                answer  = "\n\n".join(section_bodies)
                sources = [{"file": c["source"], "section": c["section"]} for c in chunks]
                seen, unique_sources = set(), []
                for s in sources:
                    key = (s["file"], s["section"])
                    if key not in seen:
                        seen.add(key)
                        unique_sources.append(s)
                rag_result = {"answer": answer, "sources": unique_sources}

            else:
                # Fallback — healthy prompt, single call
                _render_progress_pills([], "General Care", progress_placeholder)
                from rag.chain import _format_context
                from rag.prompts import HEALTHY_PROMPT
                from langchain_google_genai import ChatGoogleGenerativeAI
                from langchain_core.messages import HumanMessage

                fallback_query = rag_cfg.get("fallback_query", "guava orchard management prevention care")
                chunks   = retriever.retrieve(fallback_query)
                context  = _format_context(chunks)
                api_key  = os.getenv("GEMINI_API_KEY")
                llm = ChatGoogleGenerativeAI(
                    model=rag_cfg["llm_model"],
                    google_api_key=api_key,
                    temperature=0.2,
                    model_kwargs={"generation_config": {
                        "max_output_tokens": 2048,
                        "thinking_config": {"thinking_budget": 0},
                    }},
                )
                prompt_val = HEALTHY_PROMPT.format(
                    confidence=f"{query_conf*100:.1f}", context=context
                )
                response = llm.invoke([HumanMessage(content=prompt_val)])
                sources  = [{"file": c["source"], "section": c["section"]} for c in chunks]
                rag_result = {"answer": response.content, "sources": sources}

            # Clear progress pills
            progress_placeholder.empty()

            st.session_state.last_prediction = current_pred
            st.session_state.last_rag_result  = rag_result

        except Exception as e:
            progress_placeholder.empty()
            st.error(f"RAG chain error: {e}")
            st.info("Ensure GEMINI_API_KEY is set in .env and the knowledge base is indexed.")
            rag_result = None

    if rag_result:
        render_rag_panel(rag_result, advisory_label)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align:center; margin-top:56px; padding-top:24px;
            border-top:1px solid #d1fae5;">
    <p style="font-family:'DM Sans',sans-serif; font-size:15px;
        font-weight:700; color:#64748b; margin:0;">
        GuavaScan &nbsp;&middot;&nbsp; ViT (DeiT-tiny) + RAG Advisory &nbsp;&middot;&nbsp;
        Gemini 2.5 Flash &nbsp;&middot;&nbsp; ChromaDB &middot; BM25 &middot; sentence-transformers
    </p>
</div>
""", unsafe_allow_html=True)