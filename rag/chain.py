"""
rag/chain.py
------------
LangChain LCEL RAG chain for GuavaScan.

Public API:
    run_rag_chain(predicted_class, confidence, cfg, retriever) -> dict
        Returns: {"answer": str, "sources": list[dict]}

Fixes applied (v3):
    1. thinking_budget=0  — Gemini 2.5 Flash's thinking tokens ate into
       max_output_tokens, truncating output mid-sentence even at 2048.
       Disabling thinking gives the full budget to response text.

    2. max_output_tokens passed via model_kwargs (not top-level kwarg) —
       LangChain's ChatGoogleGenerativeAI silently drops top-level
       generation params; model_kwargs is the correct passthrough path.

    3. finish_reason logging — surfaces MAX_TOKENS / SAFETY silently so
       truncation root cause is always visible in console.

    4. Section-by-section generation — instead of one 6-section call that
       risks truncation, each advisory section is generated independently.
       Each call retrieves its own targeted chunks (from the multi-query
       pool) and writes one section only.  Sections are then assembled.

    5. Information completeness check — before generating a section, we
       score whether the retrieved chunks contain enough signal for it.
       If avg relevance score < threshold, the section is marked
       SKIP_SECTION without an LLM call (saves latency + tokens).
"""

import os
import sys
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from rag.prompts import DISEASE_PROMPT, HEALTHY_PROMPT, SECTION_PROMPTS
from rag.retriever import GuavaRetriever


# ── Config loader ──────────────────────────────────────────────────────────────
def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(ROOT, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Context formatter ─────────────────────────────────────────────────────────
def _format_context(retrieved_chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        header = f"[Source {i}: {chunk['source']} — {chunk['section']}]"
        parts.append(f"{header}\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


# ── Query builder ─────────────────────────────────────────────────────────────
def _build_query(predicted_class: str, cfg: dict) -> str:
    query_map = cfg.get("class_query_map", {})
    return query_map.get(
        predicted_class,
        f"guava {predicted_class} disease symptoms treatment management control",
    )


# ── Comprehensive multi-query retriever ───────────────────────────────────────
def _retrieve_comprehensive(
    predicted_class: str,
    cfg: dict,
    retriever: "GuavaRetriever",
) -> list[dict]:
    """
    Run 6 targeted queries per disease and merge results (max 15 chunks).
    Returns all chunks with their retrieval scores preserved.
    """
    base_query   = _build_query(predicted_class, cfg)
    disease_name = predicted_class.replace("Guava_", "").replace("_", " ")

    queries = [
        base_query,
        f"guava {disease_name} chemical treatment fungicide pesticide dosage",
        f"guava {disease_name} immediate action urgent steps control spread",
        f"guava {disease_name} biological organic biocontrol Trichoderma",
        f"guava {disease_name} symptoms visual signs confirm diagnosis",
        f"guava {disease_name} prevention cultural practices management",
    ]

    seen_content = set()
    all_chunks   = []

    for q in queries:
        chunks = retriever.retrieve(q)
        for c in chunks:
            key = c["content"][:100]
            if key not in seen_content:
                seen_content.add(key)
                all_chunks.append(c)

    return all_chunks[:15]


# ── Information completeness check ────────────────────────────────────────────
# Section-specific keyword signals — if none of these words appear in the
# top chunks for this section, the context is almost certainly too thin to
# generate anything useful and we skip the LLM call entirely.
_SECTION_KEYWORDS = {
    "Diagnosis Summary":       ["cause", "causal", "fungus", "bacteria", "virus", "pathogen",
                                 "disease", "condition", "infection", "mite", "insect"],
    "Symptoms to Confirm":     ["symptom", "sign", "lesion", "spot", "discolor", "wilt",
                                 "yellow", "brown", "necrosis", "appear", "look"],
    "Immediate Actions":       ["action", "urgent", "remove", "prune", "isolate", "spray",
                                 "apply", "treat", "control", "sanitize", "step"],
    "Chemical Treatment":      ["fungicide", "pesticide", "chemical", "copper", "mancozeb",
                                 "chlorothalonil", "spray", "dose", "concentration", "ppm",
                                 "ml", "g/l", "application"],
    "Biological Alternatives": ["biocontrol", "biological", "trichoderma", "neem", "organic",
                                 "biopesticide", "beneficial", "bacillus", "pseudomonas"],
    "Preventive Measures":     ["prevent", "cultural", "sanitation", "prune", "spacing",
                                 "drainage", "avoid", "rotation", "resistant"],
    # healthy-leaf sections
    "Plant Health Status":     ["healthy", "health", "vigour", "vigor", "normal", "good"],
    "Routine Monitoring Checklist": ["monitor", "inspect", "check", "watch", "observe", "weekly"],
    "Preventive Spray Schedule":    ["spray", "schedule", "calendar", "preventive", "interval"],
    "Nutritional Maintenance":      ["nutrient", "fertiliz", "nitrogen", "potassium", "phosphorus",
                                     "micronutrient", "soil"],
    "Orchard Hygiene Practices":    ["hygiene", "sanitation", "fallen", "debris", "clean", "orchard"],
}

_COMPLETENESS_THRESHOLD = 2  # min keyword hits required to attempt generation


def _has_sufficient_context(section_title: str, chunks: list[dict]) -> bool:
    """
    Return True if the chunks contain enough signal for this section.
    Counts unique keyword hits across all chunk content.
    """
    keywords = _SECTION_KEYWORDS.get(section_title, [])
    if not keywords:
        return True  # unknown section — don't block it

    combined = " ".join(c["content"].lower() for c in chunks)
    hits = sum(1 for kw in keywords if kw.lower() in combined)
    sufficient = hits >= _COMPLETENESS_THRESHOLD

    if not sufficient:
        print(f"[chain] SKIP {section_title!r} — only {hits}/{_COMPLETENESS_THRESHOLD} "
              f"keyword hits in retrieved context")
    return sufficient


# ── LLM factory ───────────────────────────────────────────────────────────────
def _build_llm(cfg: dict, api_key: str) -> ChatGoogleGenerativeAI:
    """
    Build ChatGoogleGenerativeAI with:
      - thinking_budget=0 so max_output_tokens is fully available for response
      - max_output_tokens passed via model_kwargs (LangChain's correct path)
      - temperature 0.2 for factual advisory text
    """
    return ChatGoogleGenerativeAI(
        model=cfg["llm_model"],
        google_api_key=api_key,
        temperature=0.2,
        # ── FIX: pass generation config via model_kwargs, not top-level ──────
        # Top-level max_output_tokens is silently dropped by LangChain wrapper.
        model_kwargs={
            "generation_config": {
                "max_output_tokens": 8192,        # large ceiling — sections are small
                "thinking_config": {
                    "thinking_budget": 0,          # FIX: disable thinking tokens
                },
            }
        },
    )


# ── finish_reason logger ──────────────────────────────────────────────────────
def _log_finish_reason(response, section_label: str = ""):
    """
    Log finish_reason and token counts for every LLM call.
    Surfaces MAX_TOKENS truncation that would otherwise be silent.
    """
    try:
        candidates = getattr(response, "response_metadata", {})
        reason     = candidates.get("finish_reason", "unknown")
        usage      = candidates.get("usage_metadata", {})
        out_tokens = usage.get("candidates_token_count", "?")
        print(f"[chain] {section_label or 'call'} → finish_reason={reason}, "
              f"output_tokens={out_tokens}")
        if reason == "MAX_TOKENS":
            print(f"[chain] ⚠️  MAX_TOKENS hit on {section_label!r} — "
                  f"consider raising max_output_tokens further or splitting the section.")
    except Exception:
        pass  # logging failure should never break the pipeline


# ── Section-by-section generation ────────────────────────────────────────────
def _generate_section(
    section_title: str,
    section_prompt_template: str,
    disease_name: str,
    confidence: str,
    chunks: list[dict],
    llm: ChatGoogleGenerativeAI,
) -> str:
    """
    Generate ONE section.

    Steps:
      1. Run information completeness check — if context is too thin, return SKIP_SECTION.
      2. Filter chunks to those most relevant to this section (scored by keyword hits).
      3. Call LLM with a focused single-section prompt.
      4. Log finish_reason.
      5. Return the section body (## heading + content).
    """
    # Step 1: completeness check
    if not _has_sufficient_context(section_title, chunks):
        return f"## {section_title}\nSKIP_SECTION"

    # Step 2: select the most relevant chunks for this section (top 5)
    keywords = _SECTION_KEYWORDS.get(section_title, [])
    if keywords:
        def _score(chunk):
            text = chunk["content"].lower()
            return sum(1 for kw in keywords if kw.lower() in text)
        section_chunks = sorted(chunks, key=_score, reverse=True)[:5]
    else:
        section_chunks = chunks[:5]

    context = _format_context(section_chunks)

    # Step 3: fill the section-specific prompt
    prompt = section_prompt_template.format(
        disease_name=disease_name,
        confidence=confidence,
        context=context,
    )

    # Step 4: call LLM
    parser   = StrOutputParser()
    response = llm.invoke([HumanMessage(content=prompt)])
    _log_finish_reason(response, section_title)
    body = parser.invoke(response).strip()

    # Ensure the section starts with the correct ## heading
    if not body.startswith(f"## {section_title}"):
        body = f"## {section_title}\n{body}"

    return body


# ── Healthy leaf — single-call (shorter, 5 sections, lower risk) ──────────────
def _run_healthy_chain(
    confidence: str,
    cfg: dict,
    chunks: list[dict],
    llm: ChatGoogleGenerativeAI,
) -> str:
    context     = _format_context(chunks)
    prompt_val  = HEALTHY_PROMPT.format(confidence=confidence, context=context)
    parser      = StrOutputParser()
    response    = llm.invoke([HumanMessage(content=prompt_val)])
    _log_finish_reason(response, "HEALTHY_full")
    return parser.invoke(response).strip()


# ── Main run function ─────────────────────────────────────────────────────────
def run_rag_chain(
    predicted_class: str,
    confidence: float,
    cfg: dict,
    retriever: GuavaRetriever,
) -> dict:
    """
    Run the full RAG chain for a given prediction.

    For disease classes: uses section-by-section generation.
    For healthy leaf:    uses single call (short prompt, 5 sections).

    Args:
        predicted_class: Model output class name (e.g. "Guava_Rust")
        confidence:      Model confidence 0.0–1.0
        cfg:             Loaded config dict
        retriever:       Initialised GuavaRetriever instance

    Returns:
        {"answer": str, "sources": list[dict]}
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
        )

    # ── Comprehensive multi-query retrieval ───────────────────────────────────
    chunks       = _retrieve_comprehensive(predicted_class, cfg, retriever)
    conf_str     = f"{confidence * 100:.1f}"
    healthy_class = cfg.get("healthy_class", "Guava_healthy")
    is_healthy    = predicted_class == healthy_class

    # ── Build LLM (shared across all section calls) ───────────────────────────
    llm = _build_llm(cfg, api_key)

    # ── Generate advisory ─────────────────────────────────────────────────────
    if is_healthy:
        answer = _run_healthy_chain(conf_str, cfg, chunks, llm)
    else:
        display_name  = predicted_class.replace("Guava_", "").replace("_", " ").title()
        section_bodies = []

        # SECTION_PROMPTS is an OrderedDict of {section_title: prompt_template}
        # defined in prompts.py — one focused prompt per section
        for section_title, prompt_template in SECTION_PROMPTS.items():
            print(f"[chain] Generating section: {section_title}")
            body = _generate_section(
                section_title=section_title,
                section_prompt_template=prompt_template,
                disease_name=display_name,
                confidence=conf_str,
                chunks=chunks,
                llm=llm,
            )
            section_bodies.append(body)

        answer = "\n\n".join(section_bodies)

    # ── Build sources ─────────────────────────────────────────────────────────
    sources = [{"file": c["source"], "section": c["section"]} for c in chunks]
    seen, unique_sources = set(), []
    for s in sources:
        key = (s["file"], s["section"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return {"answer": answer, "sources": unique_sources}