"""
rag/prompts.py
--------------
Prompt templates for GuavaScan RAG pipeline.

Three exports:
  DISEASE_PROMPT   — legacy single-call template (kept for reference / fallback)
  HEALTHY_PROMPT   — 5-section single call for healthy leaf
  SECTION_PROMPTS  — OrderedDict of {section_title: prompt_template}
                     used by chain.py for section-by-section generation

Why section-by-section?
  Gemini 2.5 Flash with thinking_budget=0 and max_output_tokens=8192 can
  still produce truncated output when asked to write 6 long sections in one
  shot because the *prompt* itself (15 retrieved chunks + instructions) is
  very large.  Splitting into 6 focused calls means:
    • Each prompt is ~5 chunks + 1 section instruction → much smaller input
    • Each output is 1 section → far below any token ceiling
    • A thin-context section is caught by the completeness check BEFORE the
      LLM call, so we never waste a round-trip on a near-empty section

Format rules for all section prompts:
  - {disease_name}, {confidence}, {context} are the only variables
  - Use ## heading at the top so _parse_rag_answer() splits correctly
  - Ask for bullet points, not paragraphs — keeps output compact
  - Each prompt is self-contained (no references to other sections)
"""

from collections import OrderedDict

# ══════════════════════════════════════════════════════════════════════════════
# SECTION-BY-SECTION PROMPTS (primary path)
# ══════════════════════════════════════════════════════════════════════════════

SECTION_PROMPTS = OrderedDict([

    ("Diagnosis Summary", """\
You are an expert plant pathologist specialising in guava (Psidium guajava).

A ViT model detected: {disease_name}  (confidence: {confidence}%)

Using ONLY the CONTEXT below, write the "Diagnosis Summary" section of a disease advisory.

CONTEXT:
{context}

---

Respond with exactly this heading and 2–3 concise sentences beneath it.
Do NOT add bullet points — prose only for this section.
Do NOT include any other sections.

## Diagnosis Summary
"""),

    ("Symptoms to Confirm", """\
You are an expert plant pathologist specialising in guava (Psidium guajava).

Detected condition: {disease_name}  (confidence: {confidence}%)

Using ONLY the CONTEXT below, write the "Symptoms to Confirm" section.
This section helps a farmer verify the diagnosis on the actual plant.

CONTEXT:
{context}

---

If the context does not contain symptom information for this condition, write exactly:
SKIP_SECTION

Otherwise respond with exactly this heading and 4–6 bullet points.
Each bullet: one specific visual symptom the farmer should look for.
Do NOT include any other sections.

## Symptoms to Confirm
"""),

    ("Immediate Actions", """\
You are an expert agronomist specialising in guava (Psidium guajava).

Detected condition: {disease_name}  (confidence: {confidence}%)

Using ONLY the CONTEXT below, write the "Immediate Actions" section.
Focus on what must be done within 24–72 hours to limit spread.

CONTEXT:
{context}

---

If the context contains no actionable guidance for this condition, write exactly:
SKIP_SECTION

Otherwise respond with exactly this heading and 3–4 bullet points.
Each bullet: one concrete, urgent action step.
Do NOT include any other sections.

## Immediate Actions
"""),

    ("Chemical Treatment", """\
You are an expert agronomist specialising in guava (Psidium guajava).

Detected condition: {disease_name}  (confidence: {confidence}%)

Using ONLY the CONTEXT below, write the "Chemical Treatment" section.
Include fungicide/pesticide names, concentrations, application frequency, and
any resistance-management rotation notes if present in the context.

CONTEXT:
{context}

---

If this condition is purely abiotic/nutritional OR the context contains no
chemical treatment data, write exactly:
SKIP_SECTION

Otherwise respond with exactly this heading and bullet points.
Do NOT include any other sections.

## Chemical Treatment
"""),

    ("Biological Alternatives", """\
You are an expert agronomist specialising in guava (Psidium guajava).

Detected condition: {disease_name}  (confidence: {confidence}%)

Using ONLY the CONTEXT below, write the "Biological Alternatives" section.
Cover biocontrol agents, biopesticides, and organic/cultural options only.

CONTEXT:
{context}

---

If the context contains no biological or organic control data for this condition,
write exactly:
SKIP_SECTION

Otherwise respond with exactly this heading and bullet points.
Do NOT include any other sections.

## Biological Alternatives
"""),

    ("Preventive Measures", """\
You are an expert agronomist specialising in guava (Psidium guajava).

Detected condition: {disease_name}  (confidence: {confidence}%)

Using ONLY the CONTEXT below, write the "Preventive Measures" section.
Focus on cultural and agronomic practices to prevent recurrence.

CONTEXT:
{context}

---

If the context contains no prevention data for this condition, write exactly:
SKIP_SECTION

Otherwise respond with exactly this heading and 4–5 bullet points.
Do NOT include any other sections.

## Preventive Measures
"""),

])


# ══════════════════════════════════════════════════════════════════════════════
# HEALTHY LEAF PROMPT (single-call — 5 shorter sections, low truncation risk)
# ══════════════════════════════════════════════════════════════════════════════

HEALTHY_PROMPT = """\
You are an expert agronomist specialising in guava (Psidium guajava) orchard management.

A ViT model classified a guava leaf as HEALTHY.
- Model Confidence: {confidence}%

This leaf shows no signs of disease. Provide a preventive care and monitoring advisory.
Use ONLY the information provided in the CONTEXT below.
Do not include any curative treatment section — the plant is healthy.
If a section has no data in context, write exactly: SKIP_SECTION

CONTEXT:
{context}

---

Respond with EXACTLY the following 5 sections using ## headings.
Use - bullet points inside each section. Do not write long paragraphs.

## Plant Health Status
2–3 sentences confirming the healthy appearance and what this means for the crop.

## Routine Monitoring Checklist
- Weekly inspection checklist: what to look for to catch early disease signs.

## Preventive Spray Schedule
- Recommended preventive spray calendar and products from the context.

## Nutritional Maintenance
- Key nutrition and fertilisation practices to maintain plant immunity and vigour.

## Orchard Hygiene Practices
- Cultural practices to prevent future disease outbreaks.

---
Respond only with the 5 sections above. No preamble or conclusion outside sections.
"""


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY SINGLE-CALL DISEASE PROMPT (kept as fallback reference — not used
# by chain.py in normal operation; chain.py uses SECTION_PROMPTS instead)
# ══════════════════════════════════════════════════════════════════════════════

DISEASE_PROMPT = """\
You are an expert plant pathologist and agronomist specialising in guava (Psidium guajava) diseases.

A Vision Transformer (ViT) deep learning model has detected the following condition in a guava leaf image:
- Detected Condition: {disease_name}
- Model Confidence: {confidence}%

Use ONLY the information provided in the CONTEXT below to generate your advisory report.
Do not recommend any treatment not explicitly present in the provided context.
If information is insufficient for any section, write: "Insufficient data — consult a certified agronomist."
If a section is NOT APPLICABLE for this specific condition (for example, Chemical Treatment
for a purely abiotic or nutritional disorder, or Biological Alternatives for a pest-only issue where no biocontrol data exists in context), write exactly: SKIP_SECTION

CONTEXT:
{context}

---

Respond with EXACTLY the following 6 sections using ## headings. Do not rename, skip, or reorder sections.
Use - bullet points inside each section. Do not write long paragraphs.

## Diagnosis Summary
2–3 sentences: what this condition is, its causal agent, and why it matters for the crop.

## Symptoms to Confirm
- Bullet list of 4–6 visual symptoms the farmer should verify on the plant.

## Immediate Actions
- Bullet list of 3–4 urgent steps to take within 24–72 hours to limit spread.

## Chemical Treatment
- Bullet list of recommended fungicides or pesticides with concentration and frequency from the context.
- Include resistance-management rotation notes if present in context.

## Biological Alternatives
- Bullet list of biocontrol agents, biopesticides, or organic options from the context.

## Preventive Measures
- Bullet list of 4–5 cultural and agronomic practices to prevent recurrence.

---
Respond only with the 6 sections above. Do not add a preamble, introduction, or conclusion outside the sections.
"""