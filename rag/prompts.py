"""
rag/prompts.py
--------------
Prompt templates for GuavaScan RAG pipeline.

Exports:
  SECTION_PROMPTS  — OrderedDict of 4 sections (primary path used by chain.py)
  HEALTHY_PROMPT   — 3-section single-call for healthy leaf
  DISEASE_PROMPT   — legacy reference (not used by chain.py)

Reduced from 6 to 4 sections:
  1. Diagnosis Summary
  2. Key Symptoms
  3. Treatment Recommendations  (merged: immediate + chemical + biological)
  4. Prevention Tips

chain.py iterates SECTION_PROMPTS unchanged — no edits to chain.py needed.
"""

from collections import OrderedDict

# ══════════════════════════════════════════════════════════════════════════════
# SECTION-BY-SECTION PROMPTS — used by chain.py (primary path)
# ══════════════════════════════════════════════════════════════════════════════

SECTION_PROMPTS = OrderedDict([

    ("Diagnosis Summary", """\
You are an expert plant pathologist specialising in guava (Psidium guajava).

A ViT model detected: {disease_name}  (confidence: {confidence}%)

Using ONLY the CONTEXT below, write the "Diagnosis Summary" section.

CONTEXT:
{context}

---

Respond with exactly this heading followed by 2-3 concise sentences.
Prose only, no bullet points.
Cover: what the condition is, its causal agent, and why it matters for the crop.
Do NOT include any other sections.

## Diagnosis Summary
"""),

    ("Key Symptoms", """\
You are an expert plant pathologist specialising in guava (Psidium guajava).

Detected condition: {disease_name}  (confidence: {confidence}%)

Using ONLY the CONTEXT below, write the "Key Symptoms" section.
Help the farmer visually confirm the diagnosis on the actual plant.

CONTEXT:
{context}

---

If the context contains no symptom information, write exactly: SKIP_SECTION

Otherwise respond with exactly this heading and 4-6 bullet points.
Each bullet: one specific visual symptom with colour, texture, or location on plant.
Do NOT include any other sections.

## Key Symptoms
"""),

    ("Treatment Recommendations", """\
You are an expert agronomist specialising in guava (Psidium guajava).

Detected condition: {disease_name}  (confidence: {confidence}%)

Using ONLY the CONTEXT below, write the "Treatment Recommendations" section.
This section combines immediate actions, chemical control, and biological alternatives.

CONTEXT:
{context}

---

If this condition is purely abiotic/nutritional AND the context has no treatment data,
write exactly: SKIP_SECTION

Otherwise respond with exactly this heading, then use these bold sub-labels:

## Treatment Recommendations

**Immediate Actions (24-72 hrs)**
- 2-3 urgent steps to limit spread or damage right now.

**Chemical Control**
- Fungicides or pesticides with concentration and frequency from context.
- If not applicable or no data in context: write SKIP_SECTION under this sub-label only.

**Biological and Organic Alternatives**
- Biocontrol agents or organic options from context.
- If no data in context: write SKIP_SECTION under this sub-label only.

Do NOT include any other sections.
"""),

    ("Prevention Tips", """\
You are an expert agronomist specialising in guava (Psidium guajava).

Detected condition: {disease_name}  (confidence: {confidence}%)

Using ONLY the CONTEXT below, write the "Prevention Tips" section.
Focus on cultural and agronomic practices to prevent recurrence.

CONTEXT:
{context}

---

If the context contains no prevention data, write exactly: SKIP_SECTION

Otherwise respond with exactly this heading and 4-5 bullet points.
Do NOT include any other sections.

## Prevention Tips
"""),

])


# ══════════════════════════════════════════════════════════════════════════════
# HEALTHY PROMPT — 3-section single call
# ══════════════════════════════════════════════════════════════════════════════

HEALTHY_PROMPT = """\
You are an expert agronomist specialising in guava (Psidium guajava) orchard management.

A ViT model classified a guava leaf as HEALTHY.
- Model Confidence: {confidence}%

The plant shows no signs of disease. Provide a preventive care advisory.
Use ONLY the information in the CONTEXT below.
Do not include any curative treatment.
If a section has no relevant data in context, write exactly: SKIP_SECTION

CONTEXT:
{context}

---

Respond with EXACTLY these 3 sections using ## headings.
Use - bullet points. Keep responses concise.

## Plant Health Status
2-3 sentences confirming healthy appearance and what proactive monitoring means for yield.

## Monitoring and Prevention
- Weekly inspection checklist: early signs to watch for.
- Preventive spray schedule or products recommended in context.
- Orchard hygiene practices to prevent future outbreaks.

## Nutritional and Crop Care
- Key fertilisation and nutrition practices from context.
- Irrigation and soil management tips.

---
Respond only with the 3 sections above. No preamble or conclusion outside sections.
"""


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY DISEASE PROMPT — reference only, not used by chain.py
# ══════════════════════════════════════════════════════════════════════════════

DISEASE_PROMPT = """\
You are an expert plant pathologist and agronomist specialising in guava (Psidium guajava).

Detected condition: {disease_name}  (confidence: {confidence}%)

Use ONLY the CONTEXT below.
If a section is not applicable, write exactly: SKIP_SECTION

CONTEXT:
{context}

---

## Diagnosis Summary
2-3 sentences on the condition, causal agent, and crop significance.

## Key Symptoms
- 4-6 visual symptoms to confirm the diagnosis.

## Treatment Recommendations
**Immediate Actions (24-72 hrs)**
- Urgent steps to limit spread.
**Chemical Control**
- Fungicides or pesticides from context with dosage.
**Biological and Organic Alternatives**
- Biocontrol or organic options from context.

## Prevention Tips
- 4-5 cultural practices to prevent recurrence.
"""