"""OpenAI calls: query rewriting + filter extraction, answer synthesis, judging."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from openai import OpenAI

from app.config import get_settings

# $ per 1M tokens (gpt-4o-mini, Jan 2025 list price)
PRICES = {
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-4o": (2.50, 10.00),
}

FILTERABLE = {
    "growth_habit": "Tree | Shrub | Subshrub | Forb/herb | Graminoid | Vine | Lichenous",
    "duration": "Annual | Biennial | Perennial",
    "family": "botanical family name",
    "genus": "genus name",
    "native_regions": "L48 | AK | HI | CAN | PR | VI (native to)",
    "drought_tolerance": "None | Low | Medium | High",
    "shade_tolerance": "Intolerant | Intermediate | Tolerant",
    "salinity_tolerance": "None | Low | Medium | High",
    "fire_tolerance": "None | Low | Medium | High",
    "moisture_use": "Low | Medium | High",
    "growth_rate": "Slow | Moderate | Rapid",
    "toxicity": "None | Slight | Moderate | Severe",
    "nitrogen_fixation": "None | Low | Medium | High",
    "palatable_human": "Yes | No",
    "bloom_period": "e.g. Spring, Early Summer, Late Spring",
    "height_mature_ft": 'numeric range object, e.g. {"lte": 6}',
    "temp_min_f": "numeric range object (cold hardiness, lower is hardier)",
    "ph_min": "numeric range object",
    "ph_max": "numeric range object",
    "precip_min_in": "numeric range object",
    "precip_max_in": "numeric range object",
}

REWRITE_SYSTEM = f"""You prepare user questions for retrieval over USDA PLANTS species documents.

Return JSON with:
  "rewritten": a keyword-rich search query. Expand horticultural jargon, add the
      botanical vocabulary USDA uses (e.g. "part shade" -> "shade tolerance intermediate
      tolerant"), resolve common names to genus when you are confident, and keep any Latin
      name or USDA symbol verbatim.
  "filters": hard metadata filters, ONLY when the user states an unambiguous constraint.
      Available fields: {json.dumps(FILTERABLE, indent=2)}
      Use a list for "any of". Use {{"lte": x}} / {{"gte": x}} for numeric bounds.
      Prefer no filter over a guessed filter; an over-filtered search returns nothing.
  "archetype": one of "name_lookup", "trait_filter", "site_recommendation",
      "care_howto", "comparison", "safety", "other".

Output JSON only."""

PROMPTS = {
    "v1": """You are a plant-selection assistant grounded in USDA PLANTS data.
Answer the question using ONLY the numbered context. Be concise (max 150 words).
Cite species you use as [n] matching the context numbering.
If the context does not support an answer, say so plainly.""",
    "v2": """You are a plant-selection assistant grounded in USDA PLANTS data.

Rules:
- Use ONLY the numbered context; never rely on outside botanical knowledge.
- Open with a one-sentence direct answer.
- Then list each recommended or discussed species as a bullet: common name (Scientific name)
  followed by the specific traits from the context that justify it, with a [n] citation.
- Quote concrete values (pH range, mature height, tolerance ratings, bloom period) rather
  than vague adjectives.
- If a constraint in the question is not covered by the context, state that explicitly
  instead of guessing.
- If the context contains nothing relevant, reply exactly: "The USDA PLANTS knowledge base
  in this app does not contain enough information to answer that." """,
    "v3": """You are a plant-selection assistant grounded in USDA PLANTS data.
Use ONLY the numbered context.

Return, in order:
1. **Answer** — one or two sentences.
2. **Candidates** — a markdown table with columns:
   Species | Habit | Mature ht (ft) | Shade | Drought | pH | Source.
   Put the [n] citation in the Source column. Use "—" for values missing from the context.
3. **Caveats** — anything the context cannot confirm, or constraints not verifiable.

If the context contains nothing relevant, reply exactly: "The USDA PLANTS knowledge base in
this app does not contain enough information to answer that." """,
}

JUDGE_SYSTEM = """You grade a plant assistant's answer against the retrieved context.

Score each 1-5:
  relevance: does it answer the question asked?
  groundedness: is every factual claim supported by the context (5 = fully, 1 = fabricated)?
  citation_validity: are [n] citations present and pointing at the right context items?
Also set "hallucinated": true if any claim contradicts or is absent from the context.

Return JSON: {"relevance": int, "groundedness": int, "citation_validity": int,
"hallucinated": bool, "reason": "one sentence"}"""


@lru_cache
def client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_price, completion_price = PRICES.get(model, PRICES["gpt-4o-mini"])
    return (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1_000_000


def _chat(
    system: str,
    user: str,
    model: str | None = None,
    json_mode: bool = False,
    temperature: float = 0.0,
) -> tuple[str, dict[str, Any]]:
    settings = get_settings()
    model = model or settings.openai_model
    response = client().chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **({"response_format": {"type": "json_object"}} if json_mode else {}),
    )
    usage = response.usage
    meta = {
        "model": model,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }
    meta["cost_usd"] = cost_usd(model, meta["prompt_tokens"], meta["completion_tokens"])
    return response.choices[0].message.content or "", meta


def rewrite_query(question: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, meta = _chat(REWRITE_SYSTEM, question, json_mode=True)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    return (
        {
            "rewritten": parsed.get("rewritten") or question,
            "filters": parsed.get("filters") or {},
            "archetype": parsed.get("archetype") or "other",
        },
        meta,
    )


def format_context(hits: list[dict[str, Any]]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        payload = hit.get("payload") or {}
        header = f"[{index}] {hit.get('title')} | section: {hit.get('section')}"
        if payload.get("source_url"):
            header += " | USDA PDF"
        blocks.append(f"{header}\n{hit.get('text')}")
    return "\n\n".join(blocks)


def answer(
    question: str, hits: list[dict[str, Any]], variant: str | None = None
) -> tuple[str, dict[str, Any]]:
    variant = variant or get_settings().prompt_variant
    system = PROMPTS.get(variant, PROMPTS["v2"])
    user = f"Question: {question}\n\nContext:\n{format_context(hits)}"
    text, meta = _chat(system, user, temperature=0.1)
    meta["prompt_variant"] = variant
    return text, meta


def judge(question: str, hits: list[dict[str, Any]], generated: str) -> dict[str, Any]:
    settings = get_settings()
    user = f"Question: {question}\n\nContext:\n{format_context(hits)}\n\nAnswer:\n{generated}"
    raw, meta = _chat(JUDGE_SYSTEM, user, model=settings.openai_judge_model, json_mode=True)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    parsed["_meta"] = meta
    return parsed
