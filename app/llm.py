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

# USDA reports cold hardiness as a minimum temperature, not a zone, so zone talk in a
# question has to be translated before it can be filtered on.
ZONE_MIN_TEMP_F = {
    1: -60.0,
    2: -50.0,
    3: -40.0,
    4: -30.0,
    5: -20.0,
    6: -10.0,
    7: 0.0,
    8: 10.0,
    9: 20.0,
    10: 30.0,
    11: 40.0,
    12: 50.0,
    13: 60.0,
}

CATEGORICAL_VOCAB = {
    "growth_habit": ("Tree", "Shrub", "Subshrub", "Forb/herb", "Graminoid", "Vine", "Lichenous"),
    "duration": ("Annual", "Biennial", "Perennial"),
    "native_regions": ("L48", "AK", "HI", "CAN", "PR", "VI"),
    "drought_tolerance": ("None", "Low", "Medium", "High"),
    "shade_tolerance": ("Intolerant", "Intermediate", "Tolerant"),
    "salinity_tolerance": ("None", "Low", "Medium", "High"),
    "fire_tolerance": ("None", "Low", "Medium", "High"),
    "moisture_use": ("Low", "Medium", "High"),
    "growth_rate": ("Slow", "Moderate", "Rapid"),
    "toxicity": ("None", "Slight", "Moderate", "Severe"),
    "nitrogen_fixation": ("None", "Low", "Medium", "High"),
    "palatable_human": ("Yes", "No"),
}
NUMERIC_FIELDS = (
    "height_mature_ft",
    "temp_min_f",
    "ph_min",
    "ph_max",
    "precip_min_in",
    "precip_max_in",
)
FREE_TEXT_FIELDS = ("family", "genus", "bloom_period")
BOUNDS = ("gt", "gte", "lt", "lte")

REWRITE_SYSTEM = f"""You prepare user questions for retrieval over USDA PLANTS species documents.

Return JSON with:
  "rewritten": a keyword-rich search query. Expand horticultural jargon, add the
      botanical vocabulary USDA uses (e.g. "part shade" -> "shade tolerance intermediate
      tolerant"), resolve common names to genus when you are confident, and keep any Latin
      name or USDA symbol verbatim.
  "filters": hard metadata filters, ONLY when the user states an unambiguous constraint.
      Available fields: {json.dumps(FILTERABLE, indent=2)}
      Use a list for "any of". Use {{"lte": x}} / {{"gte": x}} for numeric bounds.
      Two derived constraints, which you must emit instead of guessing at the raw fields:
        "hardiness_zone": <int> for "zone 5", "USDA zone 6b" etc.
        "soil_ph": <number> for "my soil is pH 6.5" (the plant must tolerate that value).
      Omit a field entirely rather than emitting an empty object, a null or a guess;
      an over-filtered search returns nothing.
      Categorical values must be copied verbatim from the vocabulary listed for the field.
  "archetype": one of "name_lookup", "trait_filter", "site_recommendation",
      "care_howto", "comparison", "safety", "other".

Output JSON only."""

PROMPTS = {
    "v1": """You are a plant-selection assistant grounded in USDA PLANTS data.
Answer the question using ONLY the numbered context. Be concise (max 150 words).
Every sentence that states a fact ends with the [n] of the context item it came from; use
only numbers that appear in the context and never merge them as [1, 2] (write [1][2]).
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
- Every bullet ends with the [n] of the context item it came from; use only numbers that
  appear in the context and never merge them as [1, 2] (write [1][2]).
- If the context contains nothing relevant, reply exactly: "The USDA PLANTS knowledge base
  in this app does not contain enough information to answer that." """,
    "v3": """You are a plant-selection assistant grounded in USDA PLANTS data.
Use ONLY the numbered context.

Return, in order:
1. **Answer** — one or two sentences.
2. **Candidates** — a markdown table with columns:
   Species | Habit | Mature ht (ft) | Shade | Drought | pH | Source.
   Put the [n] citation in the Source column, and also cite [n] in the Answer sentences.
   Use only numbers that appear in the context; never merge them as [1, 2] (write [1][2]).
   Use "—" for values missing from the context.
3. **Caveats** — anything the context cannot confirm, or constraints not verifiable.

If the context contains nothing relevant, reply exactly: "The USDA PLANTS knowledge base in
this app does not contain enough information to answer that." """,
}

JUDGE_SYSTEM = """You grade a plant assistant's answer against the retrieved context.

Context items are numbered [1]..[N]; the answer cites them as [n]. A citation counts as
correct when the cited item supports the claim it is attached to, and one citation at the
end of a sentence, bullet or table row covers that whole sentence, bullet or row. A table
cell holding [n] cites the row it sits in. Do not require a citation on the opening summary
sentence, on caveats, or on a statement that the context lacks something.

Score each 1-5:
  relevance: does it answer the question asked?
  groundedness: is every factual claim supported by the context (5 = fully, 1 = fabricated)?
  citation_validity:
    5 = every claim-bearing sentence/bullet/row carries a citation and each points at a
        context item that supports it
    4 = all citations are correct, one claim-bearing line is missing one
    3 = citations present, one points at the wrong item
    2 = several citations are missing or point at the wrong item
    1 = no [n] citations at all, or the numbers are outside 1..N
Also set "hallucinated": true if any claim contradicts or is absent from the context.

Return JSON: {"relevance": int, "groundedness": int, "citation_validity": int,
"hallucinated": bool, "reason": "one sentence naming any miscited claim"}"""


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


def _numeric_bounds(value: Any) -> dict[str, float] | None:
    """Coerce a model-emitted numeric constraint into Qdrant range bounds."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return {"lte": float(value)}
    if not isinstance(value, dict):
        return None
    bounds = {
        key: float(bound)
        for key, bound in value.items()
        if key in BOUNDS and isinstance(bound, int | float) and not isinstance(bound, bool)
    }
    return bounds or None


def _categorical(field: str, value: Any) -> list[str] | None:
    """Keep only values that exist in the USDA vocabulary, matched case-insensitively."""
    vocab = CATEGORICAL_VOCAB[field]
    lookup = {allowed.lower(): allowed for allowed in vocab}
    values = value if isinstance(value, list) else [value]
    kept = [lookup[str(v).strip().lower()] for v in values if str(v).strip().lower() in lookup]
    return kept or None


def sanitize_filters(filters: Any) -> dict[str, Any]:
    """Drop everything the extractor cannot justify and expand the derived constraints.

    An empty object, an unknown field or a value outside the USDA vocabulary silently
    removes every correct result, which is the dominant failure mode for trait and site
    queries, so anything not understood here is discarded rather than passed to Qdrant.
    """
    if not isinstance(filters, dict):
        return {}
    clean: dict[str, Any] = {}
    for field, value in filters.items():
        if value in (None, "", [], {}):
            continue
        if field in CATEGORICAL_VOCAB:
            values = _categorical(field, value)
            if values:
                clean[field] = values
        elif field in NUMERIC_FIELDS:
            bounds = _numeric_bounds(value)
            if bounds:
                clean[field] = bounds
        elif field in FREE_TEXT_FIELDS:
            values = [str(v).strip() for v in (value if isinstance(value, list) else [value])]
            values = [v for v in values if v]
            if values:
                clean[field] = values

    zone = filters.get("hardiness_zone")
    if isinstance(zone, str):
        zone = zone.strip().rstrip("ab")
    try:
        zone = int(zone)
    except (TypeError, ValueError):
        zone = None
    if zone in ZONE_MIN_TEMP_F:
        # hardy *to* the zone: the species must survive at least that low
        clean["temp_min_f"] = {"lte": ZONE_MIN_TEMP_F[zone]}

    soil_ph = filters.get("soil_ph")
    if isinstance(soil_ph, int | float) and not isinstance(soil_ph, bool):
        # tolerance is a containment test, not a bound on either endpoint
        clean["ph_min"] = {"lte": float(soil_ph)}
        clean["ph_max"] = {"gte": float(soil_ph)}
    return clean


def rewrite_query(question: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, meta = _chat(REWRITE_SYSTEM, question, json_mode=True)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    return (
        {
            "rewritten": parsed.get("rewritten") or question,
            "filters": sanitize_filters(parsed.get("filters")),
            "archetype": parsed.get("archetype") or "other",
        },
        meta,
    )


def format_context(hits: list[dict[str, Any]]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        payload = hit.get("payload") or {}
        header = f"[{index}] {hit.get('title')}"
        if payload.get("symbol"):
            header += f" | USDA symbol: {payload['symbol']}"
        header += f" | section: {hit.get('section')}"
        if payload.get("source_url"):
            header += " | USDA PDF"
        blocks.append(f"{header}\n{hit.get('text')}")
    return "\n\n".join(blocks)


def answer(
    question: str, hits: list[dict[str, Any]], variant: str | None = None
) -> tuple[str, dict[str, Any]]:
    variant = variant or get_settings().prompt_variant
    system = PROMPTS.get(variant, PROMPTS["v1"])
    user = (
        f"Question: {question}\n\n"
        f"Context items are numbered 1..{len(hits)}; cite only those numbers.\n\n"
        f"Context:\n{format_context(hits)}"
    )
    text, meta = _chat(system, user, temperature=0.1)
    meta["prompt_variant"] = variant
    return text, meta


def judge(question: str, hits: list[dict[str, Any]], generated: str) -> dict[str, Any]:
    settings = get_settings()
    user = (
        f"Question: {question}\n\n"
        f"Context (N={len(hits)}):\n{format_context(hits)}\n\n"
        f"Answer:\n{generated}"
    )
    raw, meta = _chat(JUDGE_SYSTEM, user, model=settings.openai_judge_model, json_mode=True)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    parsed["_meta"] = meta
    return parsed
