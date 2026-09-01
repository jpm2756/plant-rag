"""End-to-end query flow: rewrite -> filters -> hybrid retrieve -> rerank -> tools -> answer."""

from __future__ import annotations

import re
import time
from typing import Any

from app import db, llm, tools
from app.config import get_settings
from app.retrieval import retrieve

INSUFFICIENT = "does not contain enough information"
CITATION_RE = re.compile(r"\[(\d+)\]")


def _tool_context(question: str, plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Run exact lookups where vector search is structurally weak."""
    used: list[str] = []
    blocks: list[dict[str, Any]] = []
    filters = plan.get("filters") or {}
    numeric = {k: v for k, v in filters.items() if isinstance(v, dict)}

    if plan.get("archetype") == "name_lookup":
        result = tools.lookup_species(plan.get("rewritten") or question)
        used.append("lookup_species")
        if result["matches"]:
            blocks.append(
                {
                    "title": "Exact name lookup (USDA structured record)",
                    "section": "tool:lookup_species",
                    "text": _render_rows(result["matches"]),
                    "payload": {"tool": "lookup_species", "in_corpus": result["in_corpus"]},
                    "species_id": None,
                    "score": None,
                    "arms": ["tool"],
                }
            )
    if numeric:
        result = tools.filter_species(filters)
        used.append("filter_species")
        if result["matches"]:
            blocks.append(
                {
                    "title": f"Exact trait filter {filters}",
                    "section": "tool:filter_species",
                    "text": _render_rows(result["matches"]),
                    "payload": {"tool": "filter_species", "n": result["n"]},
                    "species_id": None,
                    "score": None,
                    "arms": ["tool"],
                }
            )
    return blocks, used


def _render_rows(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        parts = [
            f"{row.get('common_name') or ''} ({row.get('scientific_name')}) [{row.get('symbol')}]"
        ]
        for label, key in (
            ("habit", "growth_habit"),
            ("duration", "duration"),
            ("mature height ft", "height_mature_ft"),
            ("shade", "shade_tolerance"),
            ("drought", "drought_tolerance"),
            ("pH min", "ph_min"),
            ("pH max", "ph_max"),
            ("min temp F", "temp_min_f"),
            ("toxicity", "toxicity"),
        ):
            value = row.get(key)
            if value in (None, "", []):
                continue
            if isinstance(value, list):
                value = "/".join(str(v) for v in value)
            parts.append(f"{label}: {value}")
        lines.append("- " + "; ".join(parts))
    return "\n".join(lines)


def ask(
    question: str,
    mode: str | None = None,
    prompt_variant: str | None = None,
    use_rewrite: bool = True,
    use_tools: bool = True,
    judge: bool = False,
    log: bool = True,
) -> dict[str, Any]:
    settings = get_settings()
    mode = mode or settings.retrieval_mode
    started = time.perf_counter()

    rewrite_ms = 0
    plan: dict[str, Any] = {"rewritten": question, "filters": {}, "archetype": "other"}
    rewrite_meta: dict[str, Any] = {}
    if use_rewrite:
        stage = time.perf_counter()
        plan, rewrite_meta = llm.rewrite_query(question)
        rewrite_ms = int((time.perf_counter() - stage) * 1000)

    retrieved = retrieve(plan["rewritten"], mode=mode, filters=plan.get("filters") or None)
    hits = retrieved["hits"]

    tool_blocks, tools_used = ([], [])
    if use_tools:
        tool_blocks, tools_used = _tool_context(question, plan)

    context = tool_blocks + hits
    stage = time.perf_counter()
    text, answer_meta = llm.answer(question, context, variant=prompt_variant)
    generate_ms = int((time.perf_counter() - stage) * 1000)

    cited = sorted({int(n) for n in CITATION_RE.findall(text)})
    cited_symbols = [
        (context[i - 1].get("payload") or {}).get("symbol")
        for i in cited
        if 0 < i <= len(context) and (context[i - 1].get("payload") or {}).get("symbol")
    ]
    total_ms = int((time.perf_counter() - started) * 1000)
    cost = float(rewrite_meta.get("cost_usd", 0.0)) + float(answer_meta.get("cost_usd", 0.0))

    result: dict[str, Any] = {
        "question": question,
        "rewritten_query": plan["rewritten"],
        "archetype": plan.get("archetype"),
        "filters": plan.get("filters") or {},
        "filters_relaxed": retrieved.get("filters_relaxed", False),
        "retrieval_mode": retrieved["mode"],
        "prompt_variant": answer_meta.get("prompt_variant"),
        "model": answer_meta.get("model"),
        "answer": text,
        "cited_symbols": cited_symbols,
        "tools_used": tools_used,
        "n_results": len(hits),
        "insufficient": INSUFFICIENT in text.lower(),
        "sources": [
            {
                "doc_id": hit.get("doc_id"),
                "context_index": len(tool_blocks) + index,
                "species_id": hit.get("species_id"),
                "title": hit.get("title"),
                "section": hit.get("section"),
                "score": hit.get("score"),
                "rerank_score": hit.get("rerank_score"),
                "arms": hit.get("arms"),
                "symbol": (hit.get("payload") or {}).get("symbol"),
                "profile_url": (hit.get("payload") or {}).get("profile_url"),
                "source_url": (hit.get("payload") or {}).get("source_url"),
                "text": hit.get("text"),
            }
            for index, hit in enumerate(hits, start=1)
        ],
        "latency_ms": total_ms,
        "latency_rewrite_ms": rewrite_ms,
        "latency_retrieve_ms": retrieved["latency_retrieve_ms"],
        "latency_rerank_ms": retrieved["latency_rerank_ms"],
        "latency_generate_ms": generate_ms,
        "prompt_tokens": int(rewrite_meta.get("prompt_tokens", 0))
        + int(answer_meta.get("prompt_tokens", 0)),
        "completion_tokens": int(rewrite_meta.get("completion_tokens", 0))
        + int(answer_meta.get("completion_tokens", 0)),
        "cost_usd": round(cost, 6),
    }

    if judge:
        verdict = llm.judge(question, context, text)
        result["judge"] = verdict
        result["judge_relevance"] = verdict.get("relevance")
        result["judge_groundedness"] = verdict.get("groundedness")
        result["judge_hallucinated"] = verdict.get("hallucinated")

    if log:
        record = {
            key: result.get(key)
            for key in (
                "question",
                "rewritten_query",
                "archetype",
                "filters",
                "retrieval_mode",
                "prompt_variant",
                "model",
                "answer",
                "cited_symbols",
                "tools_used",
                "n_results",
                "insufficient",
                "latency_ms",
                "latency_rewrite_ms",
                "latency_retrieve_ms",
                "latency_rerank_ms",
                "latency_generate_ms",
                "prompt_tokens",
                "completion_tokens",
                "cost_usd",
                "judge_relevance",
                "judge_groundedness",
                "judge_hallucinated",
            )
            if key in result
        }
        record["retrieved"] = [
            {k: s[k] for k in ("doc_id", "symbol", "section", "score", "arms")}
            for s in result["sources"]
        ]
        result["conversation_id"] = db.log_conversation(record)

    return result
