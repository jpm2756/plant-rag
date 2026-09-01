"""Compare the three answer prompts with LLM-as-judge + citation and cost metrics."""

from __future__ import annotations

import json
import random
import statistics
import sys
from typing import Any

from tqdm import tqdm

from app import flow
from app.config import DATA_DIR
from app.llm import PROMPTS

GROUND_TRUTH = DATA_DIR / "ground_truth.json"
OUT_JSON = DATA_DIR / "llm_eval.json"
OUT_MD = DATA_DIR / "llm_eval.md"


def evaluate(variant: str, questions: list[dict[str, Any]]) -> dict[str, Any]:
    relevance, groundedness, citations, hallucinated = [], [], [], []
    cited_target, latency, cost = [], [], []
    for row in tqdm(questions, desc=variant):
        result = flow.ask(row["question"], prompt_variant=variant, judge=True, log=False)
        verdict = result.get("judge") or {}
        relevance.append(verdict.get("relevance") or 0)
        groundedness.append(verdict.get("groundedness") or 0)
        citations.append(verdict.get("citation_validity") or 0)
        hallucinated.append(int(bool(verdict.get("hallucinated"))))
        target_symbol = next(
            (s["symbol"] for s in result["sources"] if s["species_id"] == row["species_id"]), None
        )
        cited_target.append(int(bool(target_symbol and target_symbol in result["cited_symbols"])))
        latency.append(result["latency_ms"])
        cost.append(result["cost_usd"])
    return {
        "variant": variant,
        "n": len(questions),
        "relevance": round(statistics.mean(relevance), 3),
        "groundedness": round(statistics.mean(groundedness), 3),
        "citation_validity": round(statistics.mean(citations), 3),
        "hallucination_rate": round(statistics.mean(hallucinated), 3),
        "cited_target_species": round(statistics.mean(cited_target), 3),
        "p50_latency_ms": int(statistics.median(latency)),
        "avg_cost_usd": round(statistics.mean(cost), 6),
    }


def main(sample: int = 30, seed: int = 11) -> int:
    if not GROUND_TRUTH.exists():
        print("run eval_suite.generate_ground_truth first", file=sys.stderr)
        return 1
    questions = json.loads(GROUND_TRUTH.read_text())
    random.seed(seed)
    questions = random.sample(questions, min(sample, len(questions)))

    results = [evaluate(variant, questions) for variant in PROMPTS]
    best = max(
        results,
        key=lambda r: (r["relevance"] + r["groundedness"] + r["citation_validity"]) / 3
        - r["hallucination_rate"],
    )
    OUT_JSON.write_text(json.dumps({"results": results, "best_variant": best["variant"]}, indent=2))

    columns = [
        "relevance",
        "groundedness",
        "citation_validity",
        "hallucination_rate",
        "cited_target_species",
        "p50_latency_ms",
        "avg_cost_usd",
    ]
    lines = [
        f"# LLM evaluation (n={results[0]['n']}, judge = gpt-4o-mini)",
        "",
        "| variant | " + " | ".join(columns) + " |",
        "| --- |" + " --- |" * len(columns),
    ]
    for row in results:
        lines.append(f"| `{row['variant']}` | " + " | ".join(str(row[c]) for c in columns) + " |")
    lines += ["", f"Best: **{best['variant']}** (set `PROMPT_VARIANT={best['variant']}`)"]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 30))
