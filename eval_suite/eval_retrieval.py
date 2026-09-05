"""Score five retrieval approaches on the ground-truth set.

Modes
-----
sparse                    BM25 only
dense                     bge-small only
hybrid                    sparse + dense fused with RRF
hybrid_rerank             + cross-encoder re-ranking
hybrid_rerank_norewrite   same as above but the raw question is used (rewriting ablation)
hybrid_rerank_dual        raw question AND rewritten query fused (keeps both signals)

Metrics: hit-rate@k and MRR@k on the exact document, plus a species-level hit rate
(a different section of the right species is still a useful retrieval).
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from typing import Any

from tqdm import tqdm

from app import llm
from app.config import DATA_DIR, get_settings
from app.retrieval import MODES, retrieve

GROUND_TRUTH = DATA_DIR / "ground_truth.json"
OUT_JSON = DATA_DIR / "retrieval_eval.json"
OUT_MD = DATA_DIR / "retrieval_eval.md"


def _rewrite_cache(questions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Rewrite each question once and reuse it for every mode (same input, fair + cheap)."""
    cache: dict[str, dict[str, Any]] = {}
    for row in tqdm(questions, desc="rewrite"):
        question = row["question"]
        if question in cache:
            continue
        plan, _ = llm.rewrite_query(question)
        cache[question] = plan
    return cache


def evaluate(mode: str, questions: list[dict[str, Any]], rewrites, top_k: int) -> dict[str, Any]:
    hits_at_k, reciprocal, species_hits = [], [], []
    per_archetype: dict[str, list[int]] = defaultdict(list)
    for row in tqdm(questions, desc=mode):
        if mode.endswith("_norewrite"):
            query, filters = row["question"], None
        else:
            plan = rewrites[row["question"]]
            query, filters = plan["rewritten"], plan.get("filters") or None
        result = retrieve(
            query, mode=mode, filters=filters, top_k=top_k, extra_queries=[row["question"]]
        )
        doc_ids = [h["doc_id"] for h in result["hits"]]
        species_ids = [h["species_id"] for h in result["hits"]]
        hit = row["doc_id"] in doc_ids
        hits_at_k.append(int(hit))
        species_hits.append(int(row["species_id"] in species_ids))
        reciprocal.append(1 / (doc_ids.index(row["doc_id"]) + 1) if hit else 0.0)
        per_archetype[row.get("archetype", "other")].append(int(hit))
    return {
        "mode": mode,
        "n": len(questions),
        f"hit_rate@{top_k}": round(statistics.mean(hits_at_k), 4),
        f"mrr@{top_k}": round(statistics.mean(reciprocal), 4),
        f"species_hit_rate@{top_k}": round(statistics.mean(species_hits), 4),
        "per_archetype": {
            key: round(statistics.mean(values), 4) for key, values in sorted(per_archetype.items())
        },
    }


def main(top_k: int | None = None) -> int:
    settings = get_settings()
    top_k = top_k or settings.top_k
    if not GROUND_TRUTH.exists():
        print("run eval_suite.generate_ground_truth first", file=sys.stderr)
        return 1
    questions = json.loads(GROUND_TRUTH.read_text())
    rewrites = _rewrite_cache(questions)

    results = [evaluate(mode, questions, rewrites, top_k) for mode in MODES]
    best = max(results, key=lambda r: (r[f"hit_rate@{top_k}"], r[f"mrr@{top_k}"]))
    payload = {"top_k": top_k, "results": results, "best_mode": best["mode"]}
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    header = f"| mode | hit-rate@{top_k} | MRR@{top_k} | species hit-rate@{top_k} |"
    lines = [
        f"# Retrieval evaluation (n={len(questions)})",
        "",
        header,
        "| --- | --- | --- | --- |",
    ]
    for row in results:
        lines.append(
            f"| `{row['mode']}` | {row[f'hit_rate@{top_k}']:.3f} | {row[f'mrr@{top_k}']:.3f} | "
            f"{row[f'species_hit_rate@{top_k}']:.3f} |"
        )
    lines += ["", f"Best: **{best['mode']}**", "", "## Hit-rate by query archetype", ""]
    archetypes = sorted({a for row in results for a in row["per_archetype"]})
    lines.append("| mode | " + " | ".join(archetypes) + " |")
    lines.append("| --- |" + " --- |" * len(archetypes))
    for row in results:
        cells = [f"{row['per_archetype'].get(a, float('nan')):.3f}" for a in archetypes]
        lines.append(f"| `{row['mode']}` | " + " | ".join(cells) + " |")
    OUT_MD.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else None))
