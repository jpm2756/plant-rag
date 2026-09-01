"""Generate a ground-truth question set from the indexed corpus.

For a sampled document we ask the LLM for questions a real gardener would type that
that document uniquely answers, tagged by archetype so retrieval can be scored per
query type. The document's ``doc_id`` is the relevant item.
"""

from __future__ import annotations

import json
import random
import sys

from tqdm import tqdm

from app import db, llm
from app.config import DATA_DIR

OUT_PATH = DATA_DIR / "ground_truth.json"

SYSTEM = """You write evaluation questions for a plant-search engine over USDA PLANTS data.

Given one document, write 2 questions that this document specifically answers, as a
gardener or land manager would actually type them. One question must be answerable only via
the document's traits (no species name), the other may name the species (common or Latin).
Never quote the document verbatim. 8-20 words each.

Return JSON: {"questions": [{"question": str, "archetype": str}, ...]}
archetype is one of: name_lookup, trait_filter, site_recommendation, care_howto,
comparison, safety."""


def main(sample: int = 120, seed: int = 7) -> int:
    docs = db.fetch_documents()
    if not docs:
        print("no documents; run ingestion first", file=sys.stderr)
        return 1
    random.seed(seed)
    # sample across sections so the eval isn't dominated by summary cards
    by_section: dict[str, list] = {}
    for doc in docs:
        by_section.setdefault(doc["section"], []).append(doc)
    picks = []
    per_section = max(1, sample // max(1, len(by_section)))
    for section_docs in by_section.values():
        picks += random.sample(section_docs, min(per_section, len(section_docs)))
    picks = picks[:sample]

    rows = []
    for doc in tqdm(picks, desc="ground truth"):
        prompt = f"Document ({doc['section']}): {doc['title']}\n{doc['text'][:2500]}"
        raw, _ = llm._chat(SYSTEM, prompt, json_mode=True)
        try:
            questions = json.loads(raw).get("questions", [])
        except json.JSONDecodeError:
            continue
        for item in questions:
            question = (item.get("question") or "").strip()
            if not question:
                continue
            rows.append(
                {
                    "question": question,
                    "archetype": item.get("archetype") or "other",
                    "doc_id": doc["doc_id"],
                    "species_id": doc["species_id"],
                    "section": doc["section"],
                }
            )
    OUT_PATH.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} questions -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 120))
