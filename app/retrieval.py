"""Retrieval strategies. The five modes below are exactly what the eval suite scores."""

from __future__ import annotations

import time
from typing import Any

from app.config import get_settings
from app.embeddings import rerank
from app.vectorstore import collection_name, fuse_rrf, search_dense, search_sparse

MODES = ("sparse", "dense", "hybrid", "hybrid_rerank", "hybrid_rerank_norewrite")


def _dedupe_by_species(hits: list[dict[str, Any]], per_species: int = 2) -> list[dict[str, Any]]:
    """Keep the corpus diverse: at most N chunks of the same species in the context."""
    seen: dict[int, int] = {}
    kept = []
    for hit in hits:
        species_id = hit.get("species_id")
        count = seen.get(species_id, 0)
        if count >= per_species:
            continue
        seen[species_id] = count + 1
        kept.append(hit)
    return kept


def retrieve(
    query: str,
    mode: str | None = None,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    candidate_k: int | None = None,
    collection: str | None = None,
    dense_model_name: str | None = None,
    relax_empty: bool = True,
) -> dict[str, Any]:
    """Run one retrieval mode and return hits plus per-stage timings."""
    settings = get_settings()
    mode = (mode or settings.retrieval_mode).removesuffix("_norewrite")
    top_k = top_k or settings.top_k
    candidate_k = candidate_k or settings.candidate_k
    collection = collection or collection_name(dense_model_name)

    started = time.perf_counter()
    arms: list[list[dict[str, Any]]] = []
    if mode in ("sparse", "hybrid", "hybrid_rerank"):
        arms.append(search_sparse(query, candidate_k, filters, name=collection))
    if mode in ("dense", "hybrid", "hybrid_rerank"):
        arms.append(
            search_dense(
                query, candidate_k, filters, name=collection, dense_model_name=dense_model_name
            )
        )
    candidates = arms[0] if len(arms) == 1 else fuse_rrf(arms, k=settings.rrf_k)

    relaxed = False
    if not candidates and filters and relax_empty:
        # Over-eager extracted filters are the main cause of empty results; retry unfiltered
        # rather than telling the user there is nothing.
        relaxed = True
        result = retrieve(
            query,
            mode=mode,
            filters=None,
            top_k=top_k,
            candidate_k=candidate_k,
            collection=collection,
            dense_model_name=dense_model_name,
            relax_empty=False,
        )
        result["filters_relaxed"] = True
        return result
    retrieve_ms = int((time.perf_counter() - started) * 1000)

    rerank_ms = 0
    if mode == "hybrid_rerank" and candidates:
        started = time.perf_counter()
        scores = rerank(query, [f"{c['title']}\n{c['text']}" for c in candidates])
        for hit, score in zip(candidates, scores, strict=False):
            hit["rerank_score"] = float(score)
        candidates = sorted(candidates, key=lambda h: h["rerank_score"], reverse=True)
        rerank_ms = int((time.perf_counter() - started) * 1000)

    hits = _dedupe_by_species(candidates)[:top_k]
    for rank, hit in enumerate(hits, start=1):
        hit["rank"] = rank
    return {
        "hits": hits,
        "mode": mode,
        "n_candidates": len(candidates),
        "latency_retrieve_ms": retrieve_ms,
        "latency_rerank_ms": rerank_ms,
        "filters_relaxed": relaxed,
    }
