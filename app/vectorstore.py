"""Qdrant access: indexing and the individual retrieval arms.

Sparse and dense arms are queried separately and fused in Python so that (a) the
evaluation suite can score each arm on its own and (b) the UI can show which arm
retrieved each chunk.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.embeddings import (
    dense_dim,
    embed_dense,
    embed_dense_query,
    embed_sparse,
    embed_sparse_query,
)

DENSE = "dense"
SPARSE = "sparse"

# payload fields that get a Qdrant index so filtering stays fast
KEYWORD_FIELDS = (
    "symbol",
    "scientific_no_author",
    "common_name",
    "family",
    "genus",
    "growth_habit",
    "duration",
    "native_regions",
    "drought_tolerance",
    "shade_tolerance",
    "salinity_tolerance",
    "fire_tolerance",
    "moisture_use",
    "growth_rate",
    "lifespan",
    "toxicity",
    "nitrogen_fixation",
    "palatable_human",
    "section",
)
FLOAT_FIELDS = (
    "ph_min",
    "ph_max",
    "height_mature_ft",
    "temp_min_f",
    "precip_min_in",
    "precip_max_in",
)


def client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url, timeout=120)


def collection_name(dense_model_name: str | None = None, suffix: str = "") -> str:
    settings = get_settings()
    base = settings.qdrant_collection
    model = (dense_model_name or settings.dense_model).split("/")[-1].replace(".", "_")
    return f"{base}__{model}{suffix}"


def recreate_collection(name: str, dense_model_name: str | None = None) -> None:
    qdrant = client()
    if qdrant.collection_exists(name):
        qdrant.delete_collection(name)
    qdrant.create_collection(
        collection_name=name,
        vectors_config={
            DENSE: models.VectorParams(
                size=dense_dim(dense_model_name), distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field in KEYWORD_FIELDS:
        qdrant.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
    for field in FLOAT_FIELDS:
        qdrant.create_payload_index(name, field, models.PayloadSchemaType.FLOAT)
    qdrant.create_payload_index(name, "species_id", models.PayloadSchemaType.INTEGER)


def index_documents(
    docs: list[dict[str, Any]],
    name: str | None = None,
    dense_model_name: str | None = None,
    sparse_model_name: str | None = None,
    batch_size: int = 64,
) -> int:
    name = name or collection_name(dense_model_name)
    qdrant = client()
    total = 0
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        texts = [f"{d['title']}\n{d['text']}" for d in batch]
        dense_vectors = embed_dense(texts, dense_model_name)
        sparse_vectors = embed_sparse(texts, sparse_model_name)
        points = []
        for offset, doc in enumerate(batch):
            sparse = sparse_vectors[offset]
            points.append(
                models.PointStruct(
                    id=start + offset,
                    vector={
                        DENSE: dense_vectors[offset],
                        SPARSE: models.SparseVector(
                            indices=sparse.indices.tolist(), values=sparse.values.tolist()
                        ),
                    },
                    payload={
                        **doc["metadata"],
                        "doc_id": doc["doc_id"],
                        "species_id": doc["species_id"],
                        "title": doc["title"],
                        "text": doc["text"],
                    },
                )
            )
        qdrant.upsert(collection_name=name, points=points, wait=True)
        total += len(points)
    return total


# ------------------------------------------------------------------ retrieval


def build_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    """Translate extracted filters into a Qdrant filter.

    Supported shapes::

        {"growth_habit": "Shrub"}                  # match any
        {"shade_tolerance": ["Medium", "High"]}    # match any of
        {"height_mature_ft": {"lte": 6}}           # range
        {"ph_max": {"lte": 6.5}, "ph_min": {"gte": 4}}
    """
    if not filters:
        return None
    must: list[models.Condition] = []
    for key, value in filters.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, dict):
            bounds = {k: v for k, v in value.items() if k in ("gt", "gte", "lt", "lte")}
            if bounds:
                must.append(models.FieldCondition(key=key, range=models.Range(**bounds)))
            continue
        values = value if isinstance(value, list) else [value]
        must.append(models.FieldCondition(key=key, match=models.MatchAny(any=values)))
    return models.Filter(must=must) if must else None


def _hits_to_dicts(hits, arm: str) -> list[dict[str, Any]]:
    out = []
    for rank, hit in enumerate(hits, start=1):
        payload = dict(hit.payload or {})
        out.append(
            {
                "doc_id": payload.get("doc_id"),
                "species_id": payload.get("species_id"),
                "section": payload.get("section"),
                "title": payload.get("title"),
                "text": payload.get("text"),
                "payload": payload,
                "score": float(hit.score),
                "rank": rank,
                "arms": [arm],
            }
        )
    return out


def search_dense(
    query: str,
    limit: int,
    filters: dict[str, Any] | None = None,
    name: str | None = None,
    dense_model_name: str | None = None,
) -> list[dict[str, Any]]:
    hits = (
        client()
        .query_points(
            collection_name=name or collection_name(dense_model_name),
            query=embed_dense_query(query, dense_model_name),
            using=DENSE,
            limit=limit,
            query_filter=build_filter(filters),
            with_payload=True,
        )
        .points
    )
    return _hits_to_dicts(hits, "dense")


def search_sparse(
    query: str,
    limit: int,
    filters: dict[str, Any] | None = None,
    name: str | None = None,
    sparse_model_name: str | None = None,
) -> list[dict[str, Any]]:
    sparse = embed_sparse_query(query, sparse_model_name)
    hits = (
        client()
        .query_points(
            collection_name=name or collection_name(),
            query=models.SparseVector(
                indices=sparse.indices.tolist(), values=sparse.values.tolist()
            ),
            using=SPARSE,
            limit=limit,
            query_filter=build_filter(filters),
            with_payload=True,
        )
        .points
    )
    return _hits_to_dicts(hits, "sparse")


def fuse_rrf(
    arms: list[list[dict[str, Any]]], k: int = 60, limit: int | None = None
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion over independent retrieval arms."""
    merged: dict[str, dict[str, Any]] = {}
    for hits in arms:
        for hit in hits:
            key = hit["doc_id"]
            entry = merged.get(key)
            contribution = 1.0 / (k + hit["rank"])
            if entry is None:
                merged[key] = {**hit, "score": contribution, "arms": list(hit["arms"])}
            else:
                entry["score"] += contribution
                for arm in hit["arms"]:
                    if arm not in entry["arms"]:
                        entry["arms"].append(arm)
    ranked = sorted(merged.values(), key=lambda h: h["score"], reverse=True)
    for rank, hit in enumerate(ranked, start=1):
        hit["rank"] = rank
    return ranked[:limit] if limit else ranked


def count_points(name: str | None = None) -> int:
    return client().count(collection_name=name or collection_name(), exact=True).count
