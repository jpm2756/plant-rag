"""Embedding + re-ranking models (fastembed / ONNX, CPU friendly)."""

from __future__ import annotations

import os
from collections.abc import Iterable
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config import get_settings

DENSE_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


THREADS = os.cpu_count() or 1


@lru_cache
def dense_model(name: str | None = None) -> TextEmbedding:
    return TextEmbedding(model_name=name or get_settings().dense_model, threads=THREADS)


@lru_cache
def sparse_model(name: str | None = None) -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=name or get_settings().sparse_model, threads=THREADS)


@lru_cache
def cross_encoder(name: str | None = None) -> TextCrossEncoder:
    return TextCrossEncoder(model_name=name or get_settings().rerank_model, threads=THREADS)


def dense_dim(name: str | None = None) -> int:
    name = name or get_settings().dense_model
    if name in DENSE_DIMS:
        return DENSE_DIMS[name]
    return len(next(iter(dense_model(name).embed(["dimension probe"]))))


def embed_dense(texts: Iterable[str], name: str | None = None) -> list[list[float]]:
    return [vector.tolist() for vector in dense_model(name).embed(list(texts))]


def embed_dense_query(text: str, name: str | None = None) -> list[float]:
    return next(iter(dense_model(name).query_embed([text]))).tolist()


def embed_sparse(texts: Iterable[str], name: str | None = None):
    return list(sparse_model(name).embed(list(texts)))


def embed_sparse_query(text: str, name: str | None = None):
    return next(iter(sparse_model(name).query_embed([text])))


def rerank(query: str, documents: list[str], name: str | None = None) -> list[float]:
    if not documents:
        return []
    return list(cross_encoder(name).rerank(query, documents))
