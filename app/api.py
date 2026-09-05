"""FastAPI surface: /ask, /feedback, /species, /stats."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import db, flow, tools
from app.config import get_settings
from app.retrieval import MODES
from app.vectorstore import collection_name, count_points

app = FastAPI(title="PlantRAG", version="0.1.0", description="Hybrid RAG over USDA PLANTS")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    retrieval_mode: str | None = None
    prompt_variant: str | None = None
    use_rewrite: bool | None = None
    use_tools: bool = True
    judge: bool = False


class FeedbackRequest(BaseModel):
    conversation_id: str
    rating: int = Field(ge=-1, le=1)
    comment: str | None = None


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict[str, Any]:
    settings = get_settings()
    payload: dict[str, Any] = {
        "retrieval_mode": settings.retrieval_mode,
        "retrieval_modes": list(MODES),
        "filter_mode": settings.filter_mode,
        "use_rewrite": bool(settings.use_rewrite),
        "prompt_variant": settings.prompt_variant,
        "model": settings.openai_model,
        "dense_model": settings.dense_model,
        "collection": collection_name(),
    }
    try:
        payload["corpus"] = db.corpus_stats()
    except Exception as exc:  # noqa: BLE001 - stats must not 500 the UI
        payload["corpus_error"] = str(exc)
    try:
        payload["vector_points"] = count_points()
    except Exception as exc:  # noqa: BLE001
        payload["vector_error"] = str(exc)
    return payload


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    if request.retrieval_mode and request.retrieval_mode not in MODES:
        raise HTTPException(400, f"retrieval_mode must be one of {MODES}")
    return flow.ask(
        request.question,
        mode=request.retrieval_mode,
        prompt_variant=request.prompt_variant,
        use_rewrite=request.use_rewrite,
        use_tools=request.use_tools,
        judge=request.judge,
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict[str, str]:
    if request.rating == 0:
        raise HTTPException(400, "rating must be 1 or -1")
    try:
        db.log_feedback(request.conversation_id, request.rating, request.comment)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"could not store feedback: {exc}") from exc
    return {"status": "recorded"}


@app.get("/species")
def species(name: str) -> dict[str, Any]:
    return tools.lookup_species(name)


@app.get("/species/{species_id}")
def species_by_id(species_id: int) -> dict[str, Any]:
    row = db.fetch_species(species_id)
    if row is None:
        raise HTTPException(404, "unknown species id")
    return row
