"""dlt ingestion pipeline: USDA PLANTS API -> Postgres -> Qdrant.

Stages
------
enumerate : GET /characteristicSearchResults  -> species stubs
hydrate   : per species characteristics / profile / wetland / wildlife (cached)
docify    : pivot traits, render prose cards + section chunks -> documents
factsheet : fetch fact-sheet & plant-guide PDFs, extract + chunk -> documents
index     : dense + sparse embeddings -> Qdrant

Run everything with ``python -m ingestion.pipeline all``.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import dlt
from tqdm import tqdm

from app import db, vectorstore
from app.config import get_settings
from ingestion import factsheets
from ingestion.cards import build_documents, species_row
from ingestion.usda_client import UsdaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")


def _dlt_pipeline() -> dlt.Pipeline:
    settings = get_settings()
    return dlt.pipeline(
        pipeline_name="plant_rag",
        destination=dlt.destinations.postgres(credentials=settings.postgres_dsn),
        dataset_name="usda_raw",
        progress="log",
    )


@dlt.resource(name="species_stubs", write_disposition="replace", primary_key="id")
def species_stubs(client: UsdaClient, limit: int = 0):
    rows = client.characterized_species()
    if limit:
        rows = rows[:limit]
    log.info("enumerate: %d characterized species", len(rows))
    yield from rows


def stage_enumerate(client: UsdaClient) -> list[dict[str, Any]]:
    """Load the raw stub list into Postgres via dlt and return it."""
    settings = get_settings()
    rows = client.characterized_species()
    if settings.ingest_limit:
        rows = rows[: settings.ingest_limit]
    pipeline = _dlt_pipeline()
    info = pipeline.run(species_stubs(client, settings.ingest_limit))
    log.info("dlt: %s", str(info).splitlines()[0] if str(info) else "loaded")
    return rows


def stage_hydrate(client: UsdaClient, stubs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fetch per-species detail and write the typed `species` table."""
    db.init_schema()
    rows: list[dict[str, Any]] = []
    ids = [int(s["id"]) for s in stubs]
    stub_by_id = {int(s["id"]): s for s in stubs}
    batch_size = 40
    for start in tqdm(range(0, len(ids), batch_size), desc="hydrate"):
        chunk = ids[start : start + batch_size]
        hydrated = client.hydrate(chunk)
        batch_rows = []
        for species_id in chunk:
            record = hydrated[species_id]
            row = species_row(
                stub_by_id[species_id],
                record["profile"],
                record["characteristics"],
                record["wetland"],
            )
            batch_rows.append(row)
        db.upsert_species(batch_rows)
        rows.extend(batch_rows)
    log.info("hydrate: %d species written", len(rows))
    return rows


def stage_docify(client: UsdaClient, rows: list[dict[str, Any]], include_factsheets: bool) -> int:
    total = 0
    for row in tqdm(rows, desc="docify"):
        wildlife = client.wildlife(row["id"])
        docs = build_documents(row, wildlife)
        if include_factsheets:
            docs += factsheets.documents_for_species(client, row)
        total += db.replace_documents(row["id"], docs)
    log.info("docify: %d documents", total)
    return total


def stage_index(dense_model_name: str | None = None) -> int:
    settings = get_settings()
    docs = db.fetch_documents()
    name = vectorstore.collection_name(dense_model_name)
    log.info("index: %d documents -> %s", len(docs), name)
    vectorstore.recreate_collection(name, dense_model_name)
    count = vectorstore.index_documents(
        [
            {
                "doc_id": d["doc_id"],
                "species_id": d["species_id"],
                "title": d["title"],
                "text": d["text"],
                "metadata": d["metadata"],
            }
            for d in docs
        ],
        name=name,
        dense_model_name=dense_model_name or settings.dense_model,
    )
    log.info("index: %d points in %s", count, name)
    return count


def run_all() -> None:
    settings = get_settings()
    client = UsdaClient()
    stubs = stage_enumerate(client)
    rows = stage_hydrate(client, stubs)
    stage_docify(client, rows, bool(settings.include_factsheets))
    stage_index()


def main(argv: list[str]) -> int:
    stage = argv[1] if len(argv) > 1 else "all"
    settings = get_settings()
    client = UsdaClient()
    if stage == "all":
        run_all()
    elif stage == "enumerate":
        stage_enumerate(client)
    elif stage == "hydrate":
        stage_hydrate(client, stage_enumerate(client))
    elif stage == "docify":
        db.init_schema()
        rows = stage_hydrate(client, stage_enumerate(client))
        stage_docify(client, rows, bool(settings.include_factsheets))
    elif stage == "index":
        stage_index(argv[2] if len(argv) > 2 else None)
    else:
        print(f"unknown stage: {stage}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
