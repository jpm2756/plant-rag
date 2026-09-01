import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.config import REPO_ROOT, get_settings

SCHEMA_PATH = REPO_ROOT / "sql" / "schema.sql"


@contextmanager
def connect(autocommit: bool = True) -> Iterator[psycopg.Connection]:
    settings = get_settings()
    with psycopg.connect(
        settings.postgres_dsn, autocommit=autocommit, row_factory=dict_row
    ) as conn:
        yield conn


def init_schema() -> None:
    with connect() as conn:
        conn.execute(SCHEMA_PATH.read_text())


def upsert_species(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    columns = list(rows[0].keys())
    placeholders = ", ".join(["%s"] * len(columns))
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c != "id")
    sql = (
        f"INSERT INTO species ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT (id) DO UPDATE SET {updates}"
    )
    values = [
        tuple(
            Jsonb(r[c])
            if isinstance(r[c], dict | list) and c.endswith(("_status", "characteristics"))
            else r[c]
            for c in columns
        )
        for r in rows
    ]
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(sql, values)
    return len(rows)


def replace_documents(species_id: int, docs: list[dict[str, Any]]) -> int:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE species_id = %s", (species_id,))
        if not docs:
            return 0
        cur.executemany(
            """
            INSERT INTO documents
                (doc_id, species_id, section, chunk_index, title, text, content_hash, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                text = EXCLUDED.text,
                title = EXCLUDED.title,
                content_hash = EXCLUDED.content_hash,
                metadata = EXCLUDED.metadata
            """,
            [
                (
                    d["doc_id"],
                    d["species_id"],
                    d["section"],
                    d["chunk_index"],
                    d["title"],
                    d["text"],
                    d["content_hash"],
                    Jsonb(d["metadata"]),
                )
                for d in docs
            ],
        )
    return len(docs)


def fetch_documents(limit: int | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM documents ORDER BY species_id, section, chunk_index"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with connect() as conn:
        return list(conn.execute(sql).fetchall())


def fetch_species(species_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM species WHERE id = %s", (species_id,)).fetchone()


SPECIES_COLUMNS = (
    "id, symbol, scientific_name, scientific_no_author, common_name, family, genus, "
    "duration, growth_habit, wetland_status, profile_url, ph_min, ph_max, height_mature_ft, "
    "temp_min_f, precip_min_in, precip_max_in, drought_tolerance, shade_tolerance, "
    "salinity_tolerance, fire_tolerance, moisture_use, growth_rate, lifespan, bloom_period, "
    "flower_color, toxicity, nitrogen_fixation, palatable_human, n_characteristics"
)


def search_species(name: str, limit: int = 5) -> list[dict[str, Any]]:
    """Exact-ish name/symbol lookup — the tool the LLM calls for 'what is X?'."""
    pattern = f"%{name.strip().lower()}%"
    with connect() as conn:
        return list(
            conn.execute(
                f"""
                SELECT {SPECIES_COLUMNS} FROM species
                WHERE lower(symbol) = %s
                   OR lower(scientific_no_author) LIKE %s
                   OR lower(scientific_name) LIKE %s
                   OR lower(common_name) LIKE %s
                   OR EXISTS (
                        SELECT 1 FROM unnest(other_common_names) AS alias
                        WHERE lower(alias) LIKE %s
                   )
                ORDER BY n_characteristics DESC
                LIMIT %s
                """,
                (name.strip().lower(), pattern, pattern, pattern, pattern, limit),
            ).fetchall()
        )


def filter_species(where_sql: str, params: tuple, limit: int = 10) -> list[dict[str, Any]]:
    """Structured trait filter used by the numeric-constraint tool."""
    with connect() as conn:
        return list(
            conn.execute(
                f"SELECT {SPECIES_COLUMNS} FROM species WHERE {where_sql} "
                f"ORDER BY n_characteristics DESC LIMIT %s",
                (*params, limit),
            ).fetchall()
        )


def corpus_stats() -> dict[str, Any]:
    with connect() as conn:
        return conn.execute(
            """
            SELECT (SELECT count(*) FROM species)   AS species,
                   (SELECT count(*) FROM documents) AS documents,
                   (SELECT count(*) FROM documents WHERE section IN ('factsheet', 'plantguide'))
                       AS pdf_chunks,
                   (SELECT count(*) FROM conversations) AS conversations
            """
        ).fetchone()


def log_conversation(record: dict[str, Any]) -> str:
    conversation_id = record.get("id") or str(uuid.uuid4())
    record["id"] = conversation_id
    for key in ("filters", "retrieved"):
        if isinstance(record.get(key), dict | list):
            record[key] = Jsonb(record[key])
    columns = list(record.keys())
    sql = (
        f"INSERT INTO conversations ({', '.join(columns)}) "
        f"VALUES ({', '.join(['%s'] * len(columns))})"
    )
    with connect() as conn:
        conn.execute(sql, tuple(record[c] for c in columns))
    return conversation_id


def log_feedback(conversation_id: str, rating: int, comment: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO feedback (conversation_id, rating, comment) VALUES (%s, %s, %s)",
            (conversation_id, rating, comment),
        )


def as_json(value: Any) -> Any:
    """psycopg returns JSONB as python objects already; keep this for str payloads."""
    if isinstance(value, str):
        return json.loads(value)
    return value
