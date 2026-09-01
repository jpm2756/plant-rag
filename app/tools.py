"""Exact-lookup tools the agent can call when vector search is the wrong instrument.

Retrieval is good at "something like X"; these answer "exactly X" — a named species,
a hard numeric constraint, or a name that is not in the ingested corpus at all.
"""

from __future__ import annotations

from typing import Any

from app import db
from ingestion.usda_client import UsdaClient

# extracted-filter key -> SQL predicate template
SQL_FIELDS = {
    "growth_habit": "%s = ANY(growth_habit)",
    "duration": "%s = ANY(duration)",
    "family": "lower(family) = lower(%s)",
    "genus": "lower(genus) = lower(%s)",
    "drought_tolerance": "drought_tolerance = %s",
    "shade_tolerance": "shade_tolerance = %s",
    "salinity_tolerance": "salinity_tolerance = %s",
    "fire_tolerance": "fire_tolerance = %s",
    "moisture_use": "moisture_use = %s",
    "growth_rate": "growth_rate = %s",
    "toxicity": "toxicity = %s",
    "nitrogen_fixation": "nitrogen_fixation = %s",
    "palatable_human": "palatable_human = %s",
}
NUMERIC_FIELDS = {
    "height_mature_ft",
    "temp_min_f",
    "ph_min",
    "ph_max",
    "precip_min_in",
    "precip_max_in",
}
OPERATORS = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_species",
            "description": (
                "Exact species lookup by common name, scientific name or USDA symbol. "
                "Use for 'what is X' / 'tell me about X' questions and to confirm identity."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_species",
            "description": (
                "Exact structured filter over USDA traits. Use when the question has hard "
                "numeric or categorical constraints (mature height, pH, min temperature, "
                "tolerance ratings) that semantic search cannot enforce."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "description": (
                            'e.g. {"growth_habit": "Shrub", "height_mature_ft": {"lte": 6}, '
                            '"shade_tolerance": "Tolerant"}'
                        ),
                    }
                },
                "required": ["filters"],
            },
        },
    },
]


def build_where(filters: dict[str, Any]) -> tuple[str, tuple]:
    clauses: list[str] = []
    params: list[Any] = []
    for key, value in (filters or {}).items():
        if key in NUMERIC_FIELDS and isinstance(value, dict):
            for operator, sql_operator in OPERATORS.items():
                if operator in value:
                    clauses.append(f"{key} {sql_operator} %s")
                    params.append(float(value[operator]))
        elif key in SQL_FIELDS:
            values = value if isinstance(value, list) else [value]
            ors = []
            for item in values:
                ors.append(SQL_FIELDS[key])
                params.append(item)
            clauses.append("(" + " OR ".join(ors) + ")")
    return (" AND ".join(clauses) or "TRUE"), tuple(params)


def lookup_species(name: str) -> dict[str, Any]:
    rows = db.search_species(name)
    if not rows:
        # Not in the 2,186-species corpus: fall back to the live USDA taxonomy search.
        try:
            remote = UsdaClient().search(name)[:5]
        except Exception:  # noqa: BLE001 - the tool must never break the answer path
            remote = []
        return {
            "source": "usda_api",
            "in_corpus": False,
            "matches": [
                {
                    "symbol": r.get("Symbol"),
                    "scientific_name": r.get("ScientificName"),
                    "common_name": r.get("CommonName"),
                }
                for r in remote
            ],
        }
    return {"source": "postgres", "in_corpus": True, "matches": rows}


def filter_species(filters: dict[str, Any]) -> dict[str, Any]:
    where_sql, params = build_where(filters)
    rows = db.filter_species(where_sql, params, limit=10)
    return {"source": "postgres", "filters": filters, "n": len(rows), "matches": rows}


DISPATCH = {"lookup_species": lookup_species, "filter_species": filter_species}


def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    handler = DISPATCH.get(name)
    if handler is None:
        return {"error": f"unknown tool {name}"}
    return handler(**arguments)
