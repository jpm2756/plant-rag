# 🌿 PlantRAG — a hybrid RAG + agent assistant over the USDA PLANTS database

Picking the right plant for a real site is a filtering problem dressed up as a language
problem. A gardener or land manager asks *"shade-tolerant native shrubs under 6 ft for
acidic soil in the Northeast, safe around livestock"* — six constraints at once, phrased in
horticultural English. The authoritative data to answer it exists (the USDA PLANTS
database records ~80 characteristics per species plus long-form fact-sheet PDFs), but it is
only reachable through a form-based site and a REST API: you must already know which
attribute names and coded values to filter on, and nothing on that site reads the PDFs for
you. Meanwhile asking a bare LLM the same question gets you fluent, confidently wrong
plant advice — invented pH ranges, invented hardiness, invented toxicity.

**PlantRAG closes that gap**: it turns the USDA structured record and its fact sheets into a
retrievable knowledge base, translates a natural-language question into a search query plus
hard metadata filters, retrieves with hybrid (BM25 + dense) search and cross-encoder
re-ranking, and answers *only* from retrieved context with per-species citations back to
USDA. When it cannot ground an answer, it says so.

---

## Table of contents

- [What it does](#what-it-does)
- [Data source](#data-source)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Using it](#using-it)
- [Ingestion pipeline](#ingestion-pipeline)
- [Retrieval evaluation](#retrieval-evaluation)
- [LLM evaluation](#llm-evaluation)
- [Monitoring](#monitoring)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Development](#development)
- [Design notes](#design-notes)

---

## What it does

| | |
| --- | --- |
| **Knowledge base** | USDA species "plant cards" (5 prose sections each) + chunked USDA fact-sheet / plant-guide PDFs, indexed in Qdrant with dense **and** sparse vectors and structured payload filters |
| **Query understanding** | LLM rewrites the question into USDA vocabulary, extracts hard filters (`height_mature_ft <= 6`, `shade_tolerance = Tolerant`, …) and classifies the query archetype |
| **Retrieval** | BM25 + dense arms → Reciprocal Rank Fusion → cross-encoder re-rank → per-species diversification |
| **Tools** | exact Postgres lookups for named species and hard numeric constraints, with fallback to the live USDA name search when a species is outside the corpus |
| **Answering** | grounded, cited answers with an explicit "not enough information" branch |
| **Interfaces** | Streamlit chat UI + FastAPI (`/ask`, `/feedback`, `/species`, `/stats`) |
| **Evaluation** | 5 retrieval configurations and 3 answer prompts scored on an LLM-generated ground-truth set |
| **Monitoring** | every request logged to Postgres; Grafana dashboard with 9 panels; 👍/👎 + comment feedback in the UI |
| **Ops** | one `docker compose` stack, pinned dependencies, on-disk API/PDF cache for reproducible re-runs |

## Data source

[USDA PLANTS Services API](https://plantsservices.sc.egov.usda.gov/api/) —
**no API key, no signup, public US-government data**
([OpenAPI spec](https://plantsservices.sc.egov.usda.gov/swagger/v1/swagger.json)).

Measured before committing to it (sampling 40 random species):

| | USDA PLANTS | Trefle (initial candidate) |
| --- | --- | --- |
| Species with characteristics | **2,186** (178 families) | 416,473 rows… |
| Populated traits per species | **median ~80** (56–553) | ~1–5; `light`/`ph`/height on ~1% of rows |
| Common name present | **99.9%** | 9% |
| Long-form prose per species | **fact-sheet + plant-guide PDFs** | none |
| Auth | none | token required |

Trefle's public dump *is* its API's backing data, so switching to its live API would not have
fixed the sparsity — hence the move to USDA. Perenual and Permapeople need keys, OpenFarm is
dead (domain redirects to GitHub), and GBIF carries taxonomy/occurrences rather than
horticultural traits. The tradeoff accepted: USDA is US-focused and 2,186 species is small,
but every document is substantive, which is what matters for retrieval quality.

Endpoints used: `characteristicSearchResults`, `PlantCharacteristics/{id}`,
`PlantProfile/{id}`, `PlantWetland/{id}`, `PlantWildlife/{id}`, `PlantSearch?searchText=`,
plus the fact-sheet PDFs under `plants.usda.gov/DocumentLibrary/...`.

## Architecture

```
                    ┌──────────────── dlt ingestion (ingestion/pipeline.py) ────────────────┐
USDA PLANTS API ──▶ │ enumerate → hydrate (cached) → normalise traits → render plant cards │
  + fact-sheet PDFs │                            → fetch + chunk PDFs                      │
                    └───────────────┬──────────────────────────────────┬───────────────────┘
                                    ▼                                  ▼
                        Postgres (species, documents)         Qdrant (dense + sparse + payload)
                                    │                                  │
        ┌───────────────────────────┴──────────────────────────────────┴────────────────────┐
        │  FastAPI /ask  (app/flow.py)                                                      │
        │  rewrite + filter extraction → BM25 ∥ dense → RRF → cross-encoder rerank          │
        │        → exact-lookup tools (Postgres / live USDA) → grounded answer + citations   │
        └───────────────────────────┬───────────────────────────────────┬───────────────────┘
                                    ▼                                   ▼
                     Streamlit UI (retrieval panel, 👍/👎)     conversations + feedback tables
                                                                         │
                                                              Grafana (9 monitoring panels)
```

## Quickstart

Requirements: Docker + Docker Compose, and an OpenAI API key (the full build and all
evaluation sweeps cost well under $5 on `gpt-4o-mini`).

```bash
git clone <this repo> && cd plant-rag
cp .env.example .env
# put your key in .env:  OPENAI_API_KEY=sk-...
```

```bash
docker compose up -d --build postgres qdrant api ui grafana   # or: make up
docker compose run --rm ingest                                # or: make ingest
```

Ingestion takes ~45–70 min for the full corpus (USDA is rate-limited and every response is
cached to `data/`), plus embedding time for the ~18k chunks — that stage is CPU-bound and
scales with cores. To try the stack quickly, ingest a slice first:

```bash
INGEST_LIMIT=150 docker compose run --rm ingest
```

Then open:

| Service | URL |
| --- | --- |
| Streamlit UI | <http://localhost:8501> |
| FastAPI docs | <http://localhost:8000/docs> |
| Grafana (anonymous viewer) | <http://localhost:3000/d/plantrag-monitoring> |
| Qdrant dashboard | <http://localhost:6333/dashboard> |

> If port 5432 is already in use on your host, set `POSTGRES_HOST_PORT` in `.env` (the
> default is `5433`); containers always talk to Postgres on its internal port.

## Using it

UI: ask a question, then open **Retrieval detail** to see the rewritten query, the extracted
filters, which arm (`sparse`/`dense`/`tool`) produced each source, re-rank scores, per-stage
latency and token cost — then leave 👍/👎 and an optional comment. The sidebar switches
retrieval mode and prompt variant live, which is also how the ablations were explored.

API:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{
  "question": "Shade-tolerant native shrubs under 6 feet for acidic soil",
  "retrieval_mode": "hybrid_rerank_dual",
  "prompt_variant": "v1"
}' | jq '{answer, rewritten_query, filters, sources: [.sources[] | {title, section, arms}]}'

curl -s localhost:8000/feedback -H 'content-type: application/json' \
  -d '{"conversation_id": "<id from /ask>", "rating": 1, "comment": "spot on"}'

curl -s 'localhost:8000/species?name=balsam%20fir' | jq
curl -s localhost:8000/stats | jq
```

## Ingestion pipeline

Automated with **dlt** (`ingestion/pipeline.py`), runnable stage by stage
(`python -m ingestion.pipeline enumerate|hydrate|docify|index|all`):

1. **enumerate** — `characteristicSearchResults` → 2,186 non-synonym species stubs; loaded raw
   into Postgres (`usda_raw` dataset) by dlt so the untransformed source is preserved.
2. **hydrate** — characteristics + profile + wetland + wildlife per species, fetched through a
   thread pool with retry/backoff; **every response is cached as JSON on disk**, so re-runs are
   offline, fast, and reproducible even if USDA is down.
3. **docify** — pivots ~80 `{name, value}` triples into typed columns (`ph_min`,
   `height_mature_ft`, `shade_tolerance`, …; cultivar-qualified duplicates lose to the
   species-level value) and renders five prose sections per species: `summary`, `growth`,
   `site`, `propagation`, `uses`. Structured triples retrieve badly; prose retrieves well —
   and the raw values ride along in the payload as hard filters.
4. **factsheet** — downloads fact-sheet / plant-guide PDFs (cached), strips agency
   boilerplate, chunks paragraph-aware at ~1.8k chars with 300-char overlap.
5. **index** — dense (`BAAI/bge-small-en-v1.5`) + sparse (BM25, IDF-modified) vectors into one
   Qdrant collection with indexed payload fields for filtering.

Re-running is idempotent: `species` upserts by id, `documents` are replaced per species, and
the Qdrant collection is rebuilt from Postgres, so ingestion and indexing can iterate
independently.

## Retrieval evaluation

Ground truth: `python -m eval_suite.generate_ground_truth` samples documents across all
sections and asks `gpt-4o-mini` for questions each document uniquely answers — one
trait-only phrasing (no species name) and one naming the species — tagged with an archetype.
The source `doc_id` is the relevant item.

`make eval-retrieval` scores each configuration with hit-rate@5, MRR@5 and a species-level
hit rate (a different section of the right species is still useful). Full corpus, n=238:

| mode | hit-rate@5 | MRR@5 | species hit-rate@5 |
| --- | --- | --- | --- |
| `sparse` | 0.433 | 0.356 | 0.626 |
| `dense` | 0.445 | 0.402 | 0.605 |
| `hybrid` | 0.445 | 0.406 | 0.634 |
| `hybrid_rerank` | 0.483 | 0.424 | 0.651 |
| `hybrid_rerank_norewrite` | **0.622** | **0.567** | **0.744** |

Hit-rate by query archetype:

| mode | care_howto | comparison | name_lookup | safety | site_recommendation | trait_filter |
| --- | --- | --- | --- | --- | --- | --- |
| `sparse` | 0.540 | 0.500 | 0.633 | 0.630 | 0.231 | 0.282 |
| `dense` | 0.757 | 0.500 | 0.551 | 0.481 | 0.308 | 0.300 |
| `hybrid` | 0.622 | 0.500 | 0.612 | 0.518 | 0.308 | 0.309 |
| `hybrid_rerank` | 0.676 | 0.500 | 0.673 | 0.593 | 0.385 | 0.318 |
| `hybrid_rerank_norewrite` | 0.892 | 1.000 | 0.735 | 0.741 | 0.462 | 0.464 |

What the numbers say:

- **Hybrid > either arm alone, and re-ranking > hybrid** — both best-practice stages pay for
  themselves, and dense wins on paraphrased care questions while sparse wins on name lookups,
  which is the duality hybrid exists for.
- **Rewriting hurt retrieval** (0.483 → 0.622 without it). The gap is concentrated in the
  filtered archetypes (`trait_filter` 0.318 → 0.464, `site_recommendation` 0.385 → 0.462),
  because the rewrite path also applies the extracted metadata filters and one over-eager
  filter removes the target document entirely. Two fixes followed: filters are now
  **sanitised** (empty objects, unknown fields and out-of-vocabulary values dropped; hardiness
  zone → `temp_min_f` bound; soil pH → containment test on `ph_min`/`ph_max`) and applied
  **softly** by default (`FILTER_MODE=soft` fuses the filtered and unfiltered arms so a filter
  boosts instead of gates), and `hybrid_rerank_dual` fuses the raw question with the rewritten
  query so the lexical signal survives expansion.
- **Caveat on the ablation:** ground-truth questions are generated *from* the document text, so
  raw phrasings are lexically closer to their source document than a real user's would be —
  part of the no-rewrite gain is eval bias, which is why the rewrite is kept as a fused arm
  rather than deleted.
- Document-level hit-rate understates quality: five sections of the same species are near
  duplicates, so the species-level rate (0.744) is the more honest ceiling.

Results are written to `data/retrieval_eval.md` / `.json`.

## LLM evaluation

`make eval-llm` runs three answer prompts through the full flow and grades each answer with
an LLM judge (relevance / groundedness / citation validity / hallucination flag), plus a
mechanical check of whether the *target* species was actually cited, and cost & latency:

- **v1** — terse grounded answer, citations only.
- **v2** — direct answer + per-species bullets quoting concrete USDA values, explicit
  "constraint not covered" and insufficient-data branches.
- **v3** — answer + attribute comparison table + caveats section.

n=30 sampled questions, judge = `gpt-4o-mini`:

| variant | relevance | groundedness | citation_validity | hallucination_rate | cited_target_species | p50_latency_ms | avg_cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `v1` | **5.00** | **5.00** | 1.77 | **0.00** | **0.80** | 5992 | 0.000295 |
| `v2` | 4.60 | 4.60 | 1.97 | 0.10 | 0.767 | 5933 | 0.000335 |
| `v3` | 4.73 | 4.73 | 1.17 | 0.067 | 0.767 | 6348 | 0.000372 |

`v1` ships (`PROMPT_VARIANT=v1`): perfect relevance and groundedness with no hallucinations,
and the cheapest of the three. The elaborate formats cost accuracy — `v2`'s mandated
per-species bullets invite claims the context does not support (0.10 hallucination rate) and
`v3`'s table is the worst-cited of all.

Citation validity was weak across the board (1.2–2.0 of 5) while 80% of answers still cited
the target species, i.e. the citations existed but did not line up with the numbered context.
Three changes address it: every prompt now demands a `[n]` on each claim-bearing line and
forbids the `[1, 2]` form, the context header carries the USDA symbol and the item count so
the model has an unambiguous mapping, and the judge rubric spells out what 1–5 mean (previously
it graded an under-specified instruction). `answers_with_citation` and `invalid_index_rate` are
now reported alongside the judge score as deterministic checks that cannot drift with the judge.

Results in `data/llm_eval.md` / `.json`; the winner is set via `PROMPT_VARIANT`.

## Monitoring

Every request writes a row to `conversations`: question, rewritten query, archetype,
extracted filters, retrieval mode, retrieved doc ids + scores + arms, tools called, answer,
cited symbols, insufficient-data flag, per-stage latency, tokens, USD cost, and judge scores
when scoring is enabled. 👍/👎 + comments land in `feedback`.

Grafana is provisioned automatically (datasource + dashboard, anonymous viewing enabled) at
<http://localhost:3000/d/plantrag-monitoring> with **9 panels**:

1. Feedback over time (👍/👎 per hour)
2. Positive feedback rate
3. Answer volume, insufficient-data rate, avg docs used
4. Latency by stage (p50 rewrite/retrieve/rerank/generate, p95 total)
5. Hourly and cumulative OpenAI spend
6. Live LLM-judge score distribution
7. Query archetype mix
8. Most cited species
9. Unanswered or downvoted questions (the improvement backlog)

## Project structure

```
app/           config, db, embeddings, vectorstore, retrieval, tools, llm, flow, api
ingestion/     usda_client, cards (row→prose), factsheets (PDF), pipeline (dlt)
eval_suite/    generate_ground_truth, eval_retrieval, eval_llm
ui/            streamlit_app.py
sql/schema.sql species, documents, conversations, feedback
grafana/       datasource + dashboard provisioning
tests/         unit tests for normalisation, fusion, filters, chunking
DESIGN.md      data-source benchmarking and design rationale
```

## Configuration

All settings come from `.env` (see `.env.example`); the notable ones:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | required |
| `OPENAI_MODEL` / `OPENAI_JUDGE_MODEL` | `gpt-4o-mini` | generation / judging |
| `RETRIEVAL_MODE` | `hybrid_rerank_dual` | `sparse`, `dense`, `hybrid`, `hybrid_rerank`, `hybrid_rerank_norewrite`, `hybrid_rerank_dual` |
| `FILTER_MODE` | `soft` | `soft` fuses filtered + unfiltered arms, `hard` gates on the filter |
| `USE_REWRITE` | `1` | query rewriting + filter extraction on/off |
| `PROMPT_VARIANT` | `v1` | answer prompt |
| `TOP_K` / `CANDIDATE_K` / `RRF_K` | `5` / `30` / `60` | context size, per-arm candidates, RRF constant |
| `DENSE_MODEL` / `SPARSE_MODEL` / `RERANK_MODEL` | bge-small / bm25 / MiniLM cross-encoder | local ONNX models, no embedding API cost |
| `INGEST_LIMIT` | `0` (all) | ingest a slice for a fast demo |
| `INCLUDE_FACTSHEETS` | `1` | PDF stage on/off |
| `POSTGRES_HOST_PORT` | `5433` | host-side Postgres port |

Embeddings and re-ranking run locally on CPU via fastembed, so only generation, rewriting
and judging cost money.

## Development

```bash
make install          # uv venv + pinned deps
make api / make ui    # run outside docker against the compose services
make lint test        # ruff check + format check, pytest
make ground-truth eval-retrieval eval-llm
```

## Design notes

**Why hybrid rather than dense-only** — botanical queries are two languages at once. Latin
binomials, cultivar epithets and USDA symbols (`ABBA`, `Abies balsamea (L.) Mill.`) are
lexical tokens that dense embeddings smear together (every *Abies* looks alike in vector
space) but BM25 matches exactly; "purple coneflower" or "something for a boggy shady corner"
is the reverse. Fusion also degrades gracefully: if the embedding model changes or fails,
the lexical arm still answers. Rare tokens — an obscure genus, a specific pest — are
sparse-retrieval's home turf.

**Why filters on top of both** — "under 6 ft", "pH below 6.5", "hardy to −30 °F" are numeric
predicates. No amount of embedding quality enforces them, so the rewriter extracts them into
Qdrant payload conditions and, for the hardest cases, into exact SQL via the
`filter_species` tool. Over-eager extraction is the classic failure mode, so an empty
filtered result transparently retries unfiltered and flags it in the response
(`filters_relaxed`) rather than claiming nothing exists.

**Why prose cards instead of embedding the rows** — a `{"Shade Tolerance": "Tolerant"}`
triple has almost no retrievable surface. Rendered as *"tolerates full shade; prefers moist,
acidic soils of pH 4.0–6.5"*, it matches how people actually ask, while the machine-readable
value stays in the payload for filtering.

**Why the corpus is 2,186 species, not 400k** — retrieval quality is bounded by document
substance, not row count. Every species here carries dozens of verified traits and often a
multi-page USDA fact sheet.
