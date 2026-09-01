# PlantRAG — Design Doc

A hybrid RAG + agent application over the **USDA PLANTS Database API**, built to satisfy the
LLM Zoomcamp project rubric while remaining a genuinely useful plant-selection assistant.

> **v2 (2026-09-01):** source changed from Trefle to the USDA PLANTS Services API.
> Rationale and measurements in §2.

---

## 1. Problem statement

Choosing the right plant is a **multi-constraint filtering problem wrapped in fuzzy natural
language**. A gardener asks:

> "What's a drought-tolerant, shade-tolerant native shrub for acidic soil in Oregon that
> stays under 6 feet and isn't toxic?"

Answering that requires (a) resolving vague language ("drought-tolerant", "shrub") to
concrete botanical attributes, (b) filtering on numeric/categorical constraints (mature
height, pH range, growth habit, state distribution, toxicity), and (c) explaining the
result in prose with sources.

Existing plant databases expose either raw search forms (precise but unusable for a
natural-language question — USDA's own advanced search has ~80 filter fields) or generic
chatbots (fluent but prone to inventing species, hardiness ranges, and toxicity claims —
the last of which is genuinely dangerous).

**PlantRAG** sits in between: a grounded assistant that answers free-form horticultural
questions using only retrieved records from an authoritative botanical dataset, with
citations back to the source species profile, and an explicit refusal when the corpus
doesn't support an answer.

### Target users & questions
| User | Example question |
|---|---|
| Home gardener | "Shade-tolerant shrub for fine-textured soil, non-toxic, under 6 ft?" |
| Restoration / conservation planner | "Native nitrogen-fixing forbs for dry sites in New Mexico" |
| Student / naturalist | "How do *Abies balsamea* and *Abies fraseri* differ in tolerances?" |
| Landscaper | "Fast-growing trees tolerant of pH 7.5+ soil and low precipitation" |
| Wildlife / pollinator gardener | "Which species provide cover and food for terrestrial birds?" |

---

## 2. Data source — measured comparison

### 2.1 Why not Trefle (the original plan)

Trefle's hosted API is up (returns `401` for an invalid token), and a no-auth Parquet dump
exists, but **both are the same database** — the dump is generated from the API, so
switching to the live API does not change data density. Measured on the full dump
(416,473 rows × 54 cols, generated 2020-10-15):

```
edible (boolean)   99.7%     common_name         9.2%
family             98.1%     growth_habit        6.7%
distributions      83.3%     average_height_cm   1.3%
synonyms           43.7%     light / pH / soil  ~0.8-1.5%
```

Only **6,473 rows have ≥5 populated trait fields**; `edible_part` is populated on 0.03%.
So ~98% of Trefle is taxonomic stubs. It is also effectively unmaintained since 2020.
**Rejected on data density.**

### 2.2 Other candidates, checked live (2026-09-01)

| Source | Status | Verdict |
|---|---|---|
| **USDA PLANTS Services API** | **Live, no API key, public OpenAPI spec** | **Selected** |
| OpenFarm (`openfarm.cc/api/v1`) | Dead — domain now 301-redirects to GitHub | Rejected |
| Perenual | Live but requires a free key (`404` + "Missing/Issue with API Key") | Backup; houseplant-care angle |
| Permapeople | Live but requires key pair (`401`) | Backup |
| GBIF | Live, no key, excellent taxonomy + occurrences | No horticultural traits; possible enrichment |
| Trefle | Live, sparse (§2.1) | Rejected |

### 2.3 Selected source: USDA PLANTS Services API

Base: `https://plantsservices.sc.egov.usda.gov/api/`
Spec: `https://plantsservices.sc.egov.usda.gov/swagger/v1/swagger.json`
**No authentication. No rate-limit headers observed.** Public-domain US government data.

Relevant endpoints (from the spec):

| Endpoint | Use |
|---|---|
| `GET /characteristicSearchResults` | **Corpus enumeration** — every species that has characteristics data |
| `GET /PlantCharacteristics/{id}` | ~80 attribute name/value/category triples per species |
| `GET /PlantProfile?symbol=` \| `/PlantProfile/{id}` | Taxonomy, duration, growth habit, native status by region, distribution, synonyms, other common names, wildlife, wetland, legal/noxious/invasive status, fact-sheet & plant-guide PDF links, images |
| `GET /PlantWildlife/{id}`, `/PlantWetland/{id}` | Wildlife food/cover values; wetland indicator status |
| `GET /PlantEthnobotany/{id}`, `/PlantPollinator/{id}` | Traditional uses; pollinator data |
| `GET /PlantSearch?searchText=`, `POST /plants-search-results` | Name lookup across the full ~90k-name taxonomy (exact-lookup tool) |
| `GET /GrowthHabitSearch`, `/DurationSearch`, `/StateSearch`, `/NoxiousInvasiveSearch`, `/RaritySearch` | Controlled vocabularies for filter extraction + agent tools |

### 2.4 Measured density (this is the whole point)

`characteristicSearchResults` → **2,186 species**, 178 families, **99.9% with a common name**.

Sampled 40 random species and pulled `PlantCharacteristics`: **56-553 characteristic rows
per species, median 80.** Fill rate across the sample (n=40) for the fields that matter:

```
100%  Toxicity, Drought Tolerance, Shade Tolerance, Anaerobic Tolerance,
      Salinity Tolerance, CaCO3 Tolerance, Fire Tolerance/Resistance,
      Height Mature (ft), Root Depth Min (in), Temperature Min (°F),
      pH Min / pH Max, Precipitation Min / Max, Frost-Free Days Min,
      Moisture Use, Fertility Requirement, Growth Rate, Growth Form,
      Active Growth Period, Lifespan, Bloom Period, Flower Color,
      Foliage Color / Texture / Porosity, Fruit-Seed Color / Period /
      Abundance / Persistence, Nitrogen Fixation, Allelopathy,
      Resprout Ability, Coppice Potential, Hedge Tolerance,
      Seed Spread Rate, Vegetative Spread Rate, Seedling Vigor,
      Propagation (seed / cuttings / bare root / container / bulb /
      corm / sod / sprigs / tubers), Palatable Human / Browse / Graze,
      Protein Potential, Commercial Availability, Product suitability
      (lumber, pulpwood, veneer, post, fodder, Christmas tree, berry/nut/seed)
 68%  Wetland indicator status
100%  PlantWildlife endpoint reachable (food/cover values, often sparse per species)
```

**Contrast with Trefle: ~80 populated fields per species vs ~1-5.** The corpus is ~40×
smaller (2,186 vs 416k rows) but every document is substantive — which is exactly the right
trade for RAG, where document quality dominates and 2k dense documents make retrieval
differences measurable and re-indexing near-instant.

Bonus: profiles expose **fact-sheet / plant-guide PDFs** (e.g.
`/DocumentLibrary/factsheet/pdf/fs_abba.pdf`) — genuine long-form prose per species. That
gives the project a real unstructured corpus alongside the structured traits, i.e. classic
PDF-style RAG *and* structured filtering in one app.

### 2.5 Known limitations (state them in the README)

* **US-centric** — distribution and native status are US/Canada; not a global flora. Scoped
  as a US-focused assistant, which is honest and keeps evaluation coherent.
* **2,186 species with full characteristics** out of ~90k names in PLANTS. Names outside the
  corpus are still resolvable via the `PlantSearch` tool, and the app says when a species
  has no characteristics data rather than guessing.
* Cultivar/synonym rows can duplicate a characteristic name for one species — dedupe on
  `(name, cultivar, synonym)` during ingestion.
* No documented rate limit ≠ no rate limit: ingestion is polite (concurrency ≤4, backoff)
  and cached, so re-runs don't re-hit the API.

---

## 3. Why hybrid RAG (and not pure text-to-SQL, nor pure vectors)

The original instinct — "you wouldn't embed a whole plant database" — still holds, and the
design honors it: embed **2,186 dense species documents plus their fact-sheet prose**, not
416k stubs. But the retrieval method is dictated by the data:

1. **Binomials and USDA symbols are lexical.** `Abies balsamea` / `Abies fraseri` /
   `Abies concolor`, and symbols like `ABBA`, `ECPU`, are near-identical (or meaningless) in
   embedding space. BM25 discriminates them exactly; dense retrieval confuses them.
2. **Numbers and ordinal tolerances don't embed.** `pH 4.0-6.0`, `Height, Mature = 60 ft`,
   `Precipitation Min = 20 in`, `Drought Tolerance = High` — cosine similarity is blind to
   ranges, so these must be structured predicates (metadata filters), not similarity.
3. **Vague, paraphrased intent needs dense.** "something for a soggy shady corner",
   "good for erosion control on a slope", "low-maintenance native for pollinators" — no
   lexical overlap with the source vocabulary, which is where embeddings win.
4. **Fact-sheet PDFs are prose** — pure classic RAG territory.

So: sparse for names/symbols/rare tokens, dense for paraphrase and prose, metadata filters
for numbers and booleans, fused and re-ranked. Operational upsides: per-arm failure
attribution when debugging, graceful degradation if the embedding service is down, and no
re-embedding when only the lexical index changes.

Honest cost: a second index, a fusion weight to tune, more compose services. Justified here
because the name-vs-description duality is intrinsic to botanical queries.

---

## 4. Architecture

```
    USDA PLANTS API (no key)
    ├── /characteristicSearchResults   (2,186 species)     ┌────────────────────────────┐
    ├── /PlantCharacteristics/{id}     (~80 traits each)   │  dlt ingestion pipeline    │
    ├── /PlantProfile/{id}             (taxonomy, dist,  ──▶  extract → normalize →     │
    │                                   wildlife, PDFs)    │  doc-ify → embed → load   │
    └── /DocumentLibrary/.../fs_*.pdf  (prose fact sheets) └───────┬─────────────┬──────┘
                                                                   │             │
                                                     ┌─────────────▼──┐   ┌──────▼─────────────┐
                                                     │ Postgres        │   │ Qdrant            │
                                                     │ • species       │   │ • ~10k chunks:    │
                                                     │ • characteristics│  │   trait sections  │
                                                     │ • feedback      │   │   + PDF prose,    │
                                                     │ • traces        │   │   dense + sparse  │
                                                     └─────────────┬──┘   └──────┬────────────┘
                                                                   │              │
                                                           ┌───────▼──────────────▼───────┐
                                                           │  RAG / agent flow (FastAPI)  │
                                                           │  1. query rewrite + filter   │
                                                           │     extraction (LLM → JSON)  │
                                                           │  2. hybrid retrieve (RRF)    │
                                                           │  3. cross-encoder re-rank    │
                                                           │  4. tools:                   │
                                                           │     • species_lookup(name)   │
                                                           │     • trait_filter(filters)  │
                                                           │     • usda_search(text) live │
                                                           │  5. prompt + answer + cites  │
                                                           └───────┬──────────────┬───────┘
                                                                   │              │
                                                         ┌─────────▼───┐   ┌──────▼──────┐
                                                         │  Streamlit  │   │  Grafana    │
                                                         │  chat + 👍/👎│   │  6 charts   │
                                                         └─────────────┘   └─────────────┘
```

One `docker-compose.yml`: `postgres`, `qdrant`, `api`, `ui`, `grafana`, plus a one-shot
`ingest` job.

### 4.1 Document design

Raw `{name, value, category}` triples retrieve badly; prose retrieves well. Each species is
rendered by a deterministic template into **one summary card plus ~4 section chunks**
(Growth & form · Site tolerances & climate · Propagation & establishment · Uses, wildlife &
safety), with raw values preserved in chunk metadata for filtering:

```
Balsam fir (Abies balsamea (L.) Mill.) — USDA symbol ABBA, family Pinaceae.
A perennial tree, native to the lower 48 states and Canada.
Site tolerances: high shade tolerance, low drought tolerance, no salinity
tolerance, fire resistant. Soil pH 4.0-6.0; adapted to coarse and medium
textured soils. Minimum temperature -43 °F; needs 80+ frost-free days;
precipitation 20-60 in. Mature height 60 ft, minimum root depth 20 in.
Toxicity: none. Propagated by seed and bare root; cold stratification
required. Moderate growth rate, moderate lifespan.
```

Chunk metadata: `species_id, symbol, scientific_name, common_name, family, growth_habit,
duration, native_status[], states[], shade_tol, drought_tol, ph_min, ph_max, height_ft,
temp_min_f, precip_min, precip_max, toxicity, nitrogen_fixation, bloom_period, section`.

Fact-sheet/plant-guide PDFs are fetched, text-extracted, chunked (~500 tokens, 100 overlap)
and indexed with `section="factsheet"` — the unstructured half of the corpus.

### 4.2 Retrieval flow

1. **Query rewriting + filter extraction** (best-practice point, and load-bearing): one LLM
   call → `{rewritten_query, filters}`. *"low native shrub for acidic soil in Oregon, tough
   in dry shade"* → `{"query": "shrub acidic soil dry shade", "filters": {"growth_habit":
   "Shrub", "ph_max": {"lte": 6.5}, "height_ft": {"lte": 6}, "states": ["OR"],
   "drought_tol": ["Medium","High"], "shade_tol": ["Medium","High"], "native": true}}`.
   Vocabularies come from the USDA controlled-vocabulary endpoints, so extracted values are
   always legal filter values. Also expands common → scientific names and resolves pronouns
   ("is *it* toxic to dogs?").
2. **Hybrid retrieval**: Qdrant sparse+dense with extracted filters as a hard pre-filter;
   **Reciprocal Rank Fusion** (`k=60`).
3. **Re-ranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` over top ~30 → top 5-8.
4. **Tools** (the agent half, via function calling): `species_lookup(name|symbol)` for exact
   single-species questions, `trait_filter(filters)` for "how many / list all" structured
   queries over Postgres, `usda_search(text)` for live lookups of names outside the corpus.
   Retrieval-only is the default path; tools fire when the query is exact/aggregate.
5. **Answer synthesis** with per-claim citations `[SYMBOL]` linking to the USDA profile
   URL, plus an explicit "not in the corpus" branch.

### 4.3 Guardrails

Toxicity, edibility, and palatability answers must cite the source characteristic and carry
a standing disclaimer — never present an LLM inference as a safety claim. Missing values are
reported as missing, not inferred.

---

## 5. Evaluation

### 5.1 Ground truth
LLM-generate 5 questions per document over a stratified sample of ~400 species (~2,000
pairs), storing `(question, ground_truth_species_id, archetype)`. Archetypes: exact-name,
common-name-only, symbol, trait-constraint, numeric-constraint, geographic, wildlife/use,
and vague/descriptive — because the hybrid argument predicts *different winners per
archetype*, and that breakdown is the interesting result, not just one aggregate number.

### 5.2 Retrieval evaluation (≥3 approaches → best one ships)
Metrics: **hit-rate@5** and **MRR@5**, overall and per archetype.

| # | Approach |
|---|---|
| 1 | Sparse / BM25 only |
| 2 | Dense only (`all-MiniLM-L6-v2`; `bge-small-en-v1.5` as second encoder) |
| 3 | Hybrid RRF (sparse + dense) |
| 4 | Hybrid RRF + cross-encoder re-rank |
| 5 | Hybrid + LLM query rewriting (ablation: rewrite on/off, filters on/off) |

Also swept: chunking strategy (one card per species vs sectioned chunks vs sections+PDF),
and RRF `k`. Results table + charts in `notebooks/02_eval_retrieval.ipynb`; winner wired
into the app by config.

### 5.3 LLM evaluation (≥2 approaches → best one ships)
Three prompt variants — (a) terse grounded answer, (b) structured answer with a per-plant
attribute table, (c) reason-then-answer with a mandatory "insufficient data" branch —
scored two ways:
* **LLM-as-judge** on relevance / groundedness / hallucination over ~150 held-out questions
  (RELEVANT / PARTLY_RELEVANT / NON_RELEVANT plus a citation-validity check).
* **Cosine similarity** to reference answers.

Report the comparison table with cost and latency per variant; commit judge prompt and raw
scores so the choice is auditable.

---

## 6. Interface

**Streamlit** chat UI backed by a **FastAPI** service (`/ask`, `/feedback`, `/healthz`) — a
UI *and* an API. Features: streaming answer; expandable "retrieved species" panel showing
each chunk with its score and **which retrieval arm found it** (makes hybrid legible); the
extracted filter JSON; latency and token cost; 👍/👎 with optional comment; deep links to
USDA profiles and fact sheets.

---

## 7. Ingestion pipeline

**dlt** with four stages, exposed as Makefile targets and a compose one-shot:

1. `enumerate` — `GET /characteristicSearchResults` → 2,186 species stubs into Postgres.
2. `hydrate` — per species: `PlantCharacteristics`, `PlantProfile`, `PlantWildlife`,
   `PlantWetland` (concurrency ≤4, retry/backoff, raw JSON cached to disk so re-runs are
   offline and reproducible).
3. `docify` — pivot triples → typed columns, dedupe cultivar/synonym rows, render summary
   card + section chunks.
4. `index` — dense + sparse embeddings upserted into Qdrant with metadata payloads;
   incremental by `(species_id, section, content_hash)`.

Stage 5 (optional): fetch fact-sheet/plant-guide PDFs, extract text, chunk, index.

A **pinned snapshot** of the hydrated raw JSON is committed (or released as a tarball) so
the project remains reproducible even if the USDA service is down at grading time.

---

## 8. Monitoring

Postgres `conversations` (question, rewritten query, extracted filters, retrieved ids,
retrieval arm scores, answer, model, prompt variant, tokens, cost, latency per stage, judge
score) and `feedback` (👍/👎, comment). **Grafana**, provisioned as code, 6 panels:

1. 👍/👎 over time + rolling positive rate
2. p50/p95 latency split by stage (rewrite / retrieve / rerank / generate)
3. Token cost per query + cumulative daily spend
4. LLM-as-judge relevance distribution on live traffic
5. Query archetype mix + zero-result / "insufficient data" rate
6. Most-requested families & species + top unanswered queries

---

## 9. Rubric coverage

| Criterion | Max | Plan | Expected |
|---|---|---|---|
| Problem description | 2 | §1 + README | 2 |
| Retrieval flow | 2 | Qdrant KB + LLM | 2 |
| Retrieval evaluation | 2 | 5 approaches, hit-rate/MRR, per-archetype, best ships | 2 |
| LLM evaluation | 2 | 3 prompts, judge + cosine | 2 |
| Interface | 2 | Streamlit + FastAPI | 2 |
| Ingestion pipeline | 2 | dlt, 4-5 stages | 2 |
| Monitoring | 2 | feedback + 6-chart Grafana | 2 |
| Containerization | 2 | everything in docker-compose | 2 |
| Reproducibility | 2 | pinned deps, keyless API + committed snapshot, `make up` | 2 |
| Hybrid search | 1 | evaluated + shipped | 1 |
| Re-ranking | 1 | cross-encoder | 1 |
| Query rewriting | 1 | rewrite + structured filter extraction | 1 |
| Cloud deployment | 2 | bonus, §11 | 0-2 |
| **Total** | **23** | | **21-23** |

Extra-bonus candidates: the structured filter-extraction agent over USDA controlled
vocabularies; per-archetype retrieval analysis; combining structured traits with PDF prose
in one index; CI running the eval suite.

---

## 10. Repo layout

```
plant-rag/
├── docker-compose.yml
├── Makefile                  # up / ingest / eval / test / lint
├── pyproject.toml + uv.lock  # pinned versions
├── .env.example
├── ingestion/
│   ├── pipeline.py           # dlt: enumerate → hydrate → docify → index
│   ├── usda_client.py        # typed client + caching + backoff
│   ├── cards.py              # traits → prose card / section chunks
│   └── factsheets.py         # PDF fetch + text extraction (optional stage)
├── app/
│   ├── api.py                # FastAPI
│   ├── rewrite.py            # query rewriting + filter extraction
│   ├── retrieve.py           # hybrid + RRF + rerank
│   ├── tools.py              # species_lookup, trait_filter, usda_search
│   ├── prompts/              # v1 / v2 / v3
│   └── db.py                 # conversations, feedback
├── ui/streamlit_app.py
├── eval/
│   ├── generate_ground_truth.py
│   ├── eval_retrieval.py
│   └── eval_llm.py
├── notebooks/                # 01_explore, 02_eval_retrieval, 03_eval_llm
├── grafana/                  # provisioning + dashboard JSON
├── tests/
└── README.md                 # problem, architecture, results, how to run
```

---

## 11. Open decisions

1. **LLM provider** — OpenAI (`gpt-4o-mini`, needs a key) vs Ollama in compose (free, fully
   reproducible, weaker judge). Plan: OpenAI-compatible client, provider swappable by env
   var.
2. **Vector store** — Qdrant (native sparse+dense hybrid, clean payload filters) vs
   Elasticsearch (course-familiar). Recommendation: Qdrant.
3. **Fact-sheet PDF stage** — include (real prose corpus, stronger RAG story, more moving
   parts) or defer to a stretch goal.
4. **Cloud deployment** (+2) — single VM running the same compose file, or Streamlit
   Community Cloud UI + managed Qdrant free tier.
5. **Scope framing** — all 2,186 characterized species, or a themed slice (e.g. natives for
   restoration / pollinator planting) for a sharper demo narrative.

## 12. Build order

1. Repo skeleton + compose (Postgres, Qdrant) + pinned deps
2. USDA client + dlt `enumerate`/`hydrate` with on-disk cache (committed snapshot)
3. `docify` → cards & section chunks in Postgres
4. Embed + Qdrant load; smoke-test hybrid search from a notebook
5. Ground-truth generation → retrieval eval (the core experiment)
6. FastAPI flow with the winning retriever + rewriting + rerank + tools
7. Prompt variants → LLM eval → pick winner
8. Streamlit UI + feedback capture
9. Grafana provisioning + dashboard
10. README with results tables; optional PDF stage; optional cloud deploy
