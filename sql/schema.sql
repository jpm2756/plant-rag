-- PlantRAG schema: corpus tables + monitoring tables.

CREATE TABLE IF NOT EXISTS species (
    id                  INTEGER PRIMARY KEY,
    symbol              TEXT NOT NULL,
    scientific_name     TEXT NOT NULL,
    scientific_no_author TEXT,
    common_name         TEXT,
    family              TEXT,
    genus               TEXT,
    group_name          TEXT,
    duration            TEXT[],
    growth_habit        TEXT[],
    native_status       JSONB,
    states              TEXT[],
    other_common_names  TEXT[],
    synonyms            TEXT[],
    wetland_status      TEXT,
    factsheet_urls      TEXT[],
    plantguide_urls     TEXT[],
    profile_url         TEXT,
    -- pivoted characteristics used as retrieval filters
    ph_min              REAL,
    ph_max              REAL,
    height_mature_ft    REAL,
    temp_min_f          REAL,
    precip_min_in       REAL,
    precip_max_in       REAL,
    frost_free_days_min REAL,
    root_depth_min_in   REAL,
    drought_tolerance   TEXT,
    shade_tolerance     TEXT,
    salinity_tolerance  TEXT,
    fire_tolerance      TEXT,
    moisture_use        TEXT,
    growth_rate         TEXT,
    growth_form         TEXT,
    lifespan            TEXT,
    bloom_period        TEXT,
    flower_color        TEXT,
    foliage_color       TEXT,
    fruit_color         TEXT,
    toxicity            TEXT,
    nitrogen_fixation   TEXT,
    palatable_human     TEXT,
    fertility_requirement TEXT,
    active_growth_period TEXT,
    leaf_retention      TEXT,
    coarse_soil         TEXT,
    medium_soil         TEXT,
    fine_soil           TEXT,
    n_characteristics   INTEGER DEFAULT 0,
    raw_characteristics JSONB,
    ingested_at         TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS species_symbol_idx ON species (symbol);
CREATE INDEX IF NOT EXISTS species_sciname_idx ON species (lower(scientific_no_author));
CREATE INDEX IF NOT EXISTS species_common_idx ON species (lower(common_name));

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    species_id   INTEGER NOT NULL REFERENCES species (id) ON DELETE CASCADE,
    section      TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL DEFAULT 0,
    title        TEXT NOT NULL,
    text         TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_species_idx ON documents (species_id);
CREATE INDEX IF NOT EXISTS documents_section_idx ON documents (section);

CREATE TABLE IF NOT EXISTS conversations (
    id             UUID PRIMARY KEY,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    question       TEXT NOT NULL,
    rewritten_query TEXT,
    archetype      TEXT,
    filters        JSONB,
    retrieval_mode TEXT,
    prompt_variant TEXT,
    model          TEXT,
    answer         TEXT,
    cited_symbols  TEXT[],
    retrieved      JSONB,
    tools_used     TEXT[],
    n_results      INTEGER,
    insufficient   BOOLEAN DEFAULT FALSE,
    latency_ms     INTEGER,
    latency_rewrite_ms INTEGER,
    latency_retrieve_ms INTEGER,
    latency_rerank_ms  INTEGER,
    latency_generate_ms INTEGER,
    prompt_tokens  INTEGER,
    completion_tokens INTEGER,
    cost_usd       NUMERIC(10, 6),
    judge_relevance     SMALLINT,
    judge_groundedness  SMALLINT,
    judge_hallucinated  BOOLEAN
);

CREATE INDEX IF NOT EXISTS conversations_created_idx ON conversations (created_at);

CREATE TABLE IF NOT EXISTS feedback (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    rating          SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    comment         TEXT
);

CREATE INDEX IF NOT EXISTS feedback_created_idx ON feedback (created_at);
