-- weather_embeddings: one row per chunk, N rows per document.
--
-- This is the unit of *retrieval*, kept separate from weather_documents so
-- that re-embedding with different chunk sizes or a different model rewrites
-- only this table and never re-fetches from the NWS API.
--
-- VECTOR(384) matches sentence-transformers/all-MiniLM-L6-v2. If you switch
-- models, change the width here to match embeddings.MODEL_DIMENSIONS:
--     all-MiniLM-L6-v2 / all-MiniLM-L12-v2     -> 384
--     all-mpnet-base-v2 / BAAI/bge-base-en-v1.5 -> 768
--     BAAI/bge-large-en-v1.5                    -> 1024
-- The column width is fixed at DDL time, so a model change is a migration:
-- ALTER the column, then re-run the notebook with reembed_all=true.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_embeddings (
    -- Deterministic: "<document_id>::<chunk_index>". Makes a re-run of the
    -- same document produce the same keys instead of duplicate rows.
    id            TEXT PRIMARY KEY,

    -- ON DELETE CASCADE: dropping a document takes its vectors with it, so
    -- an expired alert can never leave orphaned passages that still match.
    document_id   TEXT NOT NULL REFERENCES weather_documents(id) ON DELETE CASCADE,

    chunk_index   INTEGER NOT NULL,

    -- The passage itself, stored alongside its vector. Search returns the
    -- text directly from this table - no second lookup, and the caller gets
    -- exactly the span that matched rather than a whole product.
    chunk_text    TEXT NOT NULL,
    char_count    INTEGER NOT NULL DEFAULT 0,

    -- Denormalized from the parent so filtered searches can be answered
    -- without widening the join.
    product_code  TEXT,
    office_id     TEXT,
    issued_at     TIMESTAMPTZ,

    embedding     VECTOR(384) NOT NULL,

    -- Which model produced this vector. Vectors from different models are not
    -- comparable, so search filters on it and the ingest job uses it to find
    -- rows that need regenerating after a model swap.
    model_name    TEXT NOT NULL,
    embedded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document ON weather_embeddings (document_id);
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_model    ON weather_embeddings (model_name);

-- HNSW with cosine ops, matching the `<=>` operator used by weather_store.search.
-- HNSW rather than IVFFlat: IVFFlat needs a training pass over representative
-- data and degrades when the table is small or grows in bursts, which is
-- exactly how a weather corpus behaves. HNSW builds incrementally.
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_vector
ON weather_embeddings USING hnsw (embedding vector_cosine_ops);

-- Verify
SELECT column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_name = 'weather_embeddings'
ORDER BY ordinal_position;
