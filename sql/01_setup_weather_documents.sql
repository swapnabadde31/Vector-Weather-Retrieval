-- weather_documents: the raw, unstructured weather text store.
--
-- One row per NWS text product (AFD/HWO/ESF) or active alert. This is the
-- document layer: the unit of *sync*, keyed on the upstream product id so a
-- repeated harvest is idempotent.
--
-- Applying this by hand is optional - weather_store.ensure_schema() runs the
-- identical statements on app start and at the top of the notebook. These
-- files exist so the schema is reviewable as SQL and can be applied through
-- the Lakebase SQL editor before the first run.

CREATE TABLE IF NOT EXISTS weather_documents (
    -- Upstream identifier. NWS product URLs and alert ids are already stable
    -- and globally unique, so there is no reason to mint a surrogate key.
    id              TEXT PRIMARY KEY,

    -- Provenance: nws_product | nws_alert | cpc_outlook. Lets a single corpus
    -- hold several sources and still be filterable at query time.
    source          TEXT NOT NULL,

    -- AFD | HWO | ESF | ALERT | CPC_610 ...
    product_code    TEXT NOT NULL,
    product_name    TEXT,

    -- Issuing Weather Forecast Office (OKX, FWD, ...). NULL for alerts, which
    -- identify their office by name rather than by 3-letter id.
    office_id       TEXT,
    wmo_id          TEXT,

    -- Summary fields. Displayed with results so a hit is readable without
    -- fetching the whole product.
    headline        TEXT,
    area_desc       TEXT,

    -- Alert-only classification (Severe/Extreme, Likely/Observed, Immediate).
    severity        TEXT,
    certainty       TEXT,
    urgency         TEXT,

    -- Weather text has a short shelf life, so times are first-class columns
    -- rather than JSON fields: `since_hours` filtering and freshness checks
    -- both need them indexed.
    issued_at       TIMESTAMPTZ,
    effective_at    TIMESTAMPTZ,
    expires_at      TIMESTAMPTZ,

    -- The unstructured body. Kept verbatim: chunking parameters change, and
    -- re-chunking from the stored original avoids re-fetching from the API.
    raw_text        TEXT NOT NULL,

    -- Change detector. Products are reissued on a schedule but frequently
    -- carry near-identical text; comparing the digest is what stops every
    -- sync from re-embedding the entire corpus.
    text_sha256     TEXT NOT NULL,
    char_count      INTEGER NOT NULL DEFAULT 0,

    -- Full upstream JSON, for replaying a parse without re-fetching.
    payload         JSONB,

    synced_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Only moves when raw_text actually changed. The embedding job compares
    -- against this to decide what is stale.
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_weather_documents_office ON weather_documents (office_id);
CREATE INDEX IF NOT EXISTS idx_weather_documents_code   ON weather_documents (product_code);
CREATE INDEX IF NOT EXISTS idx_weather_documents_issued ON weather_documents (issued_at DESC);
CREATE INDEX IF NOT EXISTS idx_weather_documents_source ON weather_documents (source);

-- Verify
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'weather_documents'
ORDER BY ordinal_position;
