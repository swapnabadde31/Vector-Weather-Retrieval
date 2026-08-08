# Weather RAG on Lakebase

Unstructured weather text -> chunked, embedded, and made semantically
searchable through Postgres/`pgvector` on Lakebase, following the same
architecture as the day-2 Massive/news pipeline: a raw document store, a
separate chunk/vector store, a Flask retrieval endpoint, and a notebook that
does the heavy lifting on a schedule.

```
POST /weather/search {"query": "flash flood risk this weekend"}
```

returns the most semantically relevant passages, ranked by cosine similarity,
each one traceable back to a real NWS product with its office, issuance
time, and severity.

## 1. Data source: api.weather.gov, not CPC

**Chose `api.weather.gov`.** Two products from it get embedded:

- **Text products** - Area Forecast Discussions (AFD), Hazardous Weather
  Outlooks (HWO), and Hydrologic Outlooks (ESF). These are the forecaster's
  own prose: reasoning about *why* the forecast looks the way it does, not
  just numbers. An AFD routinely contains a sentence like "flash flooding is
  a significant concern, particularly across the urban corridor" - exactly
  the kind of thing a semantic query about flood risk should retrieve.
- **Active alerts** (`/alerts/active`) - watches, warnings, and advisories.
  Their `description` and `instruction` fields are the most directly
  actionable text in the whole corpus and the fastest-moving, which is what
  makes the change-detection logic in `weather_store.py` worth having.

**Considered and rejected `cpc.ncep.noaa.gov` as the primary source.** CPC's
6-10 day, 8-14 day, and monthly outlook discussions are genuinely useful
unstructured text, and cover a forecast horizon NWS text products don't
reach. But CPC publishes them as plain-text/HTML files on a web server, not
through an API:

| | api.weather.gov | cpc.ncep.noaa.gov |
|---|---|---|
| Format | JSON-LD / GeoJSON | Plain text wrapped in HTML |
| Stable product ID | Yes (`/products/{id}`) | No - no ID in the payload |
| Issuance timestamp in payload | Yes (`issuanceTime`) | No |
| Idempotent upsert | Natural (`ON CONFLICT (id)`) | Requires synthesizing an ID |
| Change detection | Compare `issuanceTime`, or the text hash | Text hash only - the only signal available |
| Auth | None (User-Agent header) | None |
| Coverage | Hours to 7 days | 6-10 days out to a season |

An idempotent sync pipeline wants a stable upstream key and a real issuance
time; CPC has neither, which is why `weather_client.fetch_cpc_outlooks()`
exists as an **optional, off-by-default** secondary source (`include_cpc:
true` on `/weather/sync`) rather than the primary one. Where it's used, its
document `id` is synthesized as `cpc:{product}:{sha256[:16]}` and change
detection falls back to hashing the body - which is exactly the workaround
you'd expect to need for a source with no native identifiers, made visible in
the code rather than hidden behind it.

**No API key for either source.** NWS asks only for a descriptive
`User-Agent` header (`weather_client.NWS_USER_AGENT`); CPC asks for nothing.
That's a real simplification versus the day-2 Massive pipeline: no
`massive/api-key` secret to create, rotate, or exclude from git - the only
secret this app needs is the Lakebase connection URL.

## 2. Schema decisions

Two tables, mirroring the day-2 split between raw documents and vectors -
but collapsed from three tables (news / embeddings / chunk\_embeddings) into
two, because weather text products are short enough (a few KB) that
title+description and full-body embeddings would be near-duplicates of each
other. There is exactly one embedding unit: the chunk.

### `weather_documents` - the raw text store, one row per product/alert

| Column | Purpose |
|---|---|
| `id` | Upstream product/alert id (primary key) - already globally unique |
| `source` | `nws_product` \| `nws_alert` \| `cpc_outlook` |
| `product_code` | `AFD` \| `HWO` \| `ESF` \| `ALERT` \| `CPC_610` ... |
| `office_id` | Issuing WFO (`OKX`, `FWD`, ...); `NULL` for alerts, which identify their office by name, not code |
| `headline`, `area_desc`, `severity`, `certainty`, `urgency` | Filterable/displayable summary fields, mostly alert-specific |
| `issued_at`, `effective_at`, `expires_at` | First-class timestamp columns, not buried in JSON - weather text has a short shelf life, and `since_hours` filtering needs them indexed |
| `raw_text` | The unstructured body, kept verbatim so re-chunking with different parameters never requires re-fetching from the API |
| `text_sha256` | Change detector - the upsert skips writing (and the embedder skips re-embedding) when this hasn't moved |
| `payload` | Full upstream JSON, for replay/debugging |

### `weather_embeddings` - one row per chunk, N per document

| Column | Purpose |
|---|---|
| `id` | `"{document_id}::{chunk_index}"` - deterministic, so re-running ingestion doesn't create duplicate rows |
| `document_id` | `REFERENCES weather_documents(id) ON DELETE CASCADE` |
| `chunk_index`, `chunk_text` | Position and the actual passage text - returned directly to the caller, no second lookup |
| `embedding` | `VECTOR(384)` |
| `model_name` | Vectors from different models aren't comparable; search filters on this |

**Why `ON DELETE CASCADE` and delete-then-insert on re-embed, not upsert-by-index:**
a reissued product's text changes length, so a shorter reissue can leave
fewer chunks than the version it replaced. Upserting by `chunk_index` would
leave the surplus chunks behind as orphaned passages that still match
queries - a stale forecast surfacing as if it were current, which is worse
than useless for a use case built around "what's the risk *right now*."
`replace_document_embeddings()` deletes all of a document's chunks before
reinserting, making that impossible.

### Chunking parameters

- **Chunk size 900 characters / overlap 150** (~200 wordpiece tokens - well
  under all-MiniLM-L6-v2's 256-token ceiling, leaving margin so nothing gets
  silently truncated).
- **Section-aware, not just a sliding window.** `embeddings.chunk_weather_text`
  first normalizes NWS teletype line wrapping (products are hard-wrapped at
  ~69 columns with newlines falling mid-sentence - joining these back into
  paragraphs measurably helps a sentence-transformer), then splits on the
  product's *own* section markers (`.SHORT TERM...`, `.AVIATION...`, `&&`)
  before falling back to a character window inside any section still too
  long. An AFD's sections are already topic-coherent; splitting on them means
  a retrieved chunk is about one thing, not an arbitrary slice.
- Undersized sections (`.AVIATION...VFR.` two-liners) are merged into their
  neighbour rather than emitted as their own low-information vector.

### Embedding model

`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions - same choice as the
day-2 reference, for the same reason: small enough to load and run on CPU in
a web request or a serverless notebook with no GPU, fast enough that
`/weather/search` returns in well under a second including the query
encode. `embeddings.MODEL_DIMENSIONS` documents the swap path to
`all-mpnet-base-v2` (768) or `BAAI/bge-large-en-v1.5` (1024) if retrieval
quality matters more than latency; changing it is a migration (`ALTER TABLE
... TYPE VECTOR(n)`, then re-run with `reembed_all=true`), not a config flag,
because the column width is fixed at DDL time.

### Index: HNSW, not IVFFlat

`CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`. IVFFlat needs a
training pass over representative data and degrades when the table is small
or grows in bursts - which is exactly how a weather corpus behaves (it
refills every few hours, and starts from zero on a fresh deployment). HNSW
builds incrementally and needs no training step, at the cost of slightly
slower inserts - an acceptable trade for a corpus this size.

## 3. Running the pipeline end to end

Three ways to run it, in increasing order of "how you'd actually operate
this":

### A. One Flask app, driven by curl - fastest way to see it work

```bash
export LAKEBASE_URL=postgresql://role:pw@host:5432/databricks_postgres?sslmode=require
pip install -r requirements.txt
python app.py

# harvest text + alerts, then vectorize inline (fine for a small demo sync)
curl -X POST localhost:8000/weather/sync \
  -H 'Content-Type: application/json' \
  -d '{"offices": ["OKX","FWD"], "limit_per_pair": 1, "embed": true}'

# ask a question
curl -X POST localhost:8000/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "flash flood risk this weekend"}'
```

Or open `http://localhost:8000/` for the search UI, which has "Sync from
NWS" / "Embed pending" buttons and example queries.

### B. The CLI - same modules, no server

```bash
python scripts/run_pipeline.py sync --offices OKX,FWD,SEW --limit-per-pair 2
python scripts/run_pipeline.py embed
python scripts/run_pipeline.py search "damaging wind and large hail"
python scripts/run_pipeline.py stats

# or all three in one shot
python scripts/run_pipeline.py all --query "heavy mountain snow"
```

### C. The Databricks notebook - the real operating mode

`notebooks/ingest_weather_embeddings.py` is what you'd actually schedule: it
harvests, upserts, chunks, embeds, writes vectors, and runs a retrieval smoke
test in one pass, on Databricks serverless compute (works on Free Edition -
no cluster required). Import it into a Workspace Git folder and either:

- **Run it manually** once to seed the corpus, or
- **Schedule it** as a Workflow every 6 hours (a job spec is in
  `resources/ingest_weather_embeddings_job.yml`, deployable via
  `databricks bundle deploy -t dev`, or recreate the same task by hand in the
  Workflows UI - see the main `README.md` for the click-by-click version).

All three paths call the exact same `weather_client`, `embeddings`, and
`weather_store` modules, so a query embedded by the app and a chunk embedded
by the notebook land in the same vector space - the one thing that has to be
true for cosine similarity to mean anything.

Full setup (Lakebase instance, secret, deployment) is in the main
`README.md`.

## 4. Known limitations and what I'd improve with more time

- **No `article_url` fetch/chunk pass.** The day-2 news pipeline had a second
  tier - fetch each article's full body via `trafilatura` and chunk that
  separately from the title/description. NWS text products don't have an
  equivalent "click through for the full text" URL - the product *is* the
  full text - so there was nothing to add here. If CPC becomes a first-class
  source, its outlook discussions are similarly self-contained, so this gap
  is more "not applicable" than "not built."
- **Alert `office_id` is always `NULL`.** Alerts identify their issuing
  office by free-text name ("NWS Norman OK"), not the 3-letter code text
  products use, and there's no reliable API-side mapping between the two.
  Office-filtered alert search doesn't work as a result. A static
  name-to-WFO lookup table would fix this but felt like guessing at data the
  API doesn't actually provide.
- **In-request embedding (`/weather/embed`, `POST /weather/sync {"embed":
  true}`) is capped at `MAX_INLINE_EMBED_DOCS` (200) and will feel slow past
  a few dozen documents** - a model load plus CPU inference inside an HTTP
  request handler is convenient for demos, not a production path. The
  notebook is the intended production path; the inline option exists so the
  whole pipeline is exercisable through the API alone, which is what makes
  it runnable start-to-finish on Free Edition without ever touching Jobs.
- **No reranking or MMR diversification.** Results are pure cosine-similarity
  order. For a corpus this size it doesn't matter, but at real scale (dozens
  of offices, weeks of history) a query like "severe weather" would likely
  return five near-duplicate chunks from the same AFD before a second
  document's perspective shows up - `group_by_document=true` (the default)
  mitigates this by capping one chunk per document, but doesn't diversify
  beyond that.
- **CPC ingestion has no real change detection**, only a body-hash fallback,
  because CPC gives no ID and no timestamp (see the source-comparison table
  above). It's honestly the reason CPC stayed a secondary, opt-in source
  rather than becoming the primary one.
- **No retention/expiry job.** Expired alerts (`expires_at < now()`) and
  superseded AFDs stay in the corpus indefinitely; nothing currently
  auto-archives or removes documents. A cheap improvement: a scheduled
  `DELETE FROM weather_documents WHERE expires_at < now() - interval '7
  days'` (cascading to vectors), so a "this weekend" query five weeks from
  now doesn't surface a long-expired watch. Also relevant for alerts, whose
  `certainty`/`urgency` fields shift meaning post-expiry in ways the raw
  cosine score doesn't reflect.
- **Single embedding model, no A/B path.** `weather_store.search()` filters
  by `model_name`, so the schema supports two models coexisting during a
  migration, but there's no tooling to actually run and compare them
  side-by-side - you'd do it by hand with two `run_pipeline.py search --model
  ...` calls today.
- **Two Postgres drivers in one project, and they must be kept in sync by
  hand.** `weather_store.py` (used by the app and the CLI) is `psycopg2`;
  the notebook is `pg8000`, because `psycopg2` crashes Databricks Serverless
  notebook compute on import. The notebook reimplements the DDL/upsert/search
  SQL inline rather than importing `weather_store.py`, so a schema or query
  change made in one place doesn't automatically apply to the other - a real
  maintenance cost of working around the sandbox constraint. A cleaner fix
  would be a single driver-agnostic SQL layer (e.g. SQLAlchemy Core with the
  `pg8000` dialect everywhere, including the app), which would remove the
  duplication entirely; I kept `psycopg2` for the app because it's the more
  common/battle-tested choice for a long-running service and the constraint
  is specific to sandboxed notebook compute, not to Postgres apps generally.

