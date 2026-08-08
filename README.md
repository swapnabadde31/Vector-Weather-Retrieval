# Weather RAG on Lakebase

A Databricks App that:

- Harvests unstructured weather text from **api.weather.gov** (Area Forecast
  Discussions, Hazardous Weather Outlooks, Hydrologic Outlooks, active
  alerts) - no API key required.
- Chunks and vectorizes it with `sentence-transformers`, storing the vectors
  in **Lakebase** (Databricks-managed Postgres) via `pgvector`.
- Exposes a Flask API to sync, embed, and semantically search the corpus:

  ```
  POST /weather/search {"query": "flash flood risk this weekend"}
  ```

See **[README_WEATHER.md](README_WEATHER.md)** for the source comparison
(api.weather.gov vs. CPC), schema rationale, chunking parameters, and known
limitations. This file is the setup and deployment guide.

## Files

- `app.py` - Flask app: `/healthz`, `/weather/stats`, `/weather/documents`,
  `/weather/sync` (POST), `/weather/embed` (POST), `/weather/search` (POST)
- `weather_client.py` - NWS API client: text products, active alerts, and an
  optional CPC outlook reader
- `embeddings.py` - text normalization, section-aware chunking, and
  sentence-transformers embedding, shared by the app, the notebook, and the CLI
- `weather_store.py` - Lakebase DDL/migrations, upsert, and pgvector search
  (`psycopg2`-based; used by `app.py` and `scripts/run_pipeline.py`, **not**
  imported by the notebook - see the driver note below)
- `lakebase.py` - connection helper (env var locally, Databricks secret when
  deployed; also `psycopg2`-based, same restriction as above)
- `setup_secrets.py` - one-time script to store the Lakebase URL as a secret
- `notebooks/ingest_weather_embeddings.py` - self-contained ETL notebook:
  harvest -> upsert -> chunk -> embed -> verify, runs on serverless compute.
  Uses `pg8000`, not `psycopg2` - see below
- `scripts/run_pipeline.py` - CLI for sync/embed/search without a server
  (`psycopg2`-based, via `weather_store.py`/`lakebase.py`)
- `sql/` - the same DDL as `weather_store.py`, as reviewable/applyable SQL
- `templates/index.html` - search UI
- `tests/test_pipeline.py` - offline tests (normalizers, chunking, SQL construction)
- `databricks.yml` + `resources/ingest_weather_embeddings_job.yml` - optional
  Asset Bundle config to schedule the notebook as a Workflow
- `.env.example` - local dev env var template

## Why the notebook uses a different Postgres driver than everything else

`app.py`, `weather_store.py`, `lakebase.py`, and `scripts/run_pipeline.py` all
use `psycopg2`. **`notebooks/ingest_weather_embeddings.py` does not** - it
connects with [`pg8000`](https://github.com/tlocke/pg8000) instead, and it
does not import `weather_store.py` or `lakebase.py` at all (both pull in
`psycopg2` as soon as they're imported).

The reason: `psycopg2`'s compiled C extension crashes the sandboxed Python
kernel used by **Databricks Serverless notebook compute** - not a normal
`ImportError`, but a hard `SIGABRT` that kills the entire kernel the instant
the module is imported. This is a documented constraint of that environment
(Databricks' own guidance is to avoid `psycopg2` on serverless notebooks), and
it's why Free Edition - which runs everything on serverless - needs a
different driver for the one piece of this project that's a notebook rather
than an app.

`app.py` is unaffected because Databricks Apps run in a normal, unsandboxed
container, not the notebook kernel - so it keeps `psycopg2` (via
`weather_store.py`/`lakebase.py`), and so does the CLI, which runs as a plain
script outside Databricks entirely.

The notebook reimplements the DDL, upsert, embedding-write, and search SQL
directly (rather than importing `weather_store.py`), using the exact same
statements - just executed through `pg8000`'s cursor instead of `psycopg2`'s.
If you change the schema or the upsert/search logic, update it in **both**
`weather_store.py` and the notebook's inline SQL.

## Step-by-step setup

### 1. Create a Lakebase instance and a native-password role

1. In your Databricks workspace, go to **Catalog** > **Lakebase** (or search
   "Lakebase" in the workspace search bar).
2. Click **Create Lakebase instance** (a.k.a. **Create database instance**).
   Give it a name (e.g. `weather-rag-db`), pick a size/region, click
   **Create**, and wait for **Available**.
3. Open the instance, go to **Roles & Databases** (a.k.a. **Permissions**).
4. Enable **native (password) authentication** if it isn't already on -
   some instances default to OAuth-only, and this app wants a static
   password so `lakebase.py` never needs token-refresh logic.
5. **Add role** > choose **Password** auth > name it (e.g. `weather_app`) >
   let Databricks generate a password.
6. Copy the connection URL shown for the role:

   ```
   postgresql://<role>:<password>@<host>.database.cloud.databricks.com:5432/databricks_postgres?sslmode=require
   ```

   You'll paste this into `setup_secrets.py` in step 2.

> Free Edition note: Lakebase instances are available on Free Edition. If
> your workspace shows a capacity/tier limit, the smallest size is enough for
> this app's corpus (a few hundred documents, a few thousand chunks).

### 2. Store the Lakebase secret

There is **no weather API key to create** - `api.weather.gov` authenticates
with a `User-Agent` header only. The only secret this app needs is the
Lakebase URL.

From a Databricks notebook or terminal:

```python
%sh python setup_secrets.py
```

This prompts (via `getpass`, so nothing is echoed or logged) for the
connection URL from step 1, and stores it as secret `database/lakebase-url`.

### 3. Configure environment variables (local dev)

```bash
cp .env.example .env
```

Edit `.env` and set `LAKEBASE_URL` to the same connection URL from step 1.
Also worth setting `NWS_USER_AGENT` to something with real contact info - NWS
asks for this so they can reach you if your client misbehaves; it's not
enforced, but it's the polite (and traceable) thing to do.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

On a CPU-only machine, if `torch` fails to build or takes too long:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 5. Run locally

```bash
python app.py
```

Then either open `http://localhost:8000/` for the search UI, or drive it
from the terminal:

```bash
curl -X POST localhost:8000/weather/sync \
  -H 'Content-Type: application/json' \
  -d '{"offices": ["OKX","FWD"], "limit_per_pair": 1, "embed": true}'

curl -X POST localhost:8000/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "flash flood risk this weekend"}'
```

Or use the CLI, which exercises the same modules without a server:

```bash
python scripts/run_pipeline.py all --query "flash flood risk this weekend"
```

### 6. Create a Git folder in Databricks and deploy the app

All through the workspace UI, no CLI required:

1. **Workspace** > **Create** > **Git folder**, paste this repo's URL, create.
2. **Compute** > **Apps** > **Create app** > **Custom**. Name it
   (e.g. `weather-rag`).
3. Point it at the Git folder from step 1 (the one containing `app.py` and
   `app.yaml`). Databricks reads `app.yaml` automatically for the command
   and environment variables, including the Lakebase secret scope/key.
4. Click **Deploy**. After it's up, hit `GET /healthz` to confirm
   `pgvector` is installed and the app can reach Lakebase, then `POST
   /weather/sync` to pull in a first batch of documents.

Whenever you update the code: **Git folder** > **Pull**, then **Deploy**
again in the Apps UI.

### 7. Seed and refresh the corpus

The app can drive its own corpus (`POST /weather/sync` then `POST
/weather/embed`, or `{"embed": true}` on the sync call), which is enough to
try the search endpoint immediately after deploying. For anything beyond a
demo, schedule the notebook instead - see below.

## Endpoints

- `GET /healthz` - liveness + a real Lakebase round trip + pgvector check
- `GET /weather/stats` - document/chunk counts, offices covered, freshness
- `GET /weather/documents?limit=50&office=OKX&product_code=AFD` - browse
  synced documents (metadata only, no full body)
- `POST /weather/sync` - harvest from NWS and upsert into `weather_documents`.
  Body (all optional): `offices`, `product_types`, `limit_per_pair`,
  `include_alerts`, `alert_area`, `alert_limit`, `include_cpc`, `embed`
- `POST /weather/embed` - chunk and vectorize any documents without current
  embeddings. Body (optional): `{"limit": 50}`
- `POST /weather/search` - semantic search. Body:
  `{"query": "...", "limit": 5, "office": "OKX", "product_code": "AFD",
  "source": "nws_alert", "since_hours": 24, "min_similarity": 0.2,
  "group_by_document": true}` - only `query` is required

## Scheduling the ingestion notebook

`notebooks/ingest_weather_embeddings.py` is a self-contained ETL: harvest,
upsert (skipping unchanged text via SHA-256), chunk, embed, write vectors,
verify with a sample similarity query. It runs on **serverless compute**, so
it works on Databricks Free Edition with no cluster to provision.

### Option A: Databricks Asset Bundle (CLI, version-controlled)

```bash
databricks bundle deploy -t dev
databricks bundle run ingest_weather_embeddings_job -t dev   # test once
```

Then flip `pause_status: PAUSED` to `UNPAUSED` in
`resources/ingest_weather_embeddings_job.yml` and redeploy for the schedule
(every 6 hours by default - AFDs reissue at least twice daily and alerts turn
over continuously, so a daily cadence would leave the corpus stale most of
the time).

### Option B: Workflows UI (no CLI required)

1. **Workflows** > **Jobs** > **Create Job**.
2. Task type **Notebook**, path -> `notebooks/ingest_weather_embeddings.py`
   in your Git folder. Compute: **Serverless** (or a small job cluster).
3. Under **Parameters**, add the widget values you want to override -
   defaults live at the top of the notebook and match
   `resources/ingest_weather_embeddings_job.yml`.
4. **Add trigger** > **Scheduled**, e.g. every 6 hours
   (`0 0 0,6,12,18 * * ?`, UTC).
5. Add a failure notification, then **Create** and **Run now** to validate.

## Applying the SQL by hand (optional)

`weather_store.ensure_schema()` creates both tables, the `pgvector`
extension, and the HNSW index automatically on app start and at the top of
the notebook - you don't need to run anything manually. The files in `sql/`
exist so the schema is reviewable outside the Python, and so you can apply
it through the Lakebase SQL editor if you'd rather not wait for the app to
do it. See `sql/README.md`.

## Notes

- **No API key for the weather source.** `api.weather.gov` and CPC both
  authenticate with nothing but a descriptive `User-Agent` - the only secret
  this app manages is the Lakebase URL.
- **`LAKEBASE_URL` env var is checked before the Databricks secret**, so
  local development with a plain `.env` works without any CLI/workspace
  setup; the deployed app falls through to the secret automatically since
  `app.yaml` doesn't set `LAKEBASE_URL` directly.
- **Embeddings are written directly into the `VECTOR` column** via an
  explicit `%s::vector` cast on a pgvector literal - no intermediate
  `double precision[]` staging column and no follow-up `UPDATE ...
  ::vector` pass.
- For the full rationale behind the source choice, schema, and chunking
  parameters, see [README_WEATHER.md](README_WEATHER.md).
