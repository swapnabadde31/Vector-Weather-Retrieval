# SQL setup for Lakebase

These files document the schema and can be applied by hand through the
Lakebase SQL editor, but **you normally don't need to run them**.
`weather_store.ensure_schema()` executes the same statements automatically on
Flask app start and at the top of the ingestion notebook.

| File | Purpose |
|---|---|
| `01_setup_weather_documents.sql` | Raw document store + filter indexes |
| `02_setup_weather_embeddings.sql` | `pgvector` extension, chunk/vector table, HNSW index |
| `03_verify_and_query.sql` | Post-ingest checks and example similarity queries |

Run them in order if you apply them manually.

## Why this differs from the day-2 news pipeline

The reference `ticker_news` pipeline inserted embeddings into a
`double precision[]` column and then required a manual follow-up pass:

```sql
UPDATE ticker_news_embeddings SET embedding = embedding::vector WHERE embedding IS NOT NULL;
```

That step is gone here. `psycopg2` writes directly into the `VECTOR` column
because the insert casts a pgvector literal explicitly:

```sql
INSERT INTO weather_embeddings (..., embedding, ...) VALUES (..., %s::vector, ...)
```

with the parameter rendered as `'[0.1,0.2,...]'` by `embeddings.to_pgvector`.
One less manual step, and no window in which the table holds vectors the HNSW
index can't use.

## Changing the embedding model

`VECTOR(n)` is fixed at DDL time, so a model swap is a migration:

```sql
ALTER TABLE weather_embeddings ALTER COLUMN embedding TYPE VECTOR(768);
```

Then re-run the notebook with `reembed_all=true`. `weather_store.search`
filters on `model_name`, so mixed-model rows return no results rather than
silently ranking incomparable vectors against each other.
