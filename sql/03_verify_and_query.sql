-- Post-ingest checks and example queries.

-- 1. Coverage: documents in, chunks out, per product type.
SELECT d.product_code,
       count(DISTINCT d.id)     AS documents,
       count(e.id)              AS chunks,
       round(avg(e.char_count)) AS avg_chunk_chars,
       max(d.issued_at)         AS newest
FROM weather_documents d
LEFT JOIN weather_embeddings e ON e.document_id = d.id
GROUP BY d.product_code
ORDER BY documents DESC;

-- 2. Anything synced but not yet embedded? Should be zero after a full run.
SELECT count(*) AS unembedded_documents
FROM weather_documents d
WHERE NOT EXISTS (SELECT 1 FROM weather_embeddings e WHERE e.document_id = d.id);

-- 3. Confirm the vector index is being used. Expect an "Index Scan using
--    idx_weather_embeddings_vector" in the plan; a Seq Scan means the index
--    is missing, or the table is still too small for the planner to bother -
--    which is fine at demo scale.
EXPLAIN ANALYZE
WITH seed AS (SELECT embedding FROM weather_embeddings LIMIT 1)
SELECT id
FROM weather_embeddings
ORDER BY embedding <=> (SELECT embedding FROM seed)
LIMIT 5;

-- 4. Nearest neighbours to an existing chunk - a cheap sanity check on vector
--    quality that needs no model to run. Passages about the same hazard should
--    cluster; if the top matches look unrelated, chunking or the model is the
--    problem, not the SQL.
WITH seed AS (SELECT embedding FROM weather_embeddings ORDER BY embedded_at DESC LIMIT 1)
SELECT e.document_id,
       e.chunk_index,
       left(e.chunk_text, 120) AS preview,
       round((1 - (e.embedding <=> (SELECT embedding FROM seed)))::numeric, 4) AS similarity
FROM weather_embeddings e
ORDER BY e.embedding <=> (SELECT embedding FROM seed)
LIMIT 10;

-- 5. Freshness. Weather text goes stale fast; anything older than a day or
--    two is probably not what a "this weekend" query should return.
SELECT source,
       count(*) AS documents,
       min(issued_at) AS oldest,
       max(issued_at) AS newest,
       round((extract(epoch FROM now() - max(issued_at)) / 3600)::numeric, 1) AS hours_since_newest
FROM weather_documents
GROUP BY source;
