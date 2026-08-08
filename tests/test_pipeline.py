"""
Offline tests for the parts of the pipeline that don't need a database.

Run with:  python tests/test_pipeline.py     (or: pytest tests/)

Everything here is deterministic and network-free: the fixtures are real NWS
response shapes captured from api.weather.gov. What this covers is the code
most likely to break silently - the normalizers that flatten upstream JSON,
and the chunker, whose failures show up not as exceptions but as gradually
worse search results.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embeddings
import weather_client
import weather_store

# A real Area Forecast Discussion, abbreviated but structurally intact:
# WMO/AWIPS routing header, .SYNOPSIS/.SHORT TERM/.AVIATION sections, teletype
# line wrapping mid-sentence, an && separator and a $$ terminator.
SAMPLE_AFD = """404
FXUS61 KOKX 141920
AFDOKX

Area Forecast Discussion
National Weather Service New York NY
320 PM EDT Fri Jun 14 2024

.SYNOPSIS...
A frontal boundary remains draped across the region tonight into
Saturday. Waves of low pressure riding along this boundary will
bring periods of heavy rainfall through the weekend, with the
greatest flash flood threat Saturday afternoon and evening.

.NEAR TERM /THROUGH TONIGHT/...
Showers and thunderstorms will increase in coverage this evening
as a shortwave trough approaches from the west. Precipitable
water values climb to near 2.00 inches, which is around the 90th
percentile for mid June. Any storm that develops will be capable
of producing torrential downpours with rainfall rates approaching
2 inches per hour. Given the saturated antecedent conditions,
urban and poor drainage flooding is likely, and a Flash Flood
Watch may be needed for portions of the area.

.SHORT TERM /SATURDAY THROUGH SUNDAY/...
The heaviest rainfall is expected Saturday afternoon into
Saturday night as the frontal boundary stalls. Storm total
rainfall of 2 to 4 inches is anticipated, with locally higher
amounts possible where training convection sets up. Flash
flooding is a significant concern, particularly across the urban
corridor where runoff is enhanced.

.AVIATION /20Z FRIDAY THROUGH TUESDAY/...
VFR early, lowering to MVFR/IFR in showers overnight.

&&

.MARINE...
Small craft advisory conditions develop Saturday as southerly
flow increases ahead of the boundary. Seas build to 4 to 6 feet
on the ocean waters.

.HYDROLOGY...
Rainfall of this magnitude falling on already wet soils will
produce rapid rises on small streams and creeks. Flash flooding
of urban areas and small streams is likely.

$$

SYNOPSIS...JM
NEAR TERM...JM
"""

SAMPLE_PRODUCT_RESPONSE = {
    "@id": "https://api.weather.gov/products/abc-123",
    "id": "abc-123",
    "wmoCollectiveId": "FXUS61",
    "issuingOffice": "KOKX",
    "issuanceTime": "2024-06-14T19:20:00+00:00",
    "productCode": "AFD",
    "productName": "Area Forecast Discussion",
    "productText": SAMPLE_AFD,
}

SAMPLE_ALERT_FEATURE = {
    "id": "https://api.weather.gov/alerts/urn:oid:2.49.0.1.840.0.xyz",
    "properties": {
        "id": "urn:oid:2.49.0.1.840.0.xyz",
        "areaDesc": "Bronx, NY; Kings, NY; New York, NY",
        "sent": "2024-06-14T19:35:00-04:00",
        "effective": "2024-06-15T12:00:00-04:00",
        "expires": "2024-06-16T08:00:00-04:00",
        "severity": "Severe",
        "certainty": "Possible",
        "urgency": "Future",
        "event": "Flash Flood Watch",
        "senderName": "NWS New York NY",
        "headline": "Flash Flood Watch issued June 14 at 7:35PM EDT",
        "description": "Excessive rainfall is expected Saturday afternoon "
        "through Saturday night. Rainfall amounts of 2 to 4 inches are "
        "possible, which may cause flash flooding of urban areas and small "
        "streams.",
        "instruction": "Monitor later forecasts and be alert for possible "
        "Flash Flood Warnings. Do not drive through flooded roadways.",
    },
}

_failures: list[str] = []


def check(condition: bool, label: str) -> None:
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        _failures.append(label)


# ------------------------------------------------------------ normalizers --


def test_normalize_product():
    print("\nnormalize_product")
    doc = weather_client.normalize_product(SAMPLE_PRODUCT_RESPONSE)

    check(doc is not None, "returns a document")
    check(doc["id"] == "abc-123", "uses the upstream product id")
    check(doc["source"] == "nws_product", "tags the source")
    check(doc["product_code"] == "AFD", "keeps the product code")
    check(doc["office_id"] == "KOKX", "keeps the issuing office")
    check(doc["issued_at"] == "2024-06-14T19:20:00+00:00", "keeps issuance time")
    check("flash flood" in doc["raw_text"].lower(), "carries the body text")
    check(doc["payload"]["wmoCollectiveId"] == "FXUS61", "retains the raw payload")

    empty = weather_client.normalize_product({**SAMPLE_PRODUCT_RESPONSE, "productText": "  "})
    check(empty is None, "drops products with no text")


def test_normalize_alert():
    print("\nnormalize_alert")
    doc = weather_client.normalize_alert(SAMPLE_ALERT_FEATURE)

    check(doc["source"] == "nws_alert", "tags the source")
    check(doc["product_code"] == "ALERT", "uses ALERT as the code")
    check(doc["product_name"] == "Flash Flood Watch", "keeps the event name")
    check(doc["severity"] == "Severe", "keeps severity")
    check(doc["expires_at"] == "2024-06-16T08:00:00-04:00", "keeps expiry")
    # All three prose fields must survive into one body - a hit on the
    # description is useless if the instruction got dropped.
    body = doc["raw_text"]
    check("Excessive rainfall" in body, "includes the description")
    check("Do not drive through flooded roadways" in body, "includes the instruction")
    check("Affected area: Bronx" in body, "includes the area description")

    check(weather_client.normalize_alert({"properties": {}}) is None, "drops empty alerts")


def test_extract_graph():
    print("\n_extract_graph")
    check(
        weather_client._extract_graph({"@graph": [{"id": "1"}]}) == [{"id": "1"}],
        "reads @graph collections",
    )
    check(
        weather_client._extract_graph({"features": [{"id": "2"}]}) == [{"id": "2"}],
        "reads GeoJSON feature collections",
    )
    check(weather_client._extract_graph([{"id": "3"}]) == [{"id": "3"}], "accepts a bare list")
    check(weather_client._extract_graph({"unexpected": 1}) == [], "returns [] on a shape change")


# ------------------------------------------------------------------ text --


def test_normalize_text():
    print("\nnormalize_product_text")
    normalized = embeddings.normalize_product_text(SAMPLE_AFD)

    # The teletype wrap breaks "Waves of low pressure riding along this\nboundary".
    # Unwrapping is the whole point of this function.
    check(
        "Waves of low pressure riding along this boundary" in normalized,
        "rejoins sentences split by teletype line wrapping",
    )
    check(".SYNOPSIS..." in normalized, "keeps section headers on their own lines")
    check("&&" in normalized, "keeps the && separator")
    check("FXUS61 KOKX" not in normalized, "strips the WMO routing header")
    check("\n\n\n" not in normalized, "collapses blank-line runs")


def test_split_sections():
    print("\nsplit_sections")
    sections = embeddings.split_sections(embeddings.normalize_product_text(SAMPLE_AFD))

    check(len(sections) >= 3, f"finds multiple sections (got {len(sections)})")
    joined = "\n".join(sections)
    check("SHORT TERM" in joined, "keeps the SHORT TERM section")
    check("HYDROLOGY" in joined, "keeps content from after the && separator")
    # ".AVIATION...VFR early..." is two lines and would be a junk vector on its
    # own; it must have been merged into a neighbour.
    aviation_alone = [s for s in sections if s.startswith(".AVIATION") and len(s) < 200]
    check(not aviation_alone, "merges undersized sections instead of emitting stubs")


def test_chunking():
    print("\nchunk_weather_text")
    chunks = embeddings.chunk_weather_text(SAMPLE_AFD, chunk_size=400, overlap=80)

    check(len(chunks) > 1, f"produces multiple chunks (got {len(chunks)})")
    check(all(len(c) <= 400 for c in chunks), "no chunk exceeds chunk_size")
    check(all(len(c) >= 40 for c in chunks), "no chunk is below the noise floor")
    check(
        any("flash flood" in c.lower() for c in chunks),
        "the flash-flood language survives chunking",
    )

    # Overlap exists so a sentence straddling a boundary is still retrievable
    # in full from at least one chunk.
    big = embeddings.chunk_weather_text("word " * 600, chunk_size=500, overlap=100)
    check(len(big) > 1, "windows a long section with no internal structure")
    overlapped = any(
        big[i][-50:].strip() and big[i][-50:].strip()[:20] in big[i + 1]
        for i in range(len(big) - 1)
    )
    check(overlapped, "consecutive windows actually overlap")

    check(embeddings.chunk_weather_text("") == [], "empty input yields no chunks")
    check(embeddings.chunk_weather_text("   \n\n  ") == [], "whitespace yields no chunks")

    # Alerts have no NWS section markers, so they take the windowing path.
    alert_doc = weather_client.normalize_alert(SAMPLE_ALERT_FEATURE)
    alert_chunks = embeddings.chunk_weather_text(alert_doc["raw_text"])
    check(len(alert_chunks) >= 1, "alerts chunk without section markers")


def test_dimensions_and_vectors():
    print("\ndimensions and pgvector literals")
    check(embeddings.resolve_dimension("sentence-transformers/all-MiniLM-L6-v2") == 384, "MiniLM is 384")
    check(embeddings.resolve_dimension("BAAI/bge-large-en-v1.5") == 1024, "bge-large is 1024")

    try:
        embeddings.resolve_dimension("some/unregistered-model")
        check(False, "unknown models raise rather than guessing a width")
    except ValueError:
        check(True, "unknown models raise rather than guessing a width")

    literal = embeddings.to_pgvector([0.1, -0.25, 3.0])
    check(literal == "[0.1,-0.25,3]", f"renders a pgvector literal (got {literal})")
    check(embeddings.to_pgvector([]) == "[]", "handles the empty vector")


def test_digest_and_ddl():
    print("\nstore helpers")
    a = weather_store.text_digest("forecast text")
    b = weather_store.text_digest("forecast text")
    c = weather_store.text_digest("forecast text!")
    check(a == b and a != c, "digest is stable and change-sensitive")

    ddl = weather_store.embeddings_ddl(dim=384)
    joined = " ".join(ddl)
    check("CREATE EXTENSION IF NOT EXISTS vector" in joined, "enables pgvector")
    check("VECTOR(384)" in joined, "declares the vector width")
    check("hnsw (embedding vector_cosine_ops)" in joined, "builds an HNSW cosine index")
    check("ON DELETE CASCADE" in joined, "cascades deletes from documents")

    doc_ddl = " ".join(weather_store.documents_ddl())
    check("text_sha256" in doc_ddl, "stores the change-detection digest")
    check("id              TEXT PRIMARY KEY" in doc_ddl, "keys on the upstream id")


def test_search_sql_shape():
    """
    Build the search SQL without a database by capturing what would execute.

    Guards the query construction: that filters land in the WHERE clause, that
    the cosine operator matches the index's operator class, and that grouping
    pulls a wider candidate set than it returns.
    """
    print("\nsearch SQL construction")

    captured = {}

    class FakeCursor:
        description = [("chunk_id",), ("similarity",)]

        def execute(self, sql, params=None):
            captured["sql"] = sql
            captured["params"] = params

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeConn:
        def cursor(self):
            return FakeCursor()

    weather_store.search(
        FakeConn(),
        query_vector=[0.1] * 384,
        limit=5,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        office_id="okx",
        product_code="afd",
        since_hours=48,
        min_similarity=0.2,
    )

    sql, params = captured["sql"], captured["params"]
    check("<=>" in sql, "uses the cosine distance operator")
    check("1 - (e.embedding <=> %(vector)s::vector)" in sql, "converts distance to similarity")
    check("d.office_id = %(office_id)s" in sql, "applies the office filter")
    check("make_interval(hours => %(since_hours)s)" in sql, "applies the time filter")
    check("similarity >= %(min_similarity)s" in sql, "applies the similarity floor")
    check("DISTINCT ON (document_id)" in sql, "dedupes to one chunk per document")
    check(params["office_id"] == "OKX", "uppercases the office filter")
    check(params["product_code"] == "AFD", "uppercases the product code filter")
    check(params["candidates"] > params["limit"], "over-fetches candidates before deduping")
    check(params["vector"].startswith("[0.1,"), "passes the vector as a pgvector literal")

    # Ungrouped search must not over-fetch - it returns exactly `limit` rows.
    weather_store.search(FakeConn(), query_vector=[0.1] * 384, limit=3, group_by_document=False)
    check(
        "DISTINCT ON" not in captured["sql"],
        "skips deduplication when group_by_document is off",
    )


def test_upsert_row_building():
    """The upsert must skip empty bodies and emit one tuple per real document."""
    print("\nupsert row building")

    captured = {}

    class FakeCursor:
        rowcount = 0

        def execute(self, sql, params=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            captured["committed"] = True

    def fake_execute_values(cur, sql, rows, template=None, page_size=100, fetch=False):
        captured["rows"] = rows
        captured["sql"] = sql
        captured["template"] = template
        return [(r[0],) for r in rows] if fetch else None

    original = weather_store.execute_values
    weather_store.execute_values = fake_execute_values
    try:
        docs = [
            weather_client.normalize_product(SAMPLE_PRODUCT_RESPONSE),
            weather_client.normalize_alert(SAMPLE_ALERT_FEATURE),
            {"id": "blank", "raw_text": "   ", "source": "nws_product", "product_code": "AFD"},
        ]
        result = weather_store.upsert_documents(FakeConn(), docs)
    finally:
        weather_store.execute_values = original

    check(len(captured["rows"]) == 2, "skips the document with a blank body")
    check(result["received"] == 3, "reports everything it was handed")
    check("ON CONFLICT (id) DO UPDATE" in captured["sql"], "upserts rather than duplicating")
    check(
        "text_sha256 IS DISTINCT FROM EXCLUDED.text_sha256" in captured["sql"],
        "skips writes when the text has not changed",
    )
    check("%s::jsonb" in captured["template"], "casts the payload to jsonb")
    check(captured["committed"], "commits the transaction")


def main() -> int:
    print("Weather RAG pipeline - offline tests")
    print("=" * 60)
    for test in (
        test_normalize_product,
        test_normalize_alert,
        test_extract_graph,
        test_normalize_text,
        test_split_sections,
        test_chunking,
        test_dimensions_and_vectors,
        test_digest_and_ddl,
        test_search_sql_shape,
        test_upsert_row_building,
    ):
        test()

    print("\n" + "=" * 60)
    if _failures:
        print(f"{len(_failures)} FAILED:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
