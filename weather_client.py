"""
Client for the NOAA/NWS public weather API (https://api.weather.gov).

This is the weather analogue of `massive_client.py` in the day-2 reference
app: it is the single place that knows how to talk to the upstream source and
how to flatten its responses into the normalized "document" shape that the
rest of the pipeline (weather_store -> embeddings -> Flask search) consumes.

Why api.weather.gov and not cpc.ncep.noaa.gov: see README_WEATHER.md. Short
version - the NWS API is a real REST/JSON service with stable product IDs and
issuance timestamps, which is exactly what an idempotent upsert pipeline
needs, while CPC publishes flat text/GIF files on a web server with no IDs.
A small CPC reader is included at the bottom of this module anyway, because
CPC's 6-10 day and monthly outlooks cover a forecast horizon the NWS text
products do not.

Auth: none. NWS requires only a descriptive User-Agent header (they ask for
contact info so they can reach you if your client misbehaves). That is why
this pipeline needs no API-key secret at all - a meaningful advantage on
Databricks Free Edition.

Normalized document shape returned by every public method here:

    {
        "id":            str,   # stable upstream identifier (primary key)
        "source":        str,   # nws_product | nws_alert | cpc_outlook
        "product_code":  str,   # AFD | HWO | ESF | ALERT | CPC_610 ...
        "product_name":  str,   # "Area Forecast Discussion", "Flash Flood Watch"
        "office_id":     str,   # issuing WFO, e.g. OKX
        "wmo_id":        str,
        "headline":      str | None,
        "area_desc":     str | None,
        "severity":      str | None,
        "certainty":     str | None,
        "urgency":       str | None,
        "issued_at":     str | None,   # ISO 8601
        "effective_at":  str | None,
        "expires_at":    str | None,
        "raw_text":      str,   # THE unstructured payload we embed
        "payload":       dict,  # full upstream JSON, kept for replay/debug
    }
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterator

import requests

logger = logging.getLogger(__name__)

NWS_API_BASE_URL = os.environ.get("NWS_API_BASE_URL", "https://api.weather.gov")

# NWS asks every client to identify itself. Override via env so a deployed app
# advertises a real contact address.
NWS_USER_AGENT = os.environ.get(
    "NWS_USER_AGENT",
    "(databricks-lakebase-weather-rag, weather-rag@example.com)",
)

_DEFAULT_TIMEOUT = 30
_DEFAULT_RPM = 60  # NWS publishes no hard quota; 1 req/sec is a polite ceiling.

# Text product types worth embedding. These are the narrative, human-written
# products - the ones with actual prose in them - as opposed to the tabular or
# coded products (CLI, RTP, SHF) that would embed into noise.
#
#   AFD  Area Forecast Discussion  - forecaster's reasoning, 2+ per day per WFO
#   HWO  Hazardous Weather Outlook - 7-day hazard narrative, daily per WFO
#   ESF  Hydrologic Outlook        - flood/river narrative, event driven
#   PNS  Public Information Stmt   - storm reports and impact narratives
DEFAULT_PRODUCT_TYPES = [
    t.strip().upper()
    for t in os.environ.get("NWS_PRODUCT_TYPES", "AFD,HWO,ESF").split(",")
    if t.strip()
]

# A geographically spread default set of Weather Forecast Offices, chosen so a
# first sync returns text covering several distinct hazard regimes (coastal
# flooding, plains convection, mountain snow, tropical, fire weather).
DEFAULT_OFFICES = [
    o.strip().upper()
    for o in os.environ.get(
        "NWS_OFFICES",
        "OKX,LWX,MPX,FWD,SEW,MFL,BOU,OUN,SLC,TAE",
    ).split(",")
    if o.strip()
]

# Product code -> friendly name, used when the API omits productName.
_PRODUCT_NAMES = {
    "AFD": "Area Forecast Discussion",
    "HWO": "Hazardous Weather Outlook",
    "ESF": "Hydrologic Outlook",
    "PNS": "Public Information Statement",
    "FFA": "Flood Watch",
    "SPS": "Special Weather Statement",
    "NPW": "Non-Precipitation Weather Statement",
    "RFW": "Red Flag Warning",
    "WSW": "Winter Storm Warning",
}


class NWSClientError(RuntimeError):
    """Raised when the NWS API cannot be reached or returns an unusable body."""


class NWSClient:
    """
    Thin, polite wrapper around api.weather.gov.

    Handles the three things that actually bite you against this API:
      1. A User-Agent header is mandatory (bare requests get 403).
      2. There is no documented rate limit, but sustained bursts get throttled,
         so calls are spaced to `max_requests_per_minute`.
      3. Transient 429/5xx responses are common during severe weather, when
         everyone is hammering the API - so those are retried with backoff.
    """

    def __init__(
        self,
        base_url: str | None = None,
        user_agent: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        max_requests_per_minute: int = _DEFAULT_RPM,
        max_retries: int = 3,
    ):
        self.base_url = (base_url or NWS_API_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._min_interval = 60.0 / max(1, max_requests_per_minute)
        self._last_request_at = 0.0

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent or NWS_USER_AGENT,
                "Accept": "application/geo+json",
            }
        )

    # ---------------------------------------------------------------- HTTP --

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        """GET a path relative to the API root, with throttling and retries."""
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(2**attempt)
                continue

            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = requests.HTTPError(
                    f"{resp.status_code} from {url}", response=resp
                )
                # Respect Retry-After when the server sends one.
                wait = resp.headers.get("Retry-After")
                time.sleep(float(wait) if wait and wait.isdigit() else 2**attempt)
                continue

            resp.raise_for_status()
            try:
                return resp.json()
            except ValueError as exc:
                raise NWSClientError(f"Non-JSON response from {url}") from exc

        raise NWSClientError(f"GET {url} failed after {self.max_retries} attempts: {last_error}")

    # ------------------------------------------------------- text products --

    def list_products(
        self,
        product_type: str,
        office: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """
        List recent product *metadata* for one type (and optionally one office).

        Returns the `@graph` array from the JSON-LD response - each entry has
        id / wmoCollectiveId / issuingOffice / issuanceTime / productCode /
        productName, but NOT the text body. Fetching the body is a second call
        (`get_product`), which is why syncs are two-phase.

        Tries with a server-side `limit` first, since without it this
        endpoint can return a WFO's entire product history rather than just
        the most recent few. If NWS rejects the parameter (a plain 400, not
        a transient 429/5xx that `get()` already retries), the request is
        retried once without `limit` and truncated client-side instead -
        `iter_text_products` also slices its result, so a caller two levels
        up sees no difference either way.

        This same class of parameter is what became unsupported on
        `/alerts/active` (see `get_active_alerts`), so this endpoint gets the
        same defensive handling even without direct evidence it needs it yet.
        """
        if office:
            path = f"/products/types/{product_type}/locations/{office}"
        else:
            path = f"/products/types/{product_type}"

        try:
            data = self.get(path, params={"limit": limit})
            return _extract_graph(data)[:limit]
        except (requests.HTTPError, NWSClientError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status != 400:
                raise
            logger.info(
                "%s rejected the 'limit' parameter (400); retrying without it", path
            )
            data = self.get(path, params={})
            return _extract_graph(data)[:limit]

    def get_product(self, product_id: str) -> dict:
        """Fetch one product including its `productText` body."""
        return self.get(f"/products/{product_id}")

    def iter_text_products(
        self,
        product_types: list[str] | None = None,
        offices: list[str] | None = None,
        limit_per_pair: int = 2,
    ) -> Iterator[dict]:
        """
        Yield normalized documents for the cartesian product of
        (product_types x offices), newest first.

        `limit_per_pair` is deliberately small by default: an AFD runs
        3-8 KB of prose, so 3 types x 10 offices x 2 products is already
        ~50 documents and several hundred chunks - plenty to demo retrieval
        without hammering a free public API.
        """
        product_types = product_types or DEFAULT_PRODUCT_TYPES
        offices = offices or DEFAULT_OFFICES

        for product_type in product_types:
            for office in offices:
                try:
                    entries = self.list_products(product_type, office, limit=limit_per_pair)
                except (requests.HTTPError, NWSClientError) as exc:
                    # A WFO that has not issued this product type recently
                    # returns 404. That is normal, not an error worth failing on.
                    logger.info("No %s products for %s (%s)", product_type, office, exc)
                    continue

                for entry in entries[:limit_per_pair]:
                    product_id = entry.get("id") or entry.get("@id")
                    if not product_id:
                        continue
                    try:
                        full = self.get_product(product_id)
                    except (requests.HTTPError, NWSClientError) as exc:
                        logger.warning("Could not fetch product %s: %s", product_id, exc)
                        continue

                    doc = normalize_product(full, fallback_office=office)
                    if doc:
                        yield doc

    # --------------------------------------------------------------- alerts --

    def get_active_alerts(
        self,
        area: str | list[str] | None = None,
        limit: int = 50,
        severity: str | list[str] | None = None,
    ) -> list[dict]:
        """
        Fetch currently active alerts (watches / warnings / advisories).

        Alerts carry the most retrieval-relevant prose in the whole NWS
        catalog - a Flash Flood Watch's `description` is a direct answer to
        "flash flood risk this weekend" - and they are the fastest-moving
        documents in the corpus, which is what makes the re-embed-on-change
        logic in weather_store worth having.

        `area` is a state/marine code such as "TX" or a list of them.

        `limit` is enforced client-side (the response is truncated in
        Python after fetching), not sent as a query parameter. NWS's
        supported parameters on `/alerts/active` have changed before without
        notice - `limit` was accepted at one point and stopped being
        accepted at another - and there is no version pin on this endpoint
        to detect that ahead of time. Trimming locally means this method
        keeps working across that kind of change instead of depending on it.
        The cost is fetching (and normalizing) the full active-alert set
        rather than a server-truncated one; nationwide active alerts are at
        most a few hundred records, so this is a non-issue in practice.
        """
        params: dict[str, Any] = {"status": "actual", "message_type": "alert"}
        if area:
            params["area"] = ",".join(area) if isinstance(area, list) else area
        if severity:
            params["severity"] = ",".join(severity) if isinstance(severity, list) else severity

        data = self.get("/alerts/active", params=params)
        features = data.get("features") or []
        if limit:
            features = features[:limit]
        docs = []
        for feature in features:
            doc = normalize_alert(feature)
            if doc:
                docs.append(doc)
        return docs


# ------------------------------------------------------------ normalizers --


def _extract_graph(data: dict) -> list[dict]:
    """
    Pull the list of items out of an NWS JSON-LD collection response.

    The API returns `@graph` for product collections and `features` for
    GeoJSON collections; some proxies flatten it to a bare list. Handle all
    three so a change upstream degrades to "no results" rather than a crash.
    """
    if isinstance(data, list):
        return data
    for key in ("@graph", "graph", "features", "products"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def normalize_product(product: dict, fallback_office: str | None = None) -> dict | None:
    """Flatten a /products/{id} response into the shared document shape."""
    text = (product.get("productText") or "").strip()
    if not text:
        return None

    code = (product.get("productCode") or "").upper()
    product_id = product.get("id") or product.get("@id")
    if not product_id:
        return None

    return {
        "id": str(product_id),
        "source": "nws_product",
        "product_code": code or "UNKNOWN",
        "product_name": product.get("productName") or _PRODUCT_NAMES.get(code, code),
        "office_id": (product.get("issuingOffice") or fallback_office or "").upper() or None,
        "wmo_id": product.get("wmoCollectiveId"),
        "headline": _first_headline(text),
        "area_desc": None,
        "severity": None,
        "certainty": None,
        "urgency": None,
        "issued_at": product.get("issuanceTime"),
        "effective_at": product.get("issuanceTime"),
        "expires_at": None,
        "raw_text": text,
        "payload": product,
    }


def normalize_alert(feature: dict) -> dict | None:
    """
    Flatten one GeoJSON alert feature into the shared document shape.

    An alert's usable prose is spread across three fields, so they are joined
    into one body with labelled sections: the headline states what and when,
    the description states why, and the instruction states what to do. All
    three matter for retrieval, and keeping them in one document means a
    single hit returns the complete picture.
    """
    props = feature.get("properties") or {}
    alert_id = props.get("id") or feature.get("id")
    if not alert_id:
        return None

    parts = []
    if props.get("headline"):
        parts.append(str(props["headline"]))
    if props.get("areaDesc"):
        parts.append(f"Affected area: {props['areaDesc']}")
    if props.get("description"):
        parts.append(str(props["description"]))
    if props.get("instruction"):
        parts.append(f"Instructions: {props['instruction']}")

    text = "\n\n".join(p.strip() for p in parts if p and p.strip())
    if not text:
        return None

    return {
        "id": str(alert_id),
        "source": "nws_alert",
        "product_code": "ALERT",
        "product_name": props.get("event") or "Weather Alert",
        "office_id": _office_from_sender(props.get("senderName")),
        "wmo_id": None,
        "headline": props.get("headline"),
        "area_desc": props.get("areaDesc"),
        "severity": props.get("severity"),
        "certainty": props.get("certainty"),
        "urgency": props.get("urgency"),
        "issued_at": props.get("sent"),
        "effective_at": props.get("effective") or props.get("onset"),
        "expires_at": props.get("expires") or props.get("ends"),
        "raw_text": text,
        "payload": feature,
    }


def _first_headline(text: str) -> str | None:
    """
    Derive a one-line summary for a text product.

    NWS products often carry a bracketed summary near the top, e.g.
    `...Strong to severe storms possible late this afternoon...`. Failing
    that, use the first real sentence of prose.

    The text must be unwrapped first. These products are wrapped at ~69
    columns with newlines falling mid-sentence, so reading "the first line"
    off the raw body yields a fragment that stops mid-clause - which is
    exactly what then gets displayed as the result headline.
    """
    from embeddings import normalize_product_text

    body = normalize_product_text(text)

    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("...") and stripped.endswith("...") and len(stripped) > 10:
            return stripped.strip(". ").strip()

    for line in body.splitlines():
        stripped = line.strip()
        if len(stripped) < 40 or stripped.startswith((".", "&", "$")):
            continue
        # Cut at the first sentence end so the headline is a complete thought
        # rather than an arbitrary character count.
        for end in (". ", "? ", "! "):
            index = stripped.find(end)
            if 40 <= index <= 240:
                return stripped[: index + 1].strip()
        return stripped[:240].rsplit(" ", 1)[0] + ("..." if len(stripped) > 240 else "")
    return None


def _office_from_sender(sender: str | None) -> str | None:
    """
    Alerts identify their office by name ("NWS Norman OK") rather than by the
    3-letter ID that text products use, so there is no reliable mapping back
    to a WFO code. Return None rather than guess wrong - the office filter is
    then simply not available for alerts, which is the honest behaviour.
    """
    return None


# ------------------------------------------------- optional: CPC outlooks --

CPC_TEXT_PRODUCTS = {
    # NCEP/CPC publishes these as plain-text files rather than through an API.
    "CPC_610": (
        "6-10 Day Outlook Discussion",
        "https://www.cpc.ncep.noaa.gov/products/predictions/610day/fxus07.html",
    ),
    "CPC_HAZARDS": (
        "US Hazards Outlook Discussion",
        "https://www.cpc.ncep.noaa.gov/products/predictions/threats/threats.php",
    ),
}


def fetch_cpc_outlooks(
    keys: list[str] | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> list[dict]:
    """
    Optional secondary source: CPC's extended-range outlook discussions.

    Off by default. Included to make the source comparison in README_WEATHER.md
    concrete rather than theoretical, and because CPC covers the 6-10 day and
    monthly horizon that NWS text products do not reach.

    The tradeoff is visible right here in the code: there is no product ID to
    key on, so the ID has to be synthesized from the URL, and there is no
    issuance timestamp in the payload, so "has this changed?" can only be
    answered by hashing the body. That is exactly why this is the fallback
    source and not the primary one.
    """
    import hashlib
    import re
    from datetime import datetime, timezone

    keys = keys or list(CPC_TEXT_PRODUCTS)
    docs = []

    for key in keys:
        if key not in CPC_TEXT_PRODUCTS:
            continue
        name, url = CPC_TEXT_PRODUCTS[key]
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": NWS_USER_AGENT})
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Could not fetch CPC product %s: %s", key, exc)
            continue

        # Strip HTML tags crudely - these pages are <pre>-wrapped plain text,
        # not real markup, so a parser would be overkill.
        body = re.sub(r"<[^>]+>", "", resp.text)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if not body:
            continue

        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
        docs.append(
            {
                "id": f"cpc:{key}:{digest}",
                "source": "cpc_outlook",
                "product_code": key,
                "product_name": name,
                "office_id": "CPC",
                "wmo_id": None,
                "headline": name,
                "area_desc": "Contiguous United States",
                "severity": None,
                "certainty": None,
                "urgency": None,
                "issued_at": datetime.now(timezone.utc).isoformat(),
                "effective_at": None,
                "expires_at": None,
                "raw_text": body,
                "payload": {"url": url, "fetched_at": datetime.now(timezone.utc).isoformat()},
            }
        )

    return docs
