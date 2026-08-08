"""
Text preparation and vectorization for weather documents.

Kept separate from both the API client and the database layer so the notebook,
the Flask app, and the local CLI all chunk and embed text *identically*. If
the query is embedded differently from the corpus, cosine similarity is
meaningless - so there is exactly one implementation of each step, here.

Two things in this module are specific to weather text and are the main
reason it isn't just a generic splitter:

  1. `normalize_product_text` unwraps NWS hard line breaks. Text products are
     teletype heritage: wrapped at ~69 columns with a newline mid-sentence.
     Feeding that to a sentence transformer fragments sentences across the
     tokenizer and measurably degrades retrieval. Paragraph breaks, section
     headers, and the `&&` / `$$` delimiters are preserved.

  2. `chunk_weather_text` splits on the product's own section markers first
     (`.SHORT TERM...`, `.AVIATION...`, `&&`) and only falls back to a sliding
     character window inside sections that are still too long. An AFD's
     sections are already topic-coherent - splitting on them means a chunk is
     about one thing, which is what you want a retrieved passage to be.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Vector width per model. The pgvector column is VECTOR(N) with a fixed N, so
# this table and the DDL must agree; changing models means a migration.
MODEL_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}

# ~900 characters is roughly 200 wordpiece tokens of weather prose, which sits
# comfortably under all-MiniLM-L6-v2's 256-token ceiling. Going wider means the
# model silently truncates and the tail of every chunk stops contributing to
# the vector - the failure mode is invisible, which is why the default is
# conservative rather than maximal.
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 900))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 150))

# Sections shorter than this get merged into the following one. NWS products
# are full of two-line stubs (".AVIATION...VFR." and similar) that carry no
# retrievable meaning on their own but add noise as standalone vectors.
_MIN_SECTION_CHARS = 200

# A section header in an NWS text product: a line beginning with a dot,
# followed by an all-caps label, e.g. ".SHORT TERM /TONIGHT THROUGH WEDNESDAY/..."
_SECTION_HEADER_RE = re.compile(r"^\.[A-Z][A-Z0-9 /&()\-\.]{2,80}\.\.\.", re.MULTILINE)

# Product terminators and section separators used by NWS/AWIPS.
_DELIMITER_RE = re.compile(r"^\s*(?:&&|\$\$)\s*$", re.MULTILINE)

# The routing header block at the top of every product (WMO id, AWIPS id,
# office name, issuance line). Useful metadata, but it is already stored in
# dedicated columns, and embedding it makes every document look similar.
_HEADER_LINE_RE = re.compile(
    r"^(?:\d{3}\s*$|[A-Z]{4}\d{2} [A-Z]{4} \d{6}|[A-Z]{3}[A-Z]{3}\s*$)", re.MULTILINE
)


def resolve_dimension(model_name: str | None = None) -> int:
    """Return the vector width for a model, failing loudly on unknown models."""
    model_name = model_name or DEFAULT_MODEL
    try:
        return MODEL_DIMENSIONS[model_name]
    except KeyError:
        raise ValueError(
            f"Unknown embedding model {model_name!r}. Add its output dimension to "
            "MODEL_DIMENSIONS in embeddings.py, and make sure the VECTOR(n) column "
            "in weather_embeddings matches before ingesting."
        ) from None


# ------------------------------------------------------------ text hygiene --


def normalize_product_text(text: str, strip_header: bool = True) -> str:
    """
    Turn teletype-wrapped NWS text into ordinary paragraphs.

    Joins lines that are continuations of the same sentence, while keeping
    blank lines, section headers, and `&&` / `$$` delimiters intact so the
    chunker can still see the document's structure.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if strip_header:
        # Drop the leading routing block: everything before the first blank
        # line, but only if it looks like WMO/AWIPS routing rather than prose.
        head, sep, tail = text.partition("\n\n")
        if sep and _HEADER_LINE_RE.search(head) and len(head) < 400:
            text = tail

        # Then drop the title block ("Area Forecast Discussion / National
        # Weather Service New York NY / 320 PM EDT Fri Jun 14 2024"). Every
        # product of a given type carries a near-identical version of this,
        # so embedding it pushes all documents slightly closer together and
        # costs retrieval precision for nothing - the office, product name,
        # and issuance time are all stored as queryable columns already.
        head, sep, tail = text.partition("\n\n")
        if sep and "National Weather Service" in head and len(head) < 300:
            text = tail

    lines = text.split("\n")
    out: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            out.append(" ".join(part.strip() for part in buffer if part.strip()))
            buffer.clear()

    for line in lines:
        stripped = line.strip()
        is_structural = (
            not stripped
            or stripped in ("&&", "$$")
            or stripped.startswith(".")
            and stripped.endswith("...")
            or stripped.startswith("...")
        )
        if is_structural:
            flush()
            out.append(stripped)
        else:
            buffer.append(stripped)
    flush()

    # Collapse runs of blank lines produced by the joining pass.
    normalized = "\n".join(out)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def split_sections(text: str) -> list[str]:
    """
    Split a product on its own section markers, merging stubs forward.

    Returns the document as a list of topic-coherent blocks. For documents
    with no recognizable sections (alerts, CPC outlooks) this returns the
    whole body as a single block, and the character windower takes over.
    """
    if not text:
        return []

    # First split on hard delimiters, then on section headers within each part.
    blocks: list[str] = []
    for part in _DELIMITER_RE.split(text):
        part = part.strip()
        if not part:
            continue

        header_positions = [m.start() for m in _SECTION_HEADER_RE.finditer(part)]
        if not header_positions:
            blocks.append(part)
            continue

        # Anything before the first header is its own block (usually the
        # synopsis or a headline line).
        if header_positions[0] > 0:
            lead = part[: header_positions[0]].strip()
            if lead:
                blocks.append(lead)

        bounds = header_positions + [len(part)]
        for start, end in zip(bounds, bounds[1:]):
            section = part[start:end].strip()
            if section:
                blocks.append(section)

    # Merge undersized blocks into the next one so we don't emit vectors for
    # two-line stubs.
    merged: list[str] = []
    pending = ""
    for block in blocks:
        candidate = f"{pending}\n\n{block}".strip() if pending else block
        if len(candidate) < _MIN_SECTION_CHARS:
            pending = candidate
            continue
        merged.append(candidate)
        pending = ""
    if pending:
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{pending}"
        else:
            merged.append(pending)

    return merged


def _window(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Slide a character window over one oversized block, breaking at whitespace.

    Breaking mid-word would corrupt the first and last token of every chunk,
    so the cut point is walked back to the nearest space when one is close by.
    """
    if len(text) <= chunk_size:
        return [text]

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + step // 2, end)
            if space > start:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_weather_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    normalize: bool = True,
) -> list[str]:
    """
    Full chunking pipeline: normalize -> split on sections -> window the rest.

    This is the function the notebook, the app, and the CLI all call. Change
    it and re-run ingestion; do not reimplement it anywhere else.
    """
    if not text or not text.strip():
        return []

    body = normalize_product_text(text) if normalize else text
    chunks: list[str] = []
    for section in split_sections(body) or [body]:
        chunks.extend(_window(section, chunk_size, overlap))

    # Drop chunks too short to carry meaning (trailing delimiters, sign-offs).
    return [c for c in chunks if len(c) >= 40]


# -------------------------------------------------------------- embedding --

_MODEL_CACHE: dict[str, object] = {}


def load_model(model_name: str | None = None, cache_folder: str | None = None):
    """
    Load a sentence-transformers model, cached per process.

    The Flask app calls this on every /weather/search request; without the
    cache that is a multi-second model load per query. On Databricks the
    HuggingFace cache is pointed at /tmp because the driver's home directory
    is not always writable on serverless compute.
    """
    model_name = model_name or DEFAULT_MODEL
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    cache_folder = cache_folder or os.environ.get("HF_HOME", "/tmp/.cache/huggingface")
    os.environ.setdefault("HF_HOME", cache_folder)
    os.environ.setdefault("HF_HUB_CACHE", cache_folder)
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", cache_folder)

    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s", model_name)
    model = SentenceTransformer(model_name, cache_folder=cache_folder)
    _MODEL_CACHE[model_name] = model
    return model


def embed_texts(
    texts: Sequence[str],
    model_name: str | None = None,
    batch_size: int = 32,
    model=None,
) -> list[list[float]]:
    """Embed a list of strings, returning plain Python float lists."""
    if not texts:
        return []
    model = model or load_model(model_name)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return [list(map(float, v)) for v in vectors]


def embed_query(query: str, model_name: str | None = None) -> list[float]:
    """Embed a single search query using the same path as corpus text."""
    return embed_texts([query], model_name=model_name)[0]


def to_pgvector(vector: Iterable[float]) -> str:
    """
    Render a vector as a pgvector literal: '[0.1,0.2,...]'.

    Passing this string with an explicit `%s::vector` cast lets psycopg2 write
    straight into a VECTOR column. The reference news pipeline inserted into a
    `double precision[]` column and then ran a separate `UPDATE ... ::vector`
    pass; that extra step is not necessary and is skipped here.
    """
    return "[" + ",".join(f"{float(x):.7g}" for x in vector) + "]"
