"""Retrieval-Augmented Generation (RAG) pipeline for AgriSmart.

Ingests the enterprise knowledge files (inventory CSV export + logistics SOP
markdown) into a local, persistent Chroma vector store and exposes a retrieval
function. This grounds LLM responses in real operational data instead of relying
only on conversational chat.
"""

import csv
import functools
import os
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Local, persistent Chroma storage directory (a "chroma_db" folder at the
# project root). The path can be overridden via the CHROMA_DB_DIR env var, and
# is resolved relative to the project root so the store is stable regardless of
# the current working directory.
CHROMA_DB_DIR = Path(os.getenv("CHROMA_DB_DIR", "./chroma_db"))
if not CHROMA_DB_DIR.is_absolute():
    CHROMA_DB_DIR = PROJECT_ROOT / CHROMA_DB_DIR

DATA_DIR = PROJECT_ROOT / "data"
INVENTORY_CSV_PATH = DATA_DIR / "inventory_db.csv"
LOGISTICS_SOP_PATH = DATA_DIR / "logistics_sop.md"

COLLECTION_NAME = "agri_knowledge_base"
CHUNK_SIZE = 500     # approximate max characters for a single chunk
CHUNK_OVERLAP = 100   # characters of context carried from the previous chunk


# --------------------------------------------------------------------------- #
# Chroma client & collection helpers
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    """Return a process-wide persistent Chroma client bound to ./chroma_db."""
    return chromadb.PersistentClient(path=str(CHROMA_DB_DIR))


@functools.lru_cache(maxsize=1)
def get_collection() -> chromadb.Collection:
    """Return (creating if needed) the "agri_knowledge_base" Chroma collection.

    Cached per-process so the collection is initialized exactly once; upstream
    upserts (``ingest_documents``) update the same live collection object, so
    re-ingestion remains correct without recreating the wrapper.
    """
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


# --------------------------------------------------------------------------- #
# Loaders / chunking
# --------------------------------------------------------------------------- #
def load_inventory_records() -> list[dict]:
    """Read the inventory CSV and turn each row into a natural-language record."""
    records: list[dict] = []
    with INVENTORY_CSV_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            product_id = row["product_id"]
            document = (
                f"Product {product_id}: {row['item_name']} "
                f"(category: {row['category']}). "
                f"In stock: {row['stock_quantity']} units. "
                f"Warehouse: {row['warehouse_location']}. "
                f"Unit price: NPR {row['unit_price_npr']}."
            )
            records.append(
                {
                    "id": f"inventory-{product_id}",
                    "document": document,
                    "metadata": {
                        "source": "inventory_db.csv",
                        "product_id": product_id,
                        "item_name": row["item_name"],
                        "category": row["category"],
                        "warehouse_location": row["warehouse_location"],
                        "stock_quantity": int(row["stock_quantity"] or 0),
                        "unit_price_npr": float(row["unit_price_npr"] or 0),
                    },
                }
            )
    return records


def chunk_markdown(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split markdown text into section-aware, overlapping chunks.

    Paragraphs are kept intact unless a single one alone exceeds the chunk size
    (then it is split on sentence boundaries). Consecutive small paragraphs are
    grouped until the chunk fills, and the tail of the previous chunk is carried
    forward so retrieval can still see surrounding context.
    """
    def split_paragraph(paragraph: str) -> list[str]:
        if len(paragraph) <= chunk_size:
            return [paragraph]
        sentences = re.split(r"(?<=[.?!])\s+", paragraph)
        pieces, cur, cur_len = [], [], 0
        for sentence in sentences:
            if cur and cur_len + len(sentence) > chunk_size:
                pieces.append(" ".join(cur))
                cur, cur_len = [], 0
            cur.append(sentence)
            cur_len += len(sentence)
        if cur:
            pieces.append(" ".join(cur))
        return pieces

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0
    tail_overlap = ""  # context carried from the previous emitted chunk

    def flush() -> None:
        nonlocal buffer, buffer_len, tail_overlap
        if not buffer:
            return
        body = "\n".join(buffer)
        chunks.append(f"{tail_overlap}\n{body}" if tail_overlap else body)
        # Carry the last couple of lines of this chunk into the next one.
        tail_overlap = "\n".join(buffer[-2:])
        buffer = []
        buffer_len = 0

    for paragraph in paragraphs:
        for piece in split_paragraph(paragraph):
            if buffer and buffer_len + len(piece) > chunk_size:
                flush()
            buffer.append(piece)
            buffer_len += len(piece)
    flush()

    return chunks


def load_sop_records() -> list[dict]:
    """Load the logistics SOP and split it into heading-aware chunks."""
    text = LOGISTICS_SOP_PATH.read_text(encoding="utf-8")
    chunks: list[dict] = []
    heading = ""
    buffer: list[str] = []
    buffer_len = 0
    chunk_no = 0
    overlap_tail = ""

    def emit() -> None:
        nonlocal buffer, buffer_len, chunk_no, overlap_tail
        if not buffer:
            return
        body = "\n".join(buffer)
        document = body
        if overlap_tail:
            document = f"{overlap_tail}\n{document}"
        if heading:
            document = f"{heading}\n{document}"
        overlap_tail = "\n".join(buffer[-2:])
        chunks.append(
            {
                "id": f"sop-{chunk_no}",
                "document": document,
                "metadata": {"source": "logistics_sop.md", "section": heading},
            }
        )
        buffer = []
        buffer_len = 0
        chunk_no += 1

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Headings start a new chunk and become the section label.
        if re.match(r"^#{1,6}\s", line):
            emit()
            heading = line
            continue
        if buffer and buffer_len + len(line) > CHUNK_SIZE:
            emit()
        buffer.append(line)
        buffer_len += len(line) + 1

    emit()
    return chunks


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def ingest_documents() -> dict:
    """Ingest the inventory CSV and SOP markdown into the Chroma collection.

    The operation is idempotent: deterministic IDs mean re-running it simply
    upserts the records without creating duplicates.
    """
    records = load_inventory_records() + load_sop_records()
    collection = get_collection()

    if not records:
        return {"added": 0, "sources": [], "total_in_collection": collection.count()}

    collection.upsert(
        ids=[r["id"] for r in records],
        documents=[r["document"] for r in records],
        metadatas=[r["metadata"] for r in records],
    )

    return {
        "added": len(records),
        "sources": sorted({r["metadata"]["source"] for r in records}),
        "total_in_collection": collection.count(),
    }


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=256)
def search_knowledge_base(query: str, n_results: int = 3) -> list[dict]:
    """Query the vector store and return the most relevant text chunks.

    Each result is a dict with the chunk text, its metadata, and the distance
    score (lower is more similar under cosine).

    In-process response cache: identical ``(query, n_results)`` pairs are served
    straight from memory without re-touching Chroma or re-embedding. Call
    ``search_knowledge_base.cache_clear()`` after re-ingesting documents so
    stale results are not served.
    """
    collection = get_collection()
    result = collection.query(query_texts=[query], n_results=n_results)

    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    relevant = []
    for chunk_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        if document is None:
            continue
        relevant.append(
            {
                "id": chunk_id,
                "document": document,
                "metadata": metadata or {},
                "distance": distance,
            }
        )
    return relevant


# --------------------------------------------------------------------------- #
# Test block
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 72)
    print("AgriSmart RAG pipeline self-test")
    print("=" * 72)

    print("\n[1/2] Ingesting enterprise documents into Chroma...")
    ingest_result = ingest_documents()
    print(
        f"  Added    : {ingest_result['added']} chunks "
        f"(from {','.join(ingest_result['sources'])})"
    )
    print(f"  In store : {ingest_result['total_in_collection']} chunks")

    test_query = "What are the shipping rules for pesticides?"
    print(f"\n[2/2] Retrieving top results for:\n      '{test_query}'\n")
    results = search_knowledge_base(test_query, n_results=3)
    for i, item in enumerate(results, start=1):
        print(f"--- result {i} (id={item['id']}, distance={item['distance']:.4f}) ---")
        print(item["document"])
        print()
