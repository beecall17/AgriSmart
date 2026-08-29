"""AgriSmart Streamlit frontend.

Wires the RAG knowledge base (``app.rag``) into the structured agent
(``app.agent.analyze_supply_request``) and renders the resulting Pydantic
``FulfillmentDecision`` fields for field agents.

Run from the project root:

    python -m streamlit run app/main.py
"""

import logging
from pathlib import Path

import pandas as pd
import streamlit as st
import sys
from pathlib import Path

# Add the root directory (parent of 'app') to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.agent import (
    analyze_supply_request,
    build_rag_context,
    clear_decision_cache,
    decision_cache_info,
    is_cached,
)
from app.config import DEFAULT_MODEL
from app.rag import (
    COLLECTION_NAME,
    INVENTORY_CSV_PATH,
    LOGISTICS_SOP_PATH,
    get_collection,
    ingest_documents,
    search_knowledge_base,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

N_RESULTS = 4  # number of knowledge-base chunks retrieved per query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AgriSmart | Supply Chain Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Session bootstrap (must run before any widget reads session_state)
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []
if "form_error" not in st.session_state:
    st.session_state.form_error = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _collection_stats() -> dict[str, int]:
    """Return per-source chunk counts from the active Chroma collection."""
    stats: dict[str, int] = {}
    try:
        metadatas = get_collection().get(include=["metadatas"])["metadatas"]
        for meta in metadatas:
            source = (meta or {}).get("source", "unknown")
            stats[source] = stats.get(source, 0) + 1
    except Exception:
        pass
    return stats


def _ensure_ingested() -> None:
    """Ingest the enterprise files into Chroma once per browser session."""
    if st.session_state.get("agri_ingested"):
        return
    ingest_documents()
    st.session_state["agri_ingested"] = True


def _simplify_results(results: list[dict]) -> list[dict]:
    """Keep only the fields the UI needs from each RAG hit."""
    simplified: list[dict] = []
    for hit in results:
        meta = hit.get("metadata") or {}
        simplified.append(
            {
                "id": hit.get("id", ""),
                "source": meta.get("source", "unknown"),
                "section": meta.get("section", ""),
                "product_id": meta.get("product_id", ""),
                "warehouse_location": meta.get("warehouse_location", ""),
                "stock_quantity": meta.get("stock_quantity", ""),
                "document": hit.get("document", ""),
                "distance": hit.get("distance", 0.0),
            }
        )
    return simplified


# --------------------------------------------------------------------------- #
# Cached data loaders (static enterprise files are read once, not per rerun)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_inventory_frame(csv_path: str) -> pd.DataFrame:
    """Load the inventory CSV into a DataFrame, memoized across reruns.

    The file path (a str) is the cache key; results are pickled by Streamlit so
    repeated reruns and identical loads skip disk entirely.
    """
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
def load_sop_text(md_path: str) -> str:
    """Load the logistics SOP markdown as text, memoized across reruns."""
    return Path(md_path).read_text(encoding="utf-8")


# Ensure the local CSV/SOP files are surfaced in the vector store before use.
_ensure_ingested()


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def validate_query(query: str | None) -> tuple[bool, str]:
    """Validate a field-agent supply request before it enters the pipeline.

    Returns ``(valid, message)``; when ``valid`` is False, ``message`` is a
    user-friendly explanation shown as a warning banner.
    """
    if query is None or not query.strip():
        return False, (
            "Your request appears to be empty. Please describe the item and "
            'quantity you need (e.g., "100 bags of Urea for Pokhara").'
        )

    query_clean = query.strip()

    if "\x00" in query_clean or any(ord(ch) < 32 for ch in query_clean):
        return False, (
            "Your request contains unsupported characters. Please re-enter it "
            "using plain text (letters, numbers, and basic punctuation)."
        )

    if len(query_clean) < 3:
        return False, (
            "Your request is too short to process. Please include the item "
            'name and quantity (e.g., "100 bags of Urea").'
        )

    if len(query_clean) > 500:
        return False, (
            "Your request is too long to process in one message. "
            "Please keep it under 500 characters."
        )

    if not any(ch.isalnum() for ch in query_clean):
        return False, (
            "Your request does not contain any letters or numbers. Please "
            "describe the supply item you need using plain text."
        )

    return True, ""


def submit_query(query: str) -> None:
    """Validate a user request; run the pipeline or store a warning banner."""
    valid, message = validate_query(query)
    if not valid:
        st.session_state.form_error = message
        logger.info("Rejected invalid supply request: %s", message)
        return
    st.session_state.form_error = None
    st.session_state.messages.append({"role": "user", "content": query})
    run_pipeline(query)


# --------------------------------------------------------------------------- #
# Request pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(user_query: str) -> None:
    """RAG-retrieve relevant context, then ask the structured agent for a
    FulfillmentDecision, and store the outcome in the conversation history.

    Failure-safe at every layer: RAG search, context assembly, and the
    structured-agent call are individually guarded so the app never crashes.

    Performance: repeated identical requests hit the agent's response cache
    (``is_cached``), skipping both the vector search and the LLM call.
    """
    logger.info("Starting pipeline for request: %r", user_query)

    from_cache = is_cached(user_query)
    if from_cache:
        logger.info(
            "Request %r found in decision cache — skipping RAG/LLM work.",
            user_query[:80],
        )

    # 1. Retrieve relevant context from the local CSV/SOP knowledge base.
    results: list[dict] = []
    if not from_cache:
        try:
            with st.spinner("🔍 Searching the knowledge base (inventory + SOP)..."):
                results = search_knowledge_base(user_query, n_results=N_RESULTS)
            logger.info("Retrieved %d chunks from the knowledge base.", len(results))
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            st.warning("⚠️ Knowledge-base search hit a snag — continuing with best-effort context.")
            results = []

    # 2. Format the retrieved chunks into the context block for the agent.
    rag_context = None
    if not from_cache:
        try:
            rag_context = build_rag_context(
                user_query, n_results=N_RESULTS, results=results
            )
        except Exception as exc:
            logger.warning("Could not assemble RAG context: %s", exc)
            rag_context = "[Knowledge-base context is temporarily unavailable.]"

    # 3. Pass query + context to the structured agent and validate the response.
    decision_data = None
    error = None
    try:
        with st.spinner("🤖 Analyzing request with the structured agent..."):
            decision = analyze_supply_request(user_query, rag_context=rag_context)
            decision_data = decision.model_dump()
        logger.info("Structured fulfillment decision produced.")
    except Exception as exc:
        logger.error("Agent pipeline failed unexpectedly: %s", exc, exc_info=True)
        error = str(exc)

    # 4. Safely store the result (and any diagnostic error) in history.
    try:
        simplified = _simplify_results(results)
    except Exception as exc:
        logger.warning("Could not simplify retrieval results for display: %s", exc)
        simplified = []

    st.session_state.messages.append(
        {
            "role": "assistant",
            "decision": decision_data,
            "results": simplified,
            "error": error,
            "from_cache": from_cache,
        }
    )
    logger.info("Pipeline finished; assistant message stored.")


def render_decision(message: dict) -> None:
    """Render a stored assistant message (Pydantic fields + RAG context)."""
    decision = message.get("decision")

    if message.get("from_cache"):
        st.caption("⚡ **Served from response cache** — identical request was answered before (no new LLM call).")

    if decision is None:
        st.error("❌ The agent could not produce a structured decision.")
        if message.get("error"):
            st.code(message["error"], language="text")
    else:
        metrics_row = st.columns(3)
        metrics_row[0].metric("📦 Item", decision["item_name"])
        metrics_row[1].metric("🔢 Quantity requested",
                              f"{decision['quantity_requested']:,}")
        metrics_row[2].metric(
            "⏱️ Est. delivery", f"{decision['estimated_delivery_days']:.1f} days"
        )

        if decision["can_fulfill_from_stock"]:
            st.success("✅ **Fulfillment status:** Can fulfill from stock")
        else:
            st.error("❌ **Fulfillment status:** Cannot fulfill from current stock")

        st.info(f"💡 **Recommended action:** {decision['recommended_action']}")

        with st.expander("🗒️ Structured decision (JSON)"):
            st.json(decision)

    with st.expander("🔍 Retrieved context (RAG)", expanded=False):
        results = message.get("results") or []
        if not results:
            st.caption(
                "No knowledge-base chunks retrieved for this query. "
                "⚠️ This may be a cached response served without re-querying."
            )
        for i, hit in enumerate(results, start=1):
            section = f" · {hit['section']}" if hit.get("section") else ""
            extra = ""
            if hit.get("product_id"):
                extra = (
                    f" · `{hit['product_id']}` @ {hit['warehouse_location']} "
                    f"({hit['stock_quantity']} in stock)"
                )
            st.markdown(
                f"**{i}.** `{hit['source']}`{section}{extra} — "
                f"_distance {hit['distance']:.4f}_"
            )
            st.write(hit["document"])


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🌾 AgriSmart")
    st.caption("AI Supply-Chain & Inventory Coordinator")

    st.subheader("🖥️ System Status")
    try:
        chunk_count = get_collection().count()
        status_icon = "🟢" if chunk_count > 0 else "🟡"
        st.metric("Vector DB chunks", f"{chunk_count}")
        st.caption(f"{status_icon} Collection: `{COLLECTION_NAME}`")
    except Exception as exc:
        st.metric("Vector DB chunks", "unavailable")
        st.caption(f"⚠️ Vector DB error: {exc}")

    cache_info = decision_cache_info()
    st.metric(
        "🧠 Cached responses",
        f"{cache_info['size']}/{cache_info['maxsize']}",
    )
    st.caption("Identical requests are served instantly without re-hitting the LLM.")

    st.write(f"**Active model:** `{DEFAULT_MODEL}`")

    st.subheader("📁 Active Data Sources")
    for path in (INVENTORY_CSV_PATH, LOGISTICS_SOP_PATH):
        if path.exists():
            st.write(f"✅ `{path.name}` — {path.stat().st_size:,} bytes")
        else:
            st.write(f"❌ `{path.name}` — missing")

    try:
        inv_df = load_inventory_frame(str(INVENTORY_CSV_PATH))
        sop_text = load_sop_text(str(LOGISTICS_SOP_PATH))
        st.write(
            f"&nbsp;&nbsp;&nbsp;• {len(inv_df)} products · "
            f"{len(inv_df.columns)} columns (cached)"
        )
        st.write(f"&nbsp;&nbsp;&nbsp;• {len(sop_text.splitlines())} SOP lines (cached)")
    except Exception as exc:
        st.caption(f"⚠️ Could not load source preview: {exc}")

    source_stats = _collection_stats()
    if source_stats:
        for source, count in sorted(source_stats.items()):
            st.write(f"&nbsp;&nbsp;&nbsp;• `{source}` — {count} chunk(s)")

    with st.expander("📋 Data preview (cached)"):
        try:
            inv_df = load_inventory_frame(str(INVENTORY_CSV_PATH))
            st.dataframe(inv_df.head(5), use_container_width=True)
            sop_text = load_sop_text(str(LOGISTICS_SOP_PATH))
            st.caption(f"[SOP] {len(sop_text.splitlines())} lines loaded")
        except Exception as exc:
            st.caption(f"⚠️ Preview unavailable: {exc}")

    st.divider()

    if st.button("♻️ Re-ingest knowledge base", use_container_width=True):
        with st.spinner("Re-ingesting enterprise files..."):
            ingest_documents()
            # Invalidate every cache that depends on knowledge-base contents.
            clear_decision_cache()
            search_knowledge_base.cache_clear()
            st.cache_data.clear()
            st.session_state["agri_ingested"] = True
        st.success("Knowledge base refreshed.")
        st.rerun()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "📘 Sources: `data/inventory_db.csv` and `data/logistics_sop.md`"
    )
# --------------------------------------------------------------------------- #
# Main console
# --------------------------------------------------------------------------- #
st.title("🌾 AgriSmart — Supply Chain & Inventory Coordinator")
st.caption(
    "Ask a field-agent supply request: stock availability, delivery timelines, "
    "and dispatch SOPs are answered with RAG-grounded context."
)

if st.session_state.get("form_error"):
    st.warning(f"⚠️ {st.session_state['form_error']}")

st.markdown("**Try an example:**")
example_cols = st.columns(3)
examples = [
    ("🌽 Urea for Pokhara",
     "I need 100 bags of Urea Fertilizer delivered to the Pokhara depot."),
    ("🌱 Maize seeds to Kathmandu",
     "I need 150 bags of Hybrid Maize Seeds for the Kathmandu depot urgently."),
    ("🚚 Pesticide dispatch rules",
     "What are the transport and safety rules for dispatching pesticides?"),
]
for col, (label, query) in zip(example_cols, examples):
    with col:
        if st.button(label, use_container_width=True):
            submit_query(query)
            st.rerun()

st.divider()

st.subheader("💬 Request Console")

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧑‍🌾"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            render_decision(message)


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
prompt = st.chat_input(
    'Type a supply request, e.g. "I need 100 bags of Urea for the Pokhara depot"'
)
if prompt:
    submit_query(prompt)
    st.rerun()