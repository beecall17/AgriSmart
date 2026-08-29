"""AgriSmart Streamlit frontend.

Wires the RAG knowledge base (``app.rag``) into the structured agent
(``app.agent.analyze_supply_request``) and renders the resulting Pydantic
``FulfillmentDecision`` fields for field agents.

Run from the project root:

    python -m streamlit run app/main.py
"""

from pathlib import Path

import streamlit as st

from app.agent import analyze_supply_request, build_rag_context
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


# Ensure the local CSV/SOP files are surfaced in the vector store before use.
_ensure_ingested()


# --------------------------------------------------------------------------- #
# Request pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(user_query: str) -> None:
    """RAG-retrieve relevant context, then ask the structured agent for a
    FulfillmentDecision, and store the outcome in the conversation history.
    """
    # 1. Retrieve relevant context from the local CSV/SOP knowledge base.
    results: list[dict] = []
    with st.spinner("Searching the knowledge base (inventory + SOP)..."):
        try:
            results = search_knowledge_base(user_query, n_results=N_RESULTS)
        except Exception as exc:
            st.warning(f"⚠️ RAG retrieval failed ({exc}) — continuing without context.")

    # 2. Format the retrieved chunks into the context block for the agent.
    rag_context = build_rag_context(
        user_query, n_results=N_RESULTS, results=results
    )

    # 3. Pass query + context to the structured agent and validate the response.
    decision_data = None
    error = None
    with st.spinner("Analyzing request with the structured agent..."):
        try:
            decision = analyze_supply_request(user_query, rag_context=rag_context)
            decision_data = decision.model_dump()
        except Exception as exc:
            error = str(exc)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "decision": decision_data,
            "results": _simplify_results(results),
            "error": error,
        }
    )


def render_decision(message: dict) -> None:
    """Render a stored assistant message (Pydantic fields + RAG context)."""
    decision = message.get("decision")

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
            st.caption("No knowledge-base chunks were retrieved for this query.")
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

    st.write(f"**Active model:** `{DEFAULT_MODEL}`")

    st.subheader("📁 Active Data Sources")
    for path in (INVENTORY_CSV_PATH, LOGISTICS_SOP_PATH):
        if path.exists():
            st.write(f"✅ `{path.name}` — {path.stat().st_size:,} bytes")
        else:
            st.write(f"❌ `{path.name}` — missing")

    source_stats = _collection_stats()
    if source_stats:
        for source, count in sorted(source_stats.items()):
            st.write(f"&nbsp;&nbsp;&nbsp;• `{source}` — {count} chunk(s)")

    st.divider()

    if st.button("♻️ Re-ingest knowledge base", use_container_width=True):
        with st.spinner("Re-ingesting enterprise files..."):
            ingest_documents()
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
            st.session_state.messages.append({"role": "user", "content": query})
            run_pipeline(query)
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
    st.session_state.messages.append({"role": "user", "content": prompt})
    run_pipeline(prompt)
    st.rerun()