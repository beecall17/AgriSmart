import logging
from collections import OrderedDict

from pydantic import BaseModel, Field
import instructor
from litellm import completion

from app.config import DEFAULT_MODEL
from app.rag import ingest_documents, search_knowledge_base


logger = logging.getLogger(__name__)


# 1. Define the Pydantic schema for structured output
class FulfillmentDecision(BaseModel):
    item_name: str = Field(
        default="Unknown agricultural item",
        description="Name of the agricultural item requested",
    )
    quantity_requested: int = Field(
        default=0,
        ge=0,
        description="Quantity requested by the user",
    )
    can_fulfill_from_stock: bool = Field(
        default=False,
        description="True if stock is sufficient, false otherwise",
    )
    estimated_delivery_days: float = Field(
        default=0.0,
        ge=0,
        description="Estimated delivery days based on regional hubs",
    )
    recommended_action: str = Field(
        default="Contact the regional hub coordinator for manual review.",
        description="Short operational recommendation for the field agent",
    )


# 2. Patch OpenAI/LiteLLM client with Instructor
# Instructor works seamlessly with LiteLLM-supported providers
client = instructor.from_litellm(completion)


# --------------------------------------------------------------------------- #
# Fallback helpers
# --------------------------------------------------------------------------- #
# Context injected when the vector store is unreachable: downstream model calls
# can still run (and degrade gracefully) instead of raising.
RAG_FALLBACK_CONTEXT = (
    "[Knowledge-base retrieval is temporarily unavailable. "
    "Indicate that live stock levels and SOP details could not be verified "
    "and recommend manual confirmation with the nearest regional hub.]"
)


# Maximum number of FulfillmentDecision objects kept in memory.
DECISION_CACHE_MAXSIZE = 128

# Marker used by fallback decisions so they are never cached: a transient
# failure must not be replayed for later (potentially healthy) identical
# requests.
FALLBACK_ITEM_NAME = "Unavailable — system error"

_decision_cache: "OrderedDict[str, FulfillmentDecision]" = OrderedDict()


def _normalize_prompt(user_prompt: str) -> str:
    """Normalize a request for cache-keying (whitespace/case-insensitive)."""
    return " ".join(user_prompt.strip().casefold().split())


def is_cached(user_prompt: str) -> bool:
    """Return True when an identical request is already in the decision cache."""
    if not user_prompt or not user_prompt.strip():
        return False
    return _normalize_prompt(user_prompt) in _decision_cache


def clear_decision_cache() -> None:
    """Drop every cached decision (call after knowledge-base re-ingestion)."""
    _decision_cache.clear()
    logger.info("Cleared the agent decision cache.")


def decision_cache_info() -> dict[str, int]:
    """Return cache usage statistics for the UI / diagnostics."""
    return {"size": len(_decision_cache), "maxsize": DECISION_CACHE_MAXSIZE}


def _fallback_decision(
    user_prompt: str,
    error: Exception | None = None,
) -> FulfillmentDecision:
    """Build a safe, user-friendly FulfillmentDecision when processing fails.

    Used instead of crashing whenever RAG retrieval, LiteLLM, or Instructor
    raises — the UI can still render a clean, structured result.
    """
    detail = f"{type(error).__name__}: {error}" if error else "no error supplied"
    logger.warning(
        "Returning fallback FulfillmentDecision for prompt %r (%s).",
        user_prompt[:80],
        detail,
    )
    return FulfillmentDecision(
        item_name=FALLBACK_ITEM_NAME,
        quantity_requested=0,
        can_fulfill_from_stock=False,
        estimated_delivery_days=0.0,
        recommended_action=(
            "Our supply-chain service is temporarily unavailable. Please "
            "contact the regional hub coordinator directly or retry in a few "
            "minutes."
        ),
    )


# --------------------------------------------------------------------------- #
# 3. RAG context retrieval
# --------------------------------------------------------------------------- #
def _ensure_knowledge_ingested() -> None:
    """Make sure the vector store is populated before retrieving.

    Cheap and idempotent: if the collection already has records, ingest just
    upserts the same deterministic IDs and leaves the store unchanged.
    """
    ingest_documents()


def build_rag_context(
    user_prompt: str,
    n_results: int = 4,
    results: list[dict] | None = None,
) -> str:
    """Retrieve relevant inventory/SOP chunks and format them for the LLM.

    Pass pre-retrieved chunks via ``results`` (the format returned by
    ``search_knowledge_base``) to reuse a lookup already performed by the
    caller instead of hitting the vector store a second time.
    """
    if results is None:
        try:
            _ensure_knowledge_ingested()
            results = search_knowledge_base(user_prompt, n_results=n_results)
        except Exception as exc:
            logger.warning(
                "RAG retrieval failed for prompt %r: %s",
                user_prompt[:80],
                exc,
            )
            return RAG_FALLBACK_CONTEXT

    if not results:
        return "[No relevant knowledge base context was found for this query.]"

    lines = []
    for rank, item in enumerate(results, start=1):
        source = item["metadata"].get("source", "unknown")
        section = item["metadata"].get("section", "")
        header = f"Context {rank} (source: {source})"
        if section:
            header += f", section: {section}"
        lines.append(f"{header}\n{item['document']}")
    return "\n\n".join(lines)


def analyze_supply_request(
    user_prompt: str,
    rag_context: str | None = None,
) -> FulfillmentDecision:
    """
    Retrieves relevant context from the AgriSmart knowledge base (inventory CSV
    + logistics SOPs), then sends it — alongside the user's supply request — to
    the LLM through Instructor, which enforces a structured FulfillmentDecision.

    ``rag_context`` is optional: supply a pre-formatted context string (e.g.
    from ``build_rag_context``) to reuse retrieval already performed by the
    caller instead of triggering a second knowledge-base lookup.

    Hash-based response cache: identical requests (compared case-insensitively
    and whitespace-insensitively, per ``is_cached``) are served instantly from
    the in-process cache, skipping the LLM call entirely. Call
    ``clear_decision_cache()`` after re-ingesting the knowledge base.

    Failure-safe: any error during RAG retrieval, LiteLLM, or Instructor
    validation yields a graceful ``FulfillmentDecision`` fallback object instead
    of raising an exception back to the caller. Fallbacks are never cached.
    """
    if not user_prompt or not user_prompt.strip():
        return _fallback_decision(user_prompt, ValueError("Empty supply request."))

    key = _normalize_prompt(user_prompt)
    cached = _decision_cache.get(key)
    if cached is not None:
        _decision_cache.move_to_end(key)
        logger.info(
            "Decision cache hit for request %r — returning cached result.",
            user_prompt[:80],
        )
        return cached

    if rag_context is None:
        rag_context = build_rag_context(user_prompt)

    system_prompt = (
        "You are an AI logistics assistant for an agricultural cooperative. "
        "Analyze user inventory requests and return strict structured data. "
        "Ground every field on the knowledge-base context provided: use the "
        "inventory records to determine stock availability and the logistics "
        "SOP to estimate transit time and any dispatch/compliance rules. "
        "If the requested item or stock is not present in the context, set "
        "can_fulfill_from_stock accordingly and recommend a sensible action."
    )

    user_message = (
        f"USER REQUEST:\n{user_prompt}\n\n"
        f"KNOWLEDGE BASE CONTEXT (recently retrieved, use as ground truth):\n{rag_context}"
    )

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            response_model=FulfillmentDecision,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
    except Exception as exc:
        logger.error(
            "LiteLLM/Instructor call failed for prompt %r: %s — returning fallback.",
            user_prompt[:80],
            exc,
            exc_info=True,
        )
        return _fallback_decision(user_prompt, exc)

    if not isinstance(response, FulfillmentDecision):
        logger.warning(
            "Instructor returned unexpected type %r — returning fallback.",
            type(response).__name__,
        )
        return _fallback_decision(user_prompt)

    # Only genuine model decisions are cached; fallback decisions (e.g. from a
    # transient provider failure) are deliberately skipped so the next identical
    # query retries the live API instead of replaying the error response.
    if response.item_name != FALLBACK_ITEM_NAME:
        _decision_cache[key] = response
        _decision_cache.move_to_end(key)
        while len(_decision_cache) > DECISION_CACHE_MAXSIZE:
            _decision_cache.popitem(last=False)

    return response


# Quick test block if run directly
if __name__ == "__main__":
    test_query = (
        "A field agent in Pokhara needs 50 bags of Hybrid Maize Seeds urgently. "
        "Check stock and transit rules."
    )
    print(f"Testing Query: '{test_query}'\n")

    print("[RAG] Retrieving relevant knowledge-base context...")
    print(build_rag_context(test_query) + "\n")

    try:
        result = analyze_supply_request(test_query)
        print("Structured Output Result:")
        print(result.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error executing agent: {e}")
