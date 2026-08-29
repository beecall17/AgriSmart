import logging

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
        item_name="Unavailable — system error",
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

    Failure-safe: any error during RAG retrieval, LiteLLM, or Instructor
    validation yields a graceful ``FulfillmentDecision`` fallback object instead
    of raising an exception back to the caller.
    """
    if not user_prompt or not user_prompt.strip():
        return _fallback_decision(user_prompt, ValueError("Empty supply request."))

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
