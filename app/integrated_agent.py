from app.rag import search_knowledge_base
from app.local_llm import generate_local_response

def answer_supply_request_locally(user_query: str) -> str:
    """
    1. Searches the RAG vector database for relevant inventory and logistics context.
    2. Builds an augmented prompt containing the retrieved facts.
    3. Passes the prompt to the local Hugging Face model for generation.
    """
    print(f"Searching knowledge base for query: '{user_query}'...")
    retrieved_docs = search_knowledge_base(user_query, n_results=3)
    
    # Format retrieved chunks into a context block
    context_str = "\n\n---\n\n".join(doc.get("text", str(doc)) for doc in retrieved_docs)
    
    # Construct the RAG-augmented prompt
    augmented_prompt = f"""You are the AgriSmart AI Supply Chain Coordinator. Use ONLY the provided enterprise context below to answer the question accurately. If the answer is not in the context, state that you cannot find the information in the shared drive.

[Enterprise Context / Shared Drive Files]
{context_str}

[User Request]
{user_query}

Provide a concise, professional operational response:"""

    print("Generating local response using retrieved context...")
    response = generate_local_response(augmented_prompt)
    return response

if __name__ == "__main__":
    # Test query combining inventory and logistics rules
    test_query = "What is the stock quantity of Hybrid Maize Seeds in Kathmandu, and what is the standard transit time to Pokhara?"
    
    final_output = answer_supply_request_locally(test_query)
    print("\n--- Final Integrated RAG Output ---")
    print(final_output)