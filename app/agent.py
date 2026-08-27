import os
from pydantic import BaseModel, Field
import instructor
from litellm import completion
from app.config import DEFAULT_MODEL

# 1. Define the Pydantic schema for structured output
class FulfillmentDecision(BaseModel):
    item_name: str = Field(description="Name of the agricultural item requested")
    quantity_requested: int = Field(description="Quantity requested by the user")
    can_fulfill_from_stock: bool = Field(description="True if stock is sufficient, false otherwise")
    estimated_delivery_days: float = Field(description="Estimated delivery days based on regional hubs")
    recommended_action: str = Field(description="Short operational recommendation for the field agent")

# 2. Patch OpenAI/LiteLLM client with Instructor
# Instructor works seamlessly with LiteLLM-supported providers
client = instructor.from_litellm(completion)

def analyze_supply_request(user_prompt: str) -> FulfillmentDecision:
    """
    Sends a prompt to the LLM and enforces a structured JSON response matching FulfillmentDecision.
    """
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        response_model=FulfillmentDecision,
        messages=[
            {
                "role": "system",
                "content": "You are an AI logistics assistant for an agricultural cooperative. Analyze user inventory requests and return strict structured data."
            },
            {
                "role": "user", 
                "content": user_prompt
            }
        ]
    )
    return response

# Quick test block if run directly
if __name__ == "__main__":
    test_query = "I need 150 bags of Hybrid Maize Seeds delivered to our Pokhara depot urgently."
    print(f"Testing Query: '{test_query}'\n")
    
    try:
        result = analyze_supply_request(test_query)
        print("Structured Output Result:")
        print(result.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error executing agent: {e}")