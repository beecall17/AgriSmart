from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Choose a lightweight model that runs smoothly locally
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print(f"Loading local model '{MODEL_NAME}' via Hugging Face Transformers...")

# Initialize tokenizer and model for local inference (using CPU/auto device mapping)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,  # Use float32 for stable CPU execution if needed
    device_map="auto"
)

# Create the text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

def generate_local_response(prompt: str, max_new_tokens: int = 256) -> str:
    """
    Generates text locally using the Hugging Face pipeline.
    """
    messages = [
        {"role": "system", "content": "You are an AI logistics assistant for AgriSmart."},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template if available, or format manually
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    outputs = generator(
        formatted_prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract only the newly generated text response
    generated_text = outputs[0]["generated_text"]
    return generated_text

if __name__ == "__main__":
    test_prompt = "Check inventory for 50 bags of Hybrid Maize Seeds in Pokhara."
    print(f"\nTesting Local Pipeline with prompt: '{test_prompt}'\n")
    response = generate_local_response(test_prompt)
    print(response)