import os
from dotenv import load_dotenv

load_dotenv()

# Define your default model (e.g., GPT-4o-mini or Gemini 1.5 Flash for cost-efficiency)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini/gemini-3.6-flash")  # Change to your preferred model if needed