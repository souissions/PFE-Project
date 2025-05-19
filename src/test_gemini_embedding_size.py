import os
import google.generativeai as genai

# Load API key and model name from environment variables or .env
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "gemini-2.0-flash")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in environment.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

sample_text = "asthma"
result = model.embed_content(sample_text)
embedding = result['embedding']
print(f"Gemini embedding size: {len(embedding)}")
