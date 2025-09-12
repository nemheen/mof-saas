# gemini_wrapper.py
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")  # or gemini-pro-vision, etc.

def ask_gemini_rag(query: str) -> str:
    response = model.generate_content(query)
    return response.text
