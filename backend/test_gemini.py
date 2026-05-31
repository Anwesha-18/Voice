import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("Key loaded:", bool(api_key))

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content(
    """
    You are a communication assistant for non-verbal users.

    Convert these detected sign language words into a natural sentence.

    Words:
    help doctor please

    Return only the sentence.
    """
)

print("\nGemini Response:")
print(response.text)