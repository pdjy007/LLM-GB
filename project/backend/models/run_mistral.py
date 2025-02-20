import os
import socket
import google.generativeai as genai
from llama_cpp import Llama
import sqlite3

# Load Local LLM (Mistral 7B GGUF) with Optimized Parameters
model_path = r"C:\Users\pdjy0\OneDrive\Documents\Final-Year-Project\project\backend\app\models\mistral-7b-v0.1.Q4_K_M.gguf"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

local_llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
    f16_kv=True,
    use_mlock=True
)

# Function to Check Internet Connectivity
def check_internet():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

# Configure Gemini API (for Online Mode if Internet is Available)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
online_mode = bool(GOOGLE_API_KEY) and check_internet()

if online_mode:
    genai.configure(api_key=GOOGLE_API_KEY)

def generate_response(prompt: str):
    """Chooses Gemini API (if online) or Local Mistral GGUF for response generation."""
    if online_mode:
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            if response.text:
                return response.text
        except Exception as e:
            print(f"Gemini API failed: {e}. Falling back to Mistral.")

    print("Using Mistral model for response.")
    return local_llm(prompt, max_tokens=100)["choices"][0]["text"]

def detect_bias(sentence: str):
    prompt = f"Analyze the following sentence for gender bias:\n\nSentence: {sentence}\n\nBias Analysis: "
    return generate_response(prompt)

def correct_bias(sentence: str):
    prompt = f"Rewrite the following sentence in a gender-neutral way:\n\nSentence: {sentence}\n\nCorrected Sentence: "
    return generate_response(prompt)

def bias_score(sentence: str):
    prompt = f"Rate the gender bias in the following sentence on a scale from 0% (no bias) to 100% (high bias):\n\nSentence: {sentence}\n\nBias Score: "
    return generate_response(prompt)

# Testing the Model
if __name__ == "__main__":
    test_sentence = "He was a great doctor, but she was only a nurse."

    print("Online Mode:", online_mode)
    print("\nDetect Bias:", detect_bias(test_sentence))
    print("\nCorrect Bias:", correct_bias(test_sentence))
    print("\nBias Score:", bias_score(test_sentence))

    # ✅ Explicitly close the model before exiting
    local_llm = None  # Helps with garbage collection
    print("\nModel closed successfully.")
