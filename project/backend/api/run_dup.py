import os
import socket
import google.generativeai as genai
from llama_cpp import Llama
import sqlite3

# Load Local LLM (Mistral 7B GGUF) with Optimized Parameters
model_path = r"C:\Users\pdjy0\OneDrive\Documents\Final-Year-Project\project\backend\models\mistral-7b-v0.1.Q4_K_M.gguf"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

local_llm = Llama(
    model_path=model_path,
    n_ctx=2048,  # Context window
    n_threads=8,  # Use 8 threads for parallelism
    f16_kv=True,  # Use FP16 key-value cache for faster inference
    use_mlock=True  # Prevent swapping if memory is available
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

# Generate Response Function (Gemini + Mistral Fallback)
def generate_response(prompt: str, max_tokens: int = 100, temperature: float = 0.3):
    """
    Chooses Gemini API (if online) or Local Mistral GGUF for response generation.
    Ensures concise, controlled output using temperature and max_tokens.
    """
    if online_mode:
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )
            if response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Gemini API failed: {e}. Falling back to Mistral.")

    print("Using Mistral model for response.")
    response = local_llm(
        prompt + "\n\nProvide a concise, one-sentence answer.",
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response["choices"][0]["text"].strip()

# Bias Detection
def detect_bias(sentence: str):
    """
    Analyzes a sentence for gender bias and returns a concise analysis.
    """
    prompt = f"Does the following sentence contain gender bias? Answer 'Yes' or 'No' and briefly explain why if necessary:\n\nSentence: {sentence}\n\nAnswer:"
    return generate_response(prompt)

# Bias Correction
def correct_bias(sentence: str):
    """
    Provides a gender-neutral correction of a given sentence.
    """
    prompt = f"Rewrite the following sentence in a gender-neutral way. Only provide the corrected version.\n\nSentence: {sentence}\n\nCorrected Sentence:"
    return generate_response(prompt)

# Bias Scoring
def bias_score(sentence: str):
    """
    Rates gender bias on a scale from 0% (no bias) to 100% (high bias).
    """
    prompt = f"Rate the gender bias in the following sentence on a scale from 0% (no bias) to 100% (high bias). Provide only the percentage.\n\nSentence: {sentence}\n\nBias Score:"
    return generate_response(prompt)

# Properly Close Llama Model
def cleanup():
    global local_llm
    if local_llm:
        del local_llm
        local_llm = None
        print("🟢 Model closed successfully.")

# Main Execution (Test Cases)
if __name__ == "__main__":
    test_sentence = "He was a good doctor and she was a good nurse."

    print("Online Mode:", online_mode)

    # Perform Bias Detection, Correction, and Scoring
    print("\n🔍 Detect Bias:", detect_bias(test_sentence))
    print("\n✅ Correct Bias:", correct_bias(test_sentence))
    print("\n📊 Bias Score:", bias_score(test_sentence))

    # Properly Close Llama Model
    cleanup()