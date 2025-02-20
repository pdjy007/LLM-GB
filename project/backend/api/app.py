import os
import socket
import streamlit as st
import google.generativeai as genai
from llama_cpp import Llama

# Load Local LLM (Mistral 7B GGUF) with Optimized Parameters
model_path = r"C:\Users\pdjy0\OneDrive\Documents\Final-Year-Project\project\backend\models\mistral-7b-v0.1.Q4_K_M.gguf"

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()

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
            st.warning(f"⚠️ Gemini API failed: {e}. Falling back to Mistral.")

    response = local_llm(
        prompt + "\n\nProvide a concise, one-sentence answer.",
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response["choices"][0]["text"].strip()

# Bias Detection
def detect_bias(sentence: str):
    prompt = f"Does the following sentence contain gender bias? Answer 'Yes' or 'No' and briefly explain why if necessary:\n\nSentence: {sentence}\n\nAnswer:"
    return generate_response(prompt)

# Bias Correction
def correct_bias(sentence: str):
    prompt = f"Rewrite the following sentence in a gender-neutral way. Only provide the corrected version.\n\nSentence: {sentence}\n\nCorrected Sentence:"
    return generate_response(prompt)

# Bias Scoring
def bias_score(sentence: str):
    prompt = f"Rate the gender bias in the following sentence on a scale from 0% (no bias) to 100% (high bias). Provide only the percentage.\n\nSentence: {sentence}\n\nBias Score:"
    return generate_response(prompt)

# Streamlit UI
st.title("📝 Gender Bias Detection & Correction")
st.write("This tool detects and corrects gender bias in text.")

# Real-time input box
sentence = st.text_area("✍️ Enter a sentence to analyze:")

# Only process if the user enters a sentence
if sentence.strip():
    with st.spinner("Processing..."):
        bias_detection = detect_bias(sentence)
        bias_correction = correct_bias(sentence)
        bias_score_value = bias_score(sentence)

    st.subheader("📌 Results")
    st.write(f"**🔍 Bias Detection:** {bias_detection}")
    st.write(f"**✅ Corrected Sentence:** {bias_correction}")
    st.write(f"**📊 Bias Score:** {bias_score_value}")

# Properly Close Llama Model
def cleanup():
    global local_llm
    if local_llm:
        del local_llm
        local_llm = None
        st.success("🟢 Model closed successfully.")

if st.button("🚀 Close Model"):
    cleanup()
