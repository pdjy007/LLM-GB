import os
import socket
import re
import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
from llama_cpp import Llama

# 🔹 Load Local LLM (DeepSeek-R1-Distill-Llama-8B GGUF)
MODEL_PATH = r"C:\Users\pdjy0\Downloads\DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

local_llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    f16_kv=True,
    use_mlock=True
)

# 🔹 Function to Check Internet Connectivity
def check_internet():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

# 🔹 Configure Gemini API (if online)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
online_mode = bool(GOOGLE_API_KEY) and check_internet()

if online_mode:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception:
        online_mode = False  # Disable Gemini if API setup fails

# 🔹 Generate Response Function (DeepSeek + Gemini Fallback)
def generate_response(prompt: str, max_tokens: int = 100, temperature: float = 0.3):
    """
    Uses Gemini API (if online) or DeepSeek-R1 GGUF for response generation.
    Suppresses API-related errors and ensures smooth output.
    """
    if online_mode:
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(
                prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
            )
            if response and hasattr(response, 'text'):
                return response.text.strip()
        except Exception:
            pass  # 🔹 Suppress API errors silently

    # 🔹 Using DeepSeek for response
    response = local_llm(prompt + "\n\n### Answer:", max_tokens=max_tokens, temperature=temperature)

    # ✅ Extract and return response
    choices = response.get("choices", [])
    return choices[0].get("text", "").strip() if choices else "⚠️ Error: No response generated."

# 🔹 Bias Detection Function (Strict Yes/No)
def detect_bias(sentence: str):
    """
    Detects if the entire sentence contains gender bias.
    Returns 'Yes' if there is *any* clear gender bias in phrasing, tone, or implication.
    Returns 'No' if the sentence is entirely free from gender-based assumptions, stereotypes, or comparisons.
    """

    prompt = f"""
    You are a language analysis model trained to detect gender bias.

    Task:
    Review the entire sentence carefully and determine if it contains any form of gender bias — explicit or implicit.
    
    Respond ONLY with:
    - "Yes" → if the sentence contains any clear or implied gender bias
    - "No" → if the sentence is entirely gender-neutral and free from bias

    Do NOT provide any explanation. Do NOT add punctuation or additional words.

    Sentence: "{sentence}"

    Response:
    """

    response = generate_response(prompt).strip().lower()

    # Extract and sanitize the first word of response
    first_word = response.split()[0].strip(".,!?")

    if first_word == "yes":
        return "Yes"
    elif first_word == "no":
        return "No"
    else:
        return "⚠️ Error: Unexpected model response"

# 🔹 Bias Correction Function
def correct_bias(sentence: str):
    """
    Transforms a gender-biased sentence into a gender-neutral version while preserving its meaning.
    Rewrites the sentence at a semantic level to eliminate gender-based assumptions or comparisons.
    """
    prompt = f"""
    You are an expert in inclusive and unbiased language.

    Task:
    Transform the following sentence into a gender-neutral version.

    Rules:
    - Do NOT simply replace gendered words
    - Avoid ANY direct or implied comparison between genders
    - Eliminate stereotypes or assumptions based on gender
    - Preserve the original meaning and tone as much as possible
    - Ensure natural, fluent, and inclusive grammar

    Sentence: "{sentence}"

    Gender-Neutral Version:
    """
    return generate_response(prompt).strip()

# 🔹 Bias Score Function (0% if no bias, else proper percentage)
def bias_score(sentence: str):
    """
    Returns a gender bias score from 0% (no bias) to 100% (strong bias).
    Uses bias detection first to determine if the sentence is biased.
    Ensures clean percentage output with no text.
    """
    
    # Step 1: Use the bias detector
    bias_present = detect_bias(sentence)

    # Step 2: If no bias, return 0%
    if bias_present == "No":
        return "0%"

    # Step 3: If bias exists, ask for percentage
    prompt = f"""
    You are a bias evaluation expert.

    Task:
    Rate the gender bias in the following sentence on a scale from 50% (mild bias) to 100% (extreme bias).
    Respond ONLY with the percentage number followed by a percent sign (e.g., "72%").
    Do NOT include any explanation or words.

    Sentence: "{sentence}"

    Response:
    """

    response = generate_response(prompt).strip()

    # Extract clean percentage
    match = re.search(r'\b\d{1,3}(\.\d+)?%', response)
    if match:
        return match.group()
    else:
        return "⚠️ Error: Invalid score format"


# 🔹 Speech-to-Text Function (Microphone Input)
def get_audio_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.success(f"✅ Recognized Text: {text}")
            return text
        except sr.UnknownValueError:
            st.error("⚠️ Could not understand audio. Try again.")
        except sr.RequestError:
            st.error("⚠️ Speech recognition service unavailable.")
    return ""

# 🔹 Streamlit UI
st.title("📝 Gender Bias Detection & Correction")
st.write("Detect, correct, and score gender bias in text.")

# 🔹 User Input (Text or Mic)
input_mode = st.radio("Choose Input Method:", ("📝 Text Input", "🎤 Microphone"))

if input_mode == "📝 Text Input":
    user_input = st.text_area("Enter a sentence:", "")
elif input_mode == "🎤 Microphone":
    if st.button("🎙️ Record Voice"):
        user_input = get_audio_input()
    else:
        user_input = ""

# 🔹 Process & Display Results
if user_input:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔍 Bias Detection")
        st.write(detect_bias(user_input))

    with col2:
        st.subheader("✅ Gender-Neutral Correction")
        st.write(correct_bias(user_input))

    with col3:
        st.subheader("📊 Bias Score")
        st.write(bias_score(user_input))

# 🔹 Cleanup (Close Model)
def cleanup():
    global local_llm
    if local_llm:
        del local_llm
        local_llm = None

st.button("🔄 Reset", on_click=cleanup)