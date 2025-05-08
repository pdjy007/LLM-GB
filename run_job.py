import os
import socket
import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
from llama_cpp import Llama

# 🎨 Streamlit UI Config
st.set_page_config(page_title="Gender-Neutral Job Description Generator", layout="wide")

# 🌐 Check Internet Connectivity
def check_internet():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

# 🧠 Load Local DeepSeek Model (Offline Mode)
MODEL_PATH = r"C:\Users\pdjy0\Downloads\DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

local_llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,  # ✅ Increased Context Size to Avoid Output Truncation
    n_threads=8,
    f16_kv=True,
    use_mlock=True
)

# 🌍 Configure Gemini API (Online Mode)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
online_mode = bool(GOOGLE_API_KEY) and check_internet()

if online_mode:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-pro")
    except Exception:
        online_mode = False

# 📝 Generate Gender-Neutral Job Description
def generate_job_description(job_title: str, max_tokens: int = 500, temperature: float = 0.4):
    """
    Generates a gender-neutral job description.
    Uses Gemini API if online, otherwise falls back to DeepSeek.
    """
    prompt = f"""
    Generate a detailed, gender-neutral job description for the role of {job_title}.
    The description should be inclusive, avoiding gendered language, and emphasizing skills and qualifications.
    
    ---
    Job Title: {job_title}

    Responsibilities:
    - 

    Qualifications:
    - 

    Work Environment:
    - 

    Ensure the description remains gender-neutral, welcoming individuals from all backgrounds.
    Provide only the job description without additional explanations.
    """

    if online_mode:
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
            )
            if response and hasattr(response, 'text'):
                return response.text.strip()
        except Exception:
            pass  # ✅ Suppress API errors

    # 🔥 DeepSeek Model for Job Description (Offline Mode)
    response = local_llm(prompt + "\n\n### Response:", max_tokens=max_tokens, temperature=temperature)
    choices = response.get("choices", [])
    return choices[0].get("text", "").strip() if choices else "⚠️ Error: No response generated."

# 🎤 Speech-to-Text (Microphone Input)
def get_audio_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.success(f"✅ Recognized: {text}")
            return text
        except sr.UnknownValueError:
            st.error("⚠️ Could not understand. Try again.")
        except sr.RequestError:
            st.error("⚠️ Speech service unavailable.")
    return ""

# ✅ Streamlit UI
st.sidebar.title("⚙️ Settings")
temperature = st.sidebar.slider("Creativity (Temperature)", 0.0, 1.0, 0.4)
max_tokens = st.sidebar.slider("Max Tokens", 100, 700, 500)

st.title("📄 Gender-Neutral Job Description Generator")
st.markdown("**Enter a job title below, or use voice input, and a gender-neutral job description will be generated.**")

# 🔍 User Input Options
input_mode = st.radio("Choose Input Method:", ["📝 Text Input", "🎤 Microphone"])

job_title = ""

if input_mode == "📝 Text Input":
    job_title = st.text_input("🔍 Enter Job Title:", placeholder="e.g., Software Engineer")
elif input_mode == "🎤 Microphone":
    if st.button("🎙️ Record Voice"):
        job_title = get_audio_input()
        if job_title:  # ✅ Automatically generate description after voice input
            st.success(f"🔍 Recognized Job Title: {job_title}")
            description = generate_job_description(job_title, max_tokens, temperature)
            st.subheader("📑 Generated Job Description")
            st.write(description)

# 🔥 Generate and Display Job Description (For Text Input)
if st.button("Generate Description 🚀"):
    if job_title.strip():
        description = generate_job_description(job_title, max_tokens, temperature)
        st.subheader("📑 Generated Job Description")
        st.write(description)
    else:
        st.warning("⚠️ Please enter a job title.")

# 🛑 Cleanup Llama Model on Exit
def cleanup():
    global local_llm
    if local_llm:
        del local_llm
        local_llm = None

import atexit
atexit.register(cleanup)
