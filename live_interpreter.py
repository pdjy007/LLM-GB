
import streamlit as st
import speech_recognition as sr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from googletrans import Translator
from gtts import gTTS
import os
import tempfile
import requests
import time

# Set page title
st.set_page_config(page_title="Live Interpreter", layout="wide")
st.title("🔊 Live Interpreter - Real-Time Translation")

# Language selection
languages = {"Telugu": "te", "Hindi": "hi", "Kannada": "kn", "English": "en", "Japanese": "ja", "Korean": "ko"}

col1, col2 = st.columns(2)
with col1:
    lang1 = st.selectbox("User 1 Language", list(languages.keys()), index=3)
with col2:
    lang2 = st.selectbox("User 2 Language", list(languages.keys()), index=0)

lang1_code = languages[lang1]
lang2_code = languages[lang2]

# Initialize translator and speech recognizer
translator = Translator()
recognizer = sr.Recognizer()

# Check internet connectivity
def is_online():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except requests.ConnectionError:
        return False

online_mode = is_online()

# Load NLLB model for offline translation if required
if not online_mode:
    model_name = r"C:\Users\pdjy0\OneDrive\Documents\Final-Year-Project\project\backend\models\nllb-200-distilled-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Translation function (Auto selects online/offline)
def translate_text(text, src_lang, dest_lang):
    if online_mode:
        return translator.translate(text, src=src_lang, dest=dest_lang).text
    else:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        output = model.generate(**inputs)
        return tokenizer.decode(output[0], skip_special_tokens=True)

# Voice recognition function
def recognize_speech():
    with sr.Microphone() as source:
        st.write("🎤 Listening...")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except:
            return "❌ Could not recognize speech"

# Text-to-Speech (TTS) function (Automatically plays audio and waits)
def speak_text(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)

    # Play audio & wait for completion
    os.system(f'start /min wmplayer "{temp_file.name}"')
    time.sleep(len(text) * 0.5)  # Estimate duration to prevent overlap

# **Main Conversation Logic**
if "conversation_active" not in st.session_state:
    st.session_state.conversation_active = False

if st.button("Start Conversation"):
    st.session_state.conversation_active = True

if st.button("Stop Conversation"):
    st.session_state.conversation_active = False

if st.session_state.conversation_active:
    st.write("🎤 Speak when prompted...")

    while st.session_state.conversation_active:  # 🔄 Loop while active
        # User 1 speaks
        st.subheader(f"🗣 {lang1} User Speaking...")
        user1_text = recognize_speech()
        st.write(f"Recognized: {user1_text}")

        # Translate User 1's speech for User 2
        translated_text = translate_text(user1_text, lang1_code, lang2_code)
        st.success(f"Translated: {translated_text}")

        # **Play translation & wait**
        speak_text(translated_text, lang2_code)

        # **Check if Stop button was clicked**
        if not st.session_state.conversation_active:
            break

        # User 2 speaks **only after hearing translation**
        st.subheader(f"🗣 {lang2} User Speaking...")
        user2_text = recognize_speech()
        st.write(f"Recognized: {user2_text}")

        # Translate User 2's speech for User 1
        translated_text_2 = translate_text(user2_text, lang2_code, lang1_code)
        st.success(f"Translated: {translated_text_2}")

        # **Play translation & wait**
        speak_text(translated_text_2, lang1_code)

        # **Check if Stop button was clicked**
        if not st.session_state.conversation_active:
            break
