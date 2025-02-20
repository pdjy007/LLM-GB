import os
import socket
import google.generativeai as genai
from llama_cpp import Llama

# Load Local LLM (Mistral 7B GGUF) with Optimized Parameters
model_path = r"C:\Users\pdjy0\OneDrive\Documents\Final-Year-Project\project\backend\app\models\mistral-7b-v0.1.Q4_K_M.gguf"

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

# Function to Generate Gender-Neutral Job Descriptions
def generate_job_description(job_title: str, max_tokens: int = 300, temperature: float = 0.5):
    """
    Generates a gender-neutral job description for a given job title.
    Uses Gemini API if online, otherwise falls back to Mistral 7B.
    """

    prompt = f"""Generate a detailed, gender-neutral job description for the role of {job_title}.
    The description should be inclusive, avoiding gendered language, and should emphasize skills and qualifications.
    
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
            print("Using Gemini API for job description generation...")
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

    print("Using Mistral model for job description generation...")
    response = local_llm(prompt, max_tokens=max_tokens, temperature=temperature)
    
    return response["choices"][0]["text"].strip()

# Properly Close Llama Model
def cleanup():
    global local_llm
    if local_llm:
        del local_llm
        local_llm = None
        print("🟢 Model closed successfully.")

# Main Execution (Test Case)
if __name__ == "__main__":
    job_title = "Software Engineer"  # Change this for different jobs
    print("\n📄 Generated Job Description:\n")
    print(generate_job_description(job_title))

    # Properly Close Llama Model
    cleanup()
