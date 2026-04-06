import os
from google import genai
from dotenv import load_dotenv
import requests
import json
import re 
load_dotenv()

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("GENAI_API_KEY is not set in environment variables.")

client = genai.Client(api_key=GENAI_API_KEY)

def generate_gemini(prompt: str, model: str = "gemini-2.5-flash") -> str:
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response.text

def clean_model_json(text: str) -> str:
    """Clean LLM JSON by removing markdown and fixing common issues."""
    text = text.strip()

    # Remove ```json or ``` wrappers
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    # Remove trailing commas before } or ]
    text = re.sub(r",(\s*[}\]])", r"\1", text)

    return text


# def generate_gemini_content(prompt: str, model: str = "gemma3:1b"):
#     url = "http://localhost:11434/api/generate"

#     payload = {
#         "model": model,
#         "prompt": prompt
#     }

#     response = requests.post(url, json=payload, stream=True)
#     response.raise_for_status()

#     raw_text = ""

#     for line in response.iter_lines():
#         if not line:
#             continue
#         data = json.loads(line.decode("utf-8"))
#         if "response" in data:
#             raw_text += data["response"]

#     # First clean the JSON
#     cleaned = clean_model_json(raw_text)

#     # Try parsing
#     try:
#         parsed = json.loads(cleaned)
#         return parsed   # Always return dictionary
#     except Exception as e:
#         print("⚠ WARNING: Model did not return valid JSON.")
#         print("Raw output:")
#         print(raw_text)
#         print("Cleaned output:")
#         print(cleaned)
#         print("JSON Error:", e)

#         # Return EMPTY dict to avoid .get() crash
#         return {}

def generate_gemini_content(prompt: str, model: str = "gemma3:1b") -> str:
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt}

    response = requests.post(url, json=payload, stream=True)
    response.raise_for_status()

    raw_text = ""

    for line in response.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                raw_text += data["response"]
        except Exception:
            continue

    return raw_text  # Return plain string directly



def score_candidate_with_model(raw_text: str, jd: str, model: str = "llama2:latest"):
    """
    Sends candidate overview and JD to Ollama model for scoring.
    Returns a dictionary: {"score": int, "reason": str (optional)}
    """
    url = "http://localhost:11434/api/generate"

    prompt = f"""
    Job Description:
    {jd}

    Candidate Overview:
    {raw_text}

    Score the candidate relevance from 0 to 100. Only return the number, optionally followed by a dash and short reason.
    Dont assume that this profile as this skill and that skill ,it has good expereience ,take JD as the best
    if the candidate overview as per jd then only score good if it is not relevant then dont give suggestions
    """

    payload = {"model": model, "prompt": prompt}

    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
    except Exception as e:
        print("⚠ Error connecting to Ollama:", e)
        return {"score": 0, "reason": "Connection error"}

    raw_output = ""
    for line in response.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                raw_output += data["response"]
        except:
            continue

    cleaned = raw_output.strip()

    # Extract numeric score
    try:
        score_match = re.search(r'\b(\d{1,3})\b', cleaned)
        score = int(score_match.group(1)) if score_match else 0
        score = min(max(score, 0), 100)
    except:
        score = 0

    # Optional reason
    reason = cleaned.replace(str(score), "").strip(" -:") if cleaned else ""

    return {"score": score, "reason": reason}