import os
import json
import requests

INPUT_DIR = "terms_and_conditions"
OUTPUT_DIR = "cleaned_terms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_URL = "http://localhost:5000/generate"
API_KEY = "my_secret_key_123"  # 👈 replace with your actual key

def clean_terms_with_local_api(content):
    payload = {
        "prompt": f"""
        You are a text-cleaning system.
        Task: Extract ONLY the legal Terms and Conditions from the text below.  
        - Output only the cleaned terms text.  
        - Do not add explanations, apologies, or commentary.  
        - If no legal terms exist, return an empty string "".  

        Text:
        {content}
        """
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data.get("response", "").strip()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return ""

def process_all_files():
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(INPUT_DIR, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            print(f"Processing {filename}...")
            cleaned_text = clean_terms_with_local_api(raw_text)

            out_file = os.path.join(OUTPUT_DIR, filename.replace(".txt", ".json"))
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {"title": filename, "terms": cleaned_text},
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            print(f"✅ Saved cleaned file: {out_file}")

if __name__ == "__main__":
    process_all_files()
