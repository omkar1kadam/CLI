import os
import json
import requests
import shutil  # for moving files

CLEANED_DIR = "cleaned_terms"
DELETED_DIR = "deleted_files"
API_URL = "https://0cbdd9884fdb.ngrok-free.app/api/generate"  # 👈 updated
API_KEY = "my_secret_key_123"  # 👈 replace if needed

# make sure deleted dir exists
os.makedirs(DELETED_DIR, exist_ok=True)

def ai_judgement(text: str) -> bool:
    """
    Ask the AI if the text is good training data.
    Returns True if useful, False if junk.
    """
    payload = {
        "prompt": f"""
        You are a data quality filter for training a summarization model.
        Task: Decide if the following text contains meaningful policy/legal-style content
        that could be useful for training.

        ✅ Accept if:
        - The majority of the text looks like Terms & Conditions, Privacy Policy, Disclaimer, Agreement, or similar.
        - It is structured with sections, rules, obligations, disclaimers, governing law, or user responsibilities.
        - It includes legal phrases like "limitation of liability", "governing law", "intellectual property", etc.
        - Even if it contains extra disclaimers like "I'm not a lawyer" OR AI-like helper text/questions at the end.

        ❌ Reject only if:
        - The text is completely irrelevant (apology, nonsense, random text).
        - It is empty or just a placeholder.
        - It has no meaningful policy/legal content.

        Rules:
        - Reply only "YES" if the majority is useful training data.
        - Reply only "NO" if it is entirely junk.

        Text:
        {text}
        """
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            decision = response.json().get("response", "").strip().upper()
            print(f"🤖 AI decided: {decision}")  # Debug: show raw AI answer
            return decision == "YES"
        else:
            print(f"⚠️ API error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"⚠️ Request failed: {e}")
        return False



def filter_files_with_ai():
    for filename in os.listdir(CLEANED_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(CLEANED_DIR, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                terms = data.get("terms", "").strip()
                if not terms:
                    print(f"🗑️ Empty file → moving {filename} to deleted_files/")
                    shutil.move(file_path, os.path.join(DELETED_DIR, filename))
                    continue

                # Ask AI if it's good
                if ai_judgement(terms):
                    print(f"✅ Keeping {filename}")
                else:
                    print(f"🗑️ Moving junk {filename} → deleted_files/")
                    shutil.move(file_path, os.path.join(DELETED_DIR, filename))

            except Exception as e:
                print(f"⚠️ Error reading {filename}: {e}")

if __name__ == "__main__":
    filter_files_with_ai()
