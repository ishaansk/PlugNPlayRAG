import os
import requests
from dotenv import load_dotenv

load_dotenv()

class LLM:
    def __init__(self, model_name="google/flan-t5-base"):
        self.api_key = os.getenv("HF_API_KEY")
        self.base_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def ask(self, prompt, context="", temperature=0.7, max_tokens=500):  # Increase max_tokens here
        payload = {
            "inputs": f"{context}\n{prompt}",
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens
            }
        }

        response = requests.post(self.base_url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()[0]["generated_text"]