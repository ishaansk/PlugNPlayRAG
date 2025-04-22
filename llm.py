# src/llm.py

import ollama

class LLM:
    def __init__(self, model_name="mistral"):
        self.model_name = model_name

    def ask(self, prompt, context, temperature=0.7, max_tokens=300):
        full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": full_prompt}],
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        return response['message']['content']
