from openai import AzureOpenAI
import os

class LLM:
    def __init__(self):
        from openai import AzureOpenAI
        import os
        self.client = AzureOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            api_version=os.getenv("LLM_API_VERSION"),
            azure_endpoint=os.getenv("LLM_ENDPOINT")
        )
        self.model = os.getenv("LLM_DEPLOYMENT_NAME")

    def ask(self, prompt, context, temperature=0.7, max_tokens=300):
        return self._chat(prompt, context, temperature, max_tokens)

    def agentic_ask(self, prompt, context, temperature=0.7, max_tokens=300, mode="Auto"):
        task_prompt = self._route_prompt(prompt, mode)
        return self._chat(task_prompt, context, temperature, max_tokens)

    def _chat(self, prompt, context, temperature, max_tokens):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are an assistant. Use the following context: {context}"},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def _route_prompt(self, prompt, mode):
        if mode == "Summarize":
            return f"Summarize the following context based on this query: {prompt}"
        elif mode == "Reasoning":
            return f"Use step-by-step logical reasoning to answer this question: {prompt}"
        elif mode == "Number Focused":
            return f"Focus on numerical data and calculations in the context to answer: {prompt}"
        else:
            return f"Based on the context, answer this question thoughtfully: {prompt}"
