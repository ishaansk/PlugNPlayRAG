from llm import LLM

def test_llm_response():
    llm = LLM(model_name="google/flan-t5-base")  # New model

    response = llm.ask(
        prompt="What is the capital of France?",
        context="France is a European country.",
        temperature=0.5,
        max_tokens=50
    )
    print("LLM Response:", response)

if __name__ == "__main__":
    test_llm_response()
