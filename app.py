from llm import LLM
from embeddings import Embeddings
from vectordb import PineconeDB
from dotenv import load_dotenv
import os
load_dotenv()
db_key=os.getenv('db_api_key')
embedder_model_name=os.getenv('embedder_model_name')
llm_model_name=os.getenv('llm_model_name')
db_name=os.getenv('db_name')
def read_document(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def build_rag_pipeline():
    raw_text = read_document("sample.txt")
    chunks = chunk_text(raw_text)
    embedder = Embeddings(model_name=embedder_model_name)
    embeddings = embedder.generate_embeddings(chunks)

    api_key = db_key
    db = PineconeDB(api_key=api_key, db_name=db_name)

    vectors = [(f"doc-{i}", emb.tolist(), {"text": chunks[i]}) for i, emb in enumerate(embeddings)]
    db.upsert(vectors)

    question = "long term effects of AI?"
    question_embedding = embedder.generate_embeddings([question])[0].tolist()
    results = db.query(vector=question_embedding, top_k=1)

    #print("\n--- Pinecone Results ---")
    #print(results)

    matches = results.get("matches", [])
    #print("matches")
    #print(matches)
    if not matches or matches[0].get("score", 0) < 0.15:
        response = "Sorry, I couldn't find anything in the document related to your query."
        context = "none"
    else:
        context = matches[0]["metadata"]["text"]
        llm = LLM(model_name=llm_model_name)
        response = llm.ask(
            prompt=question,
            context=context,
            temperature=0.9,
            max_tokens=200
        )

    print("\n--- Question ---")
    print(question)
    print("\n--- Retrieved Context ---")
    print(context)
    print("\n--- LLM Answer ---")
    if response.endswith((".", "...", "!", "?")):
        print(response)
    else:
        print(response + "\n\n(Note: The answer may be incomplete due to token limit.)")



if __name__ == "__main__":
    build_rag_pipeline()