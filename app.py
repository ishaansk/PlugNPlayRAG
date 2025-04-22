import streamlit as st
from llm import LLM
from embeddings import Embeddings
from vectordb import PineconeDB
from dotenv import load_dotenv
import os

load_dotenv()
db_key = os.getenv('db_api_key')
embedder_model_name = os.getenv('embedder_model_name')
llm_model_name = os.getenv('llm_model_name')
db_name = os.getenv('db_name')

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_response(user_query):
    with open("sample.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    chunks = chunk_text(raw_text)
    embedder = Embeddings(model_name=embedder_model_name)
    embeddings = embedder.generate_embeddings(chunks)

    db = PineconeDB(api_key=db_key, db_name=db_name)
    vectors = [(f"doc-{i}", emb.tolist(), {"text": chunks[i]}) for i, emb in enumerate(embeddings)]
    db.upsert(vectors)

    query_embedding = embedder.generate_embeddings([user_query])[0].tolist()
    results = db.query(vector=query_embedding, top_k=1)
    matches = results.get("matches", [])

    if not matches or matches[0].get("score", 0) < 0.15:
        return "none", "Sorry, I couldn't find anything in the document related to your query."

    context = matches[0]["metadata"]["text"]
    llm = LLM(model_name=llm_model_name)
    response = llm.ask(
        prompt=user_query,
        context=context,
        temperature=0.9,
        max_tokens=200
    )
    return context, response

st.set_page_config(page_title="RAG Q&A", layout="centered")
st.title("📄🔍 Ask Your Document")

user_question = st.text_input("Enter your question:")

if st.button("Ask") and user_question.strip():
    with st.spinner("Thinking..."):
        context, answer = get_response(user_question)
        st.subheader("Answer")
        if answer.endswith((".", "...", "!", "?")):
            st.write(answer)
        else:
            st.write(answer + "\n\n(Note: The answer may be incomplete due to token limit.)")

        if context != "none":
            with st.expander("Show Retrieved Context"):
                st.write(context)