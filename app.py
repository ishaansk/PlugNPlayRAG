import streamlit as st
from llm import LLM
from embeddings import Embeddings
from vectordb import PineconeDB
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
db_key = os.getenv('db_api_key')
embedder_model_name = os.getenv('embedder_model_name')
llm_model_name = os.getenv('llm_model_name')
db_name = os.getenv('db_name')

# Initialize LLM instance once at the start
llm = LLM(model_name="google/flan-t5-base")  # New model


# Helper functions
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

def build_rag_response(query):
    try:
        # Read and chunk the document
        raw_text = read_document("sample.txt")
        chunks = chunk_text(raw_text)

        # Generate embeddings for chunks
        embedder = Embeddings(model_name=embedder_model_name)
        embeddings = embedder.generate_embeddings(chunks)

        # Interact with Pinecone DB
        db = PineconeDB(api_key=db_key, db_name=db_name)
        vectors = [(f"doc-{i}", emb.tolist(), {"text": chunks[i]}) for i, emb in enumerate(embeddings)]
        db.upsert(vectors)

        # Generate embedding for the query and search Pinecone DB
        question_embedding = embedder.generate_embeddings([query])[0].tolist()
        results = db.query(vector=question_embedding, top_k=1)
        matches = results.get("matches", [])

        if not matches or matches[0].get("score", 0) < 0.15:
            return "none", "Sorry, I couldn't find anything in the document related to your query."
        
        # Extract the relevant context
        context = matches[0]["metadata"]["text"]
        
        # Ask the LLM to generate a response based on the context and query
        response = llm.ask(prompt=query, context=context, temperature=0.7, max_tokens=200)
        return context, response
    except Exception as e:
        return "none", f"An error occurred: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Plug n Play RAG", layout="centered")
st.title("🔌 Plug n Play RAG")
st.markdown("Ask a question based on the document contents!")

# Input for query
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Searching and generating response..."):
        context, answer = build_rag_response(query)
    
    # Display the context and answer
    st.subheader("🔍 Retrieved Context")
    st.info(context if context != "none" else "No relevant context found.")

    st.subheader("🧠 Answer")
    st.success(answer)
