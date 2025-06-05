import streamlit as st
from llm import LLM
from embeddings import Embeddings
from vectordb import VectorDataBase
from dotenv import load_dotenv
import os
import PyPDF2

load_dotenv()

db_key = os.getenv('db_api_key')
embedder_model_name = os.getenv('embedder_model_name')
db_name = os.getenv('db_name')
def read_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text
def read_text(uploaded_file):
    return uploaded_file.read().decode("utf-8")
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

# UI
st.set_page_config(page_title="PlugNPlay RAG", layout="wide")
st.title("📄 Chat with Your Documents")

if "history" not in st.session_state:
    st.session_state.history = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_context" not in st.session_state:
    st.session_state.last_context = ""

# Upload
uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"])

# Sidebar
with st.sidebar:
    st.header("LLM Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    max_tokens = st.slider("Max Tokens", 50, 1000, 300, 50)
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.history = []
    if st.button("Clear Database"):
        db = VectorDataBase(api_key=db_key, db_name=db_name)
        index_description = db.index.describe_index_stats()
        if index_description["total_vector_count"] > 0:
            db.index.delete(delete_all=True)
            st.success("🧹 Database cleared!")
        else:
            st.info("✅ Database already empty.")

# File Processing
if uploaded_file:
    filetype = uploaded_file.name.split(".")[-1]
    raw_text = read_pdf(uploaded_file) if filetype == "pdf" else read_text(uploaded_file)
    with st.spinner("Reading and embedding..."):
        chunks = chunk_text(raw_text)
        embedder = Embeddings(model_name=embedder_model_name)
        embeddings = embedder.generate_embeddings(chunks)
        db = VectorDataBase(api_key=db_key, db_name=db_name)
        vectors = [(f"doc-{i}", emb.tolist(), {"text": chunks[i]}) for i, emb in enumerate(embeddings)]
        db.upsert(vectors)

    agent_mode = st.selectbox("Choose Agent Mode", ["Auto", "Summarize", "Reasoning", "Number Focused"])
    question = st.text_input("Ask a question:")

    if question:
        with st.spinner("Generating answer..."):
                question_embedding = embedder.generate_embeddings([question])[0].tolist()
                results = db.query(vector=question_embedding, top_k=3)
                #st.write("Match results", results)
                matches = results.get("matches", [])
                relevant_matches = [m for m in matches if m.get("score", 0) >= 0.1]
                if not relevant_matches:
                    st.session_state.last_context = ""
                    st.session_state.last_question = question
                    response = "❌ Sorry, I couldn't find anything relevant in the document."
                    st.session_state.history.append({
                        "question": question,
                        "context": "No context (score < 0.3 or no match)",
                        "response": response
                    })
                else:
                    st.session_state.last_context = "\n\n".join([m["metadata"]["text"] for m in relevant_matches])
                    st.session_state.last_question = question

                    llm = LLM()
                    response = llm.agentic_ask(
                        prompt=question,
                        context=st.session_state.last_context,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        mode=agent_mode
                    )

                    st.session_state.history.append({
                        "question": question,
                        "context": st.session_state.last_context,
                        "response": response
                    })
if st.session_state.history:
    st.subheader("💬 Chat History")
    for i, chat in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**Q:** {chat['question']}")
        with st.expander(f"Context (Match {i})"):
            st.code(chat['context'])
        st.success(f"A{i}: {chat['response']}")
        
        #st.markdown(agent_mode)