from flask import Flask, request, render_template, jsonify, session
from llm import LLM
from embeddings import Embeddings
from vectordb import VectorDataBase
from dotenv import load_dotenv
import os
import PyPDF2
from agentic_rag import AgenticRAGController
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("flask_key")


# Initialize environment variables and clients
db_key = os.getenv('db_api_key')
embedder_model_name = os.getenv('embedder_model_name')
db_name = os.getenv('db_name')
embedder = Embeddings(model_name=embedder_model_name)
db = VectorDataBase(api_key=db_key, db_name=db_name)
llm = LLM()
agentic_controller = AgenticRAGController(embedder=embedder, db=db, llm=llm)
def clear_database_if_not_empty():
    try:
        stats = db.index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        if total_vectors > 0:
            print(f"Database has {total_vectors} vectors. Clearing database...")
            db.index.delete(delete_all=True)
            print("Database cleared.")
        else:
            print("Database is already empty.")
    except Exception as e:
        print(f"Error checking or clearing database: {e}")
def read_pdf(file_stream):
    pdf_reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

@app.route('/')
def index():
    # Simple HTML form for upload and question input
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Clear DB if not empty before uploading new document
    clear_database_if_not_empty()

    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        raw_text = read_pdf(file)
    elif filename.endswith('.txt'):
        raw_text = file.read().decode('utf-8')
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    chunks = chunk_text(raw_text)
    embeddings = embedder.generate_embeddings(chunks)
    vectors = [(f"doc-{i}", emb.tolist(), {"text": chunks[i]}) for i, emb in enumerate(embeddings)]
    db.upsert(vectors)

    return jsonify({
        "status": "success",
        "message": "File processed and embedded successfully."
    }), 200



@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    temperature = float(data.get('temperature', 0.7))
    max_tokens = int(data.get('max_tokens', 300))

    if not question:
        return jsonify({"error": "No question provided"}), 400

    chat_history = session.get('chat_history', [])

    # Format chat history for prompt context
    formatted_history = ""
    for turn in chat_history:
        formatted_history += f"{turn['role']}: {turn['content']}\n"

    prompt_context = formatted_history + f"Human: {question}\n"

    question_embedding = embedder.generate_embeddings([question])[0].tolist()
    results = db.query(vector=question_embedding, top_k=3)
    matches = results.get("matches", [])
    relevant_matches = [m for m in matches if m.get("score", 0) >= 0.1]

    if not relevant_matches:
        answer = "❌ Sorry, I couldn't find anything relevant in the document."
    else:
        context = "\n\n".join([m["metadata"]["text"] for m in relevant_matches])
        full_context = context + "\n\n" + prompt_context

        answer = llm.agentic_ask(
            prompt=question,
            context=full_context,
            temperature=temperature,
            max_tokens=max_tokens,
            mode="Auto"
        )

    chat_history.append({"role": "Q", "content": question})
    chat_history.append({"role": "A", "content": answer})
    session['chat_history'] = chat_history

    # Return entire chat history to frontend
    return jsonify({
        "answer": answer,
        "chat_history": chat_history
    })

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session.pop('chat_history', None)
    return jsonify({"message": "Chat history cleared."})



if __name__ == '__main__':
    clear_database_if_not_empty()  # <- Call here before starting the app
    app.run(debug=True)