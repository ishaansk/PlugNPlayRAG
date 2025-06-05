from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv()
model_name=os.getenv('embedder_model_name')

class Embeddings:
    def __init__(self, model_name=model_name):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)