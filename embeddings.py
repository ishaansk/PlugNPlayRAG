# src/embeddings.py

from sentence_transformers import SentenceTransformer
import os

class Embeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the Embeddings class with a SentenceTransformer model.
        :param model_name: Name of the pre-trained SentenceTransformer model to use.
        """
        self.model = SentenceTransformer(model_name)

# embeddings.py
    def generate_embeddings(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)
