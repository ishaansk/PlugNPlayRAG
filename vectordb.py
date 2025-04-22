# src/vectordb.py

from pinecone import Pinecone
from pinecone.exceptions import PineconeException
import os
from dotenv import load_dotenv

load_dotenv()
api_key=os.getenv("db_api_key")

class PineconeDB:
    def __init__(self, api_key, db_name="rag-opensource", dimension=384, metric="cosine"):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = db_name
        self.dimension = dimension
        self.metric = metric

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec={"cloud": "aws", "region": "us-east-1"}  # Adjust as needed
            )

        self.index = self.pc.Index(self.index_name)

    def check_index_exists(self):
        try:
            stats = self.index.describe_index_stats()
            print("✅ Connected to Pinecone index.")
            print(stats)
        except PineconeException as e:
            print("❌ Failed to connect to Pinecone index:", e)

    def upsert(self, vectors):
        """
        Upsert vectors into the index.
        :param vectors: List of tuples (id, vector, metadata)
        """
        self.index.upsert(vectors)

    def query(self, vector, top_k=5, include_metadata=True):
        """
        Query the index for similar vectors.
        :param vector: A single embedding vector
        :param top_k: Number of top matches
        """
        return self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=include_metadata
        )

    def delete(self, ids):
        """
        Delete vectors by ID.
        :param ids: List of vector IDs to delete
        """
        self.index.delete(ids=ids)


# Debug/test connection when run directly
if __name__ == "__main__":
    pinecone_db = PineconeDB(api_key=api_key, db_name="rag-opensource")
    pinecone_db.check_index_exists()
