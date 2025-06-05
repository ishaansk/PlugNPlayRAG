from pinecone import Pinecone
from pinecone.exceptions import PineconeException
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read variables from .env
api_key = os.getenv("db_api_key")
db_name = os.getenv("db_name")  # Now using from .env

class VectorDataBase:
    def __init__(self, api_key, db_name, dimension=384, metric="cosine"):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = db_name
        self.dimension = dimension
        self.metric = metric

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec={"cloud": "aws", "region": "us-east-1"}
            )

        self.index = self.pc.Index(self.index_name)

    def check_index_exists(self):
        try:
            stats = self.index.describe_index_stats()
            print("Connected to Pinecone index.")
            print(stats)
        except PineconeException as e:
            print("Failed to connect to Pinecone index:", e)

    def upsert(self, vectors):
        self.index.upsert(vectors)

    def query(self, vector, top_k=5, include_metadata=True):
        return self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=include_metadata
        )
    

    def delete(self, ids):
        self.index.delete(ids=ids)

if __name__ == "__main__":
    pinecone_db = VectorDataBase(api_key=api_key, db_name=db_name)
    pinecone_db.check_index_exists()
