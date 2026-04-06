from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List

# Best free sentence-transformers model for semantic similarity
# all-MiniLM-L6-v2: fast, lightweight, great for RAG
# Downloads automatically on first run (~90MB)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class LocalEmbeddings(Embeddings):
    """
    Free local embeddings using sentence-transformers.
    No API key. No cost. Runs fully offline after first download.
    """

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()


# Singleton instance — reused across ingest and retrieval
embeddings = LocalEmbeddings()
