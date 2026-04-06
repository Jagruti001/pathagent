import chromadb
from agent.embeddings import embeddings

# Local ChromaDB — no server, no cost, stores data in ./chroma_db folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def retrieve_guidelines(query: str, n_results: int = 5) -> str:
    """
    Retrieve relevant medical guideline chunks from ChromaDB
    using free local sentence-transformer embeddings.
    """
    try:
        collection = chroma_client.get_collection(name="medical_guidelines")

        # Embed the query using local sentence-transformers (free)
        query_embedding = embeddings.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        if not results["documents"][0]:
            return "No relevant guidelines found."

        # Combine all retrieved chunks into one context string
        context = "\n\n---\n\n".join(results["documents"][0])
        return context

    except Exception as e:
        return f"Guidelines not loaded yet. Please run: python rag/ingest.py\nError: {str(e)}"
