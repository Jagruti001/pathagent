"""
Run this script ONCE before starting the app.
It loads your medical guideline PDFs into ChromaDB using FREE local embeddings.

Usage:
    python rag/ingest.py

Place PDF guidelines in the data/guidelines/ folder.
Free PDFs to download:
- ADA Standards of Care: https://diabetesjournals.org/care
- WHO Guidelines:        https://www.who.int/publications
- NIH MedlinePlus:       https://medlineplus.gov

Embedding model used: sentence-transformers/all-MiniLM-L6-v2
- FREE, runs locally, no API key needed
- Downloads automatically (~90MB, one time only)
"""

import os
import sys
import chromadb
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.embeddings import embeddings

GUIDELINES_DIR = "./data/guidelines"
CHROMA_PATH = "./chroma_db"


def ingest_guidelines():
    print("🔄 Starting ingestion of medical guidelines...")
    print(f"📦 Embedding model: sentence-transformers/all-MiniLM-L6-v2 (FREE, local)\n")

    # Check folder exists
    if not os.path.exists(GUIDELINES_DIR):
        os.makedirs(GUIDELINES_DIR)
        print(f"📁 Created {GUIDELINES_DIR}")
        print("⚠️  Please add medical guideline PDFs to that folder and run again.")
        return

    # Find PDFs
    pdf_files = [f for f in os.listdir(GUIDELINES_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️  No PDFs found in {GUIDELINES_DIR}")
        print("Please add at least one medical guideline PDF and run again.")
        return

    # Load all PDFs
    all_docs = []
    for pdf_file in pdf_files:
        path = os.path.join(GUIDELINES_DIR, pdf_file)
        print(f"📄 Loading: {pdf_file}")
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"   → {len(docs)} pages loaded")

    print(f"\n✅ Total pages loaded: {len(all_docs)}")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)
    print(f"✂️  Split into {len(chunks)} chunks\n")

    # Setup ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete old collection if re-ingesting
    try:
        client.delete_collection("medical_guidelines")
        print("🗑️  Deleted old collection")
    except Exception:
        pass

    collection = client.create_collection("medical_guidelines")

    # Embed and insert in batches
    print("🔢 Embedding chunks using sentence-transformers (this may take a minute)...")
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [doc.page_content for doc in batch]

        # Use local sentence-transformer embeddings (FREE)
        embeds = embeddings.embed_documents(texts)
        ids = [f"chunk_{i + j}" for j in range(len(batch))]

        collection.add(
            documents=texts,
            embeddings=embeds,
            ids=ids
        )
        print(f"  ✅ Inserted chunks {i} → {i + len(batch)}")

    print(f"\n🎉 Done! {len(chunks)} chunks stored in ChromaDB at {CHROMA_PATH}")
    print("You can now run: streamlit run app.py")


if __name__ == "__main__":
    ingest_guidelines()
