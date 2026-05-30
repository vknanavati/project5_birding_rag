# ============================================================
# 02_chunk_and_embed.py
#
# PURPOSE: Take the raw bird documents, split them into small
# overlapping chunks, convert each chunk into an embedding
# vector, and store everything in ChromaDB.
#
# Run this once to build the vector index. You only need to
# re-run it if you add or change documents.
# ============================================================

import os  # for file paths and directory listing
from sentence_transformers import SentenceTransformer  # converts text to vectors
import chromadb  # our vector database

# ── Settings ──────────────────────────────────────────────────

# Where our bird .txt files live
DOCS_DIR = os.path.join("data", "birds")

# Where ChromaDB will store its index files
CHROMA_DIR = "chroma_db"

# The name of our collection inside ChromaDB
# Think of a collection like a table in a regular database
COLLECTION_NAME = "birds"

# Chunk size: how many characters per chunk
# ~400 words is roughly 2000 characters — enough context without being too long
CHUNK_SIZE = 2000

# Overlap: how many characters to repeat between chunks
# This ensures sentences at chunk boundaries aren't cut off
CHUNK_OVERLAP = 200

# The embedding model we'll use — runs locally, no API needed
# all-MiniLM-L6-v2 is small, fast, and very capable for semantic search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ══════════════════════════════════════════════════════════════
# STEP 1 — CHUNKING
#
# Split a long document into smaller overlapping pieces.
#
# Analogy: imagine cutting a long newspaper article into index
# cards. Each card has ~400 words. The last 80 words of each
# card are repeated at the top of the next card — so no
# sentence gets cut in half at a boundary.
# ══════════════════════════════════════════════════════════════

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks of characters.
    Returns a list of string chunks.
    """
    chunks = []  # will hold all our chunk strings

    start = 0  # where the current chunk begins

    # Keep slicing until we've covered the whole document
    while start < len(text):

        # The end of this chunk is start + chunk_size
        end = start + chunk_size

        # Slice out this chunk
        chunk = text[start:end]

        # Only keep the chunk if it has meaningful content
        # (avoids saving nearly-empty trailing chunks)
        if len(chunk.strip()) > 50:
            chunks.append(chunk)

        # Move the start forward by chunk_size MINUS overlap
        # This is what creates the overlapping effect
        start += chunk_size - overlap

    return chunks


# ══════════════════════════════════════════════════════════════
# STEP 2 — LOAD DOCUMENTS
#
# Read every .txt file from the data/birds/ folder.
# Returns a list of dicts, each with 'species' and 'text'.
# ══════════════════════════════════════════════════════════════

def load_documents():
    """
    Load all bird .txt files from DOCS_DIR.
    Returns a list of dicts: [{'species': 'american_robin', 'text': '...'}]
    """
    documents = []

    # os.listdir returns all filenames in the folder
    for filename in os.listdir(DOCS_DIR):

        # Only process .txt files — skip anything else
        if not filename.endswith(".txt"):
            continue

        # Build the full path to the file
        filepath = os.path.join(DOCS_DIR, filename)

        # Read the file contents
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # The species name is just the filename without .txt
        # e.g. "american_robin.txt" → "american_robin"
        species = filename.replace(".txt", "")

        documents.append({
            "species": species,
            "text": text
        })

    print(f"  Loaded {len(documents)} documents from {DOCS_DIR}/")
    return documents


# ══════════════════════════════════════════════════════════════
# STEP 3 — EMBED AND STORE IN CHROMADB
#
# For each chunk:
#   1. Convert it to a vector using the embedding model
#   2. Store the vector + original text in ChromaDB
#
# ChromaDB stores three things per chunk:
#   - The embedding vector (for semantic search)
#   - The original text (so we can return it to the user)
#   - Metadata (species name, chunk number — for filtering)
#
# Analogy: ChromaDB is a library where every index card has
# been assigned GPS coordinates based on its meaning. When
# you ask a question, it finds the cards whose coordinates
# are closest to your question's coordinates.
# ══════════════════════════════════════════════════════════════

def build_index(documents):
    """
    Chunk all documents, embed them, and store in ChromaDB.
    """

    # ── Load the embedding model ───────────────────────────────
    # This downloads the model the first time (~90MB)
    # After that it's cached locally and loads instantly
    print(f"\n  Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("  Model loaded.")

    # ── Connect to ChromaDB ────────────────────────────────────
    # PersistentClient saves the index to disk in CHROMA_DIR
    # so it survives between script runs
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete the collection if it already exists
    # This lets us re-run the script cleanly if we update documents
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass  # collection didn't exist yet, that's fine

    # Create a fresh collection
    # We set embedding_function to None because we'll provide
    # our own embeddings rather than letting ChromaDB embed for us
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity for search
    )
    print(f"  Created collection '{COLLECTION_NAME}'")

    # ── Process each document ──────────────────────────────────
    total_chunks = 0

    for doc in documents:
        species = doc["species"]
        text = doc["text"]

        # Split this document into overlapping chunks
        chunks = chunk_text(text)
        print(f"\n  {species}: {len(chunks)} chunks")

        # Build lists to batch-add to ChromaDB
        ids = []         # unique ID for each chunk
        texts = []       # the raw text of each chunk
        embeddings = []  # the vector for each chunk
        metadatas = []   # extra info stored alongside each chunk

        for i, chunk in enumerate(chunks):

            # Create a unique ID: e.g. "american_robin_0", "american_robin_1"
            chunk_id = f"{species}_{i}"

            # Convert the chunk text to a vector
            # .encode() runs the embedding model and returns a numpy array
            # .tolist() converts it to a plain Python list for ChromaDB
            embedding = model.encode(chunk).tolist()

            ids.append(chunk_id)
            texts.append(chunk)
            embeddings.append(embedding)
            metadatas.append({
                "species": species,       # which bird this chunk is about
                "chunk_index": i,         # which chunk number within the document
                "chunk_count": len(chunks) # total chunks for this species
            })

        # Add all chunks for this species to ChromaDB in one batch
        collection.add(
            ids=ids,
            documents=texts,        # ChromaDB calls the text content "documents"
            embeddings=embeddings,
            metadatas=metadatas
        )

        total_chunks += len(chunks)

    print(f"\n  Done. {total_chunks} total chunks stored in ChromaDB.")
    return collection


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building vector index...\n")

    # Load all the .txt files
    documents = load_documents()

    # Chunk, embed, and store everything
    build_index(documents)

    print("\nVector index built successfully.")
    print(f"ChromaDB collection saved to: {CHROMA_DIR}/")