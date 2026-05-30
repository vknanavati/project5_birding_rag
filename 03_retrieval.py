# ============================================================
# 03_retrieval.py
#
# PURPOSE: Search the ChromaDB vector index using hybrid
# retrieval — combining semantic search (meaning-based) with
# BM25 keyword search (word-matching-based).
#
# This is the retrieval layer of our RAG pipeline.
# Good retrieval = good answers. Bad retrieval = bad answers
# no matter how good the LLM is.
# ============================================================

import os                                        # file paths
from sentence_transformers import SentenceTransformer  # for embedding the query
import chromadb                                  # our vector database
from rank_bm25 import BM25Okapi                  # keyword search algorithm

# ── Settings ──────────────────────────────────────────────────

CHROMA_DIR       = "chroma_db"          # where ChromaDB is stored on disk
COLLECTION_NAME  = "birds"             # the collection we built in script 02
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"  # must match what we used in script 02
TOP_K_SEMANTIC   = 10   # how many results to fetch from semantic search
TOP_K_KEYWORD    = 10   # how many results to fetch from keyword search
TOP_K_FINAL      = 5    # how many results to return after combining both


# ══════════════════════════════════════════════════════════════
# LOAD RESOURCES
#
# We need two things loaded before we can search:
#   1. The embedding model — to convert the query to a vector
#   2. The ChromaDB collection — to search against
#
# We load these once and reuse them across multiple queries
# rather than reloading on every single search call.
# ══════════════════════════════════════════════════════════════

def load_resources():
    """
    Load the embedding model and ChromaDB collection.
    Returns (model, collection, all_chunks).

    Plain English: open the filing cabinet and pick up the
    magnifying glass — get everything ready before searching.
    """

    # Load the same embedding model we used to build the index
    # It's cached locally now so this loads in ~1 second
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Connect to the persisted ChromaDB on disk
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Get the birds collection we created in script 02
    collection = client.get_collection(name=COLLECTION_NAME)

    # Fetch ALL chunks from ChromaDB so we can run BM25 over them
    # BM25 is not built into ChromaDB — it needs the raw text
    # get() with no filters returns everything in the collection
    print("Loading all chunks for BM25 index...")
    all_data = collection.get(include=["documents", "metadatas"])

    # all_data["documents"] is a list of all chunk texts
    # all_data["metadatas"] is a list of all chunk metadata dicts
    all_chunks = list(zip(all_data["documents"], all_data["metadatas"]))
    # all_chunks is now: [("chunk text...", {"species": "robin", ...}), ...]

    print(f"Loaded {len(all_chunks)} chunks from ChromaDB.\n")
    return model, collection, all_chunks


# ══════════════════════════════════════════════════════════════
# SEMANTIC SEARCH
#
# Convert the query to a vector, then ask ChromaDB to find
# the chunks whose vectors are most similar (closest in
# meaning-space).
#
# Analogy: describe what you're looking for to someone who
# understands concepts, not just words. They find files that
# match your intent even if the exact words differ.
# ══════════════════════════════════════════════════════════════

def semantic_search(query, model, collection, top_k=TOP_K_SEMANTIC):
    """
    Find the top_k most semantically similar chunks to the query.
    Returns a list of (chunk_text, metadata, score) tuples.

    Plain English: convert the question to a vector, then find
    the chunks with the closest vectors in ChromaDB.
    """

    # Convert the query string into a vector
    # This is the same process we used on each chunk during indexing
    query_embedding = model.encode(query).tolist()

    # Ask ChromaDB for the top_k closest chunks
    # include=["documents", "metadatas", "distances"] tells ChromaDB
    # to return the text, metadata, and similarity score for each result
    results = collection.query(
        query_embeddings=[query_embedding],  # our query vector (list of one)
        n_results=top_k,                     # how many results to return
        include=["documents", "metadatas", "distances"]
    )

    # results["documents"][0] is a list of chunk texts
    # results["metadatas"][0] is a list of metadata dicts
    # results["distances"][0] is a list of cosine distances (lower = more similar)
    hits = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        # Convert distance to similarity score: 1.0 = perfect match, 0.0 = unrelated
        # ChromaDB returns cosine distance (0=identical, 2=opposite)
        # so we subtract from 1 to flip it into a similarity score
        score = 1 - dist
        hits.append((text, meta, score))

    return hits  # list of (text, metadata, similarity_score)


# ══════════════════════════════════════════════════════════════
# KEYWORD SEARCH (BM25)
#
# BM25 is a classic information retrieval algorithm that scores
# documents by how often query words appear in them, adjusted
# for document length and word rarity across the corpus.
#
# Analogy: a speed-reader who highlights every document
# containing your exact search words, ranking by how many
# times they appear and how rare those words are overall.
# ══════════════════════════════════════════════════════════════

def keyword_search(query, all_chunks, top_k=TOP_K_KEYWORD):
    """
    Find the top_k chunks with the highest BM25 keyword match score.
    Returns a list of (chunk_text, metadata, score) tuples.

    Plain English: find chunks that contain the exact words from
    the query, ranked by word frequency and rarity.
    """

    # Extract just the text from all_chunks for BM25 to index
    texts = [chunk[0] for chunk in all_chunks]  # chunk[0] is the text

    # Tokenize: split each chunk into individual words (lowercase)
    # BM25 works on lists of tokens, not raw strings
    tokenized_chunks = [text.lower().split() for text in texts]

    # Build the BM25 index from the tokenized chunks
    # This calculates word frequencies and document lengths
    bm25 = BM25Okapi(tokenized_chunks)

    # Tokenize the query the same way
    tokenized_query = query.lower().split()

    # Get BM25 scores for every chunk against this query
    # Higher score = better keyword match
    scores = bm25.get_scores(tokenized_query)

    # Pair each chunk with its score so we can sort them
    scored_chunks = [
        (texts[i], all_chunks[i][1], scores[i])  # (text, metadata, score)
        for i in range(len(texts))
    ]

    # Sort by score descending (highest score first) and return top_k
    scored_chunks.sort(key=lambda x: x[2], reverse=True)
    return scored_chunks[:top_k]


# ══════════════════════════════════════════════════════════════
# HYBRID SEARCH
#
# Combine semantic and keyword results using Reciprocal Rank
# Fusion (RRF) — a simple, effective method for merging two
# ranked lists into one.
#
# RRF works like this: instead of using raw scores (which are
# on different scales), each result gets a rank position in
# each list. The final score is based on rank position, not
# the original score. Results appearing high in BOTH lists
# score the highest.
#
# Analogy: two reviewers each rank 10 restaurants. Instead of
# averaging their scores (which use different rating scales),
# you look at rank position. A restaurant that both reviewers
# rank #1 beats one that only one reviewer likes.
# ══════════════════════════════════════════════════════════════

def hybrid_search(query, model, collection, all_chunks, top_k=TOP_K_FINAL):
    """
    Combine semantic and keyword search using Reciprocal Rank Fusion.
    Returns the top_k most relevant chunks as (text, metadata, score) tuples.

    Plain English: run both searches, then give each result a
    combined score based on how highly each search ranked it.
    Results that both searches agree on rise to the top.
    """

    # Run both searches
    semantic_hits  = semantic_search(query, model, collection)
    keyword_hits   = keyword_search(query, all_chunks)

    # ── Reciprocal Rank Fusion ─────────────────────────────────
    # RRF formula: score = 1 / (rank + k) where k=60 is a smoothing constant
    # A result ranked #1 gets 1/(1+60) = 0.016
    # A result ranked #10 gets 1/(10+60) = 0.014
    # Small differences in rank produce small score differences
    k = 60  # standard RRF smoothing constant
    rrf_scores = {}  # maps chunk_text → combined RRF score

    # Score semantic results by their rank position
    for rank, (text, meta, score) in enumerate(semantic_hits):
        if text not in rrf_scores:
            rrf_scores[text] = {"score": 0, "meta": meta}
        rrf_scores[text]["score"] += 1 / (rank + 1 + k)  # rank is 0-indexed

    # Score keyword results by their rank position and add to existing scores
    for rank, (text, meta, score) in enumerate(keyword_hits):
        if text not in rrf_scores:
            rrf_scores[text] = {"score": 0, "meta": meta}
        rrf_scores[text]["score"] += 1 / (rank + 1 + k)

    # Sort all results by their combined RRF score (highest first)
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    # Return the top_k results as (text, metadata, score) tuples
    final = []
    for text, data in sorted_results[:top_k]:
        final.append((text, data["meta"], data["score"]))

    return final


# ══════════════════════════════════════════════════════════════
# DISPLAY RESULTS
#
# A helper function to print search results cleanly so we can
# read them in the terminal and verify they make sense.
# ══════════════════════════════════════════════════════════════

def display_results(results, query):
    """
    Print search results in a readable format.
    """
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")

    for i, (text, meta, score) in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Species : {meta['species']}")
        print(f"Chunk   : {meta['chunk_index']} of {meta['chunk_count']}")
        print(f"Score   : {score:.4f}")
        # Print just the first 300 characters of the chunk as a preview
        print(f"Preview : {text[:300]}...")


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":

    # Load everything once
    model, collection, all_chunks = load_resources()

    # ── Test queries ───────────────────────────────────────────
    # These are sample questions to verify retrieval is working.
    # The right chunks should come back for each question.
    test_queries = [
        "What birds come to sunflower seed feeders in winter?",
        "How do I tell a House Finch from a Purple Finch?",
        "What does a Carolina Wren sound like?",
        "Which birds are only winter visitors in Connecticut?",
    ]

    for query in test_queries:
        results = hybrid_search(query, model, collection, all_chunks)
        display_results(results, query)
        print()