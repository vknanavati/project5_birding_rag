# ============================================================
# 04_rerank.py
#
# PURPOSE: Take the top chunks from hybrid retrieval and
# re-score them using a CrossEncoder model that reads the
# query and each chunk together simultaneously.
#
# This is the re-ranking layer of our RAG pipeline.
# It fixes ranking errors that the bi-encoder retrieval
# makes because it never read query and chunk together.
# ============================================================

from sentence_transformers import CrossEncoder  # the re-ranking model
import os                                        # file paths

# ── Import our retrieval functions from script 03 ─────────────
# Rather than rewriting everything, we import the functions
# we already built. This is why keeping code modular matters.
from retrieval_03 import load_resources, hybrid_search

# ── Settings ──────────────────────────────────────────────────

# The CrossEncoder model for re-ranking
# ms-marco is a dataset of real search queries — this model was
# trained on it specifically for passage relevance scoring
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates to retrieve before re-ranking
# We get more than we need so the re-ranker has good options to choose from
TOP_K_RETRIEVE = 10

# How many final results to return after re-ranking
TOP_K_FINAL = 5


# ══════════════════════════════════════════════════════════════
# LOAD THE CROSS-ENCODER
#
# The CrossEncoder is a separate model from the bi-encoder
# we used for retrieval. It's specifically trained to score
# how relevant a passage is to a query when read together.
#
# Analogy: if the bi-encoder is a resume screener who reads
# applications independently, the CrossEncoder is the
# interviewer who sits with both the job description AND
# the candidate's answers at the same time.
# ══════════════════════════════════════════════════════════════

def load_reranker():
    """
    Load the CrossEncoder re-ranking model.
    Downloads on first run (~80MB), cached locally after that.
    """
    print(f"Loading re-ranking model: {RERANK_MODEL}")
    reranker = CrossEncoder(RERANK_MODEL)
    print("Re-ranking model loaded.\n")
    return reranker


# ══════════════════════════════════════════════════════════════
# RE-RANK
#
# For each candidate chunk, create a (query, chunk) pair and
# pass it to the CrossEncoder. The model reads both together
# and outputs a relevance score.
#
# Higher score = more relevant to the query.
# The scores are raw logits — they can be any number, positive
# or negative. We sort by score descending to get best first.
#
# Analogy: after the resume screener picks 10 finalists,
# the interviewer sits with each one and scores them on how
# well they actually answer the job requirements. The scores
# are absolute judgments, not just relative rankings.
# ══════════════════════════════════════════════════════════════

def rerank(query, candidates, reranker, top_k=TOP_K_FINAL):
    """
    Re-score retrieved chunks using the CrossEncoder.

    Arguments:
        query      -- the user's question (string)
        candidates -- list of (text, metadata, score) from hybrid_search
        reranker   -- the loaded CrossEncoder model
        top_k      -- how many results to return after re-ranking

    Returns a list of (text, metadata, score) tuples,
    sorted by CrossEncoder score descending.

    Plain English: read each candidate chunk alongside the query
    and assign a fresh relevance score based on how well they
    actually match each other — not just how similar their
    vectors were.
    """

    # Build a list of (query, chunk_text) pairs for the CrossEncoder
    # The model needs to see both together to score them
    pairs = [(query, candidate[0]) for candidate in candidates]
    # candidate[0] is the chunk text (index 0 of the tuple)

    # Run the CrossEncoder on all pairs at once
    # .predict() returns a numpy array of scores, one per pair
    scores = reranker.predict(pairs)

    # Attach each score back to its corresponding candidate
    reranked = []
    for i, (text, meta, old_score) in enumerate(candidates):
        reranked.append((
            text,
            meta,
            float(scores[i])  # the new CrossEncoder score
        ))

    # Sort by new CrossEncoder score, highest first
    reranked.sort(key=lambda x: x[2], reverse=True)

    # Return only the top_k results
    return reranked[:top_k]


# ══════════════════════════════════════════════════════════════
# DISPLAY RESULTS
#
# Print re-ranked results so we can compare them against the
# original retrieval order and see what changed.
# ══════════════════════════════════════════════════════════════

def display_reranked(results, query):
    """
    Print re-ranked results in a readable format.
    CrossEncoder scores can be any number — higher is better.
    Typical range is roughly -10 to +10.
    """
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")

    for i, (text, meta, score) in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Species : {meta['species']}")
        print(f"Chunk   : {meta['chunk_index']} of {meta['chunk_count']}")
        print(f"Score   : {score:.4f}")
        print(f"Preview : {text[:300]}...")


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Load retrieval resources ───────────────────────────────
    # Load the bi-encoder, ChromaDB collection, and all chunks
    model, collection, all_chunks = load_resources()

    # ── Load the re-ranker ─────────────────────────────────────
    reranker = load_reranker()

    # ── Test queries ───────────────────────────────────────────
    # Same queries as script 03 so we can directly compare
    # what changed after re-ranking
    test_queries = [
        "What birds come to sunflower seed feeders in winter?",
        "How do I tell a House Finch from a Purple Finch?",
        "What does a Carolina Wren sound like?",
        "Which birds are only winter visitors in Connecticut?",
    ]

    for query in test_queries:

        # Step 1 — retrieve top 10 candidates using hybrid search
        candidates = hybrid_search(
            query, model, collection, all_chunks, top_k=TOP_K_RETRIEVE
        )

        # Step 2 — re-rank those 10 candidates using CrossEncoder
        results = rerank(query, candidates, reranker)

        # Step 3 — display the re-ranked results
        display_reranked(results, query)
        print()