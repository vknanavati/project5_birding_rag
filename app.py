# ============================================================
# app.py
#
# PURPOSE: Flask API that wraps the full RAG pipeline.
# Accepts a POST request with a question, runs the complete
# retrieve → rerank → generate pipeline, and returns a
# cited answer as JSON.
#
# This is the final script — it exposes our birding RAG
# system as an HTTP endpoint that anything can call.
# ============================================================

import os                      # environment variables
from dotenv import load_dotenv # reads .env file
from flask import Flask, request, jsonify  # web framework
from flask_cors import CORS    # allows requests from browsers

# ── Import our pipeline components ────────────────────────────
from retrieval_03 import load_resources, hybrid_search
from rerank_04 import load_reranker, rerank
from rag_pipeline_05 import rag_query, load_groq_client

# ── Load environment variables ─────────────────────────────────
load_dotenv()

# ── Initialize Flask app ───────────────────────────────────────
app = Flask(__name__)  # create the Flask application
CORS(app)              # allow cross-origin requests from browsers

# ── Settings ───────────────────────────────────────────────────
PORT           = 5003   # port to run the API on
TOP_K_RETRIEVE = 10     # candidates before re-ranking
TOP_K_CONTEXT  = 5      # chunks passed to LLM after re-ranking


# ══════════════════════════════════════════════════════════════
# LOAD ALL RESOURCES AT STARTUP
#
# We load all three models once when the server starts,
# not on every request. This is critical for performance —
# loading models takes seconds; answering a question should
# take milliseconds.
#
# Analogy: a librarian doesn't go home and come back every
# time someone asks a question. They stay at the desk with
# all their tools ready, handling one request after another.
# ══════════════════════════════════════════════════════════════

print("Loading RAG pipeline components...")

# Load bi-encoder, ChromaDB collection, and all chunks for BM25
model, collection, all_chunks = load_resources()

# Load the CrossEncoder re-ranker
reranker = load_reranker()

# Load the Groq LLM client
groq_client = load_groq_client()

print("All components loaded. Server ready.\n")


# ══════════════════════════════════════════════════════════════
# HEALTH CHECK ENDPOINT
#
# GET /health
#
# A simple endpoint that returns "ok" to confirm the server
# is running. Useful for checking if the API is alive before
# sending real questions.
#
# Analogy: knocking on the door before walking in.
# ══════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """
    Health check — confirms the server is running.
    Returns: {"status": "ok"}
    """
    return jsonify({"status": "ok"})


# ══════════════════════════════════════════════════════════════
# ASK ENDPOINT
#
# POST /ask
#
# The main endpoint. Accepts a JSON body with a "question"
# field, runs the full RAG pipeline, and returns the answer
# with source citations.
#
# Request body:
#   {"question": "What does a Carolina Wren sound like?"}
#
# Response body:
#   {
#     "question": "What does a Carolina Wren sound like?",
#     "answer": "The Carolina Wren has a loud, ringing...",
#     "sources": [
#       {"species": "carolina_wren", "chunk_index": 0, ...},
#       ...
#     ]
#   }
#
# Analogy: you send a letter with your question, the system
# reads it, looks up the answer in the knowledge base, and
# sends back a written response with footnotes.
# ══════════════════════════════════════════════════════════════

@app.route("/ask", methods=["POST"])
def ask():
    """
    Main RAG endpoint. Accepts a question, returns a cited answer.
    """

    # ── Parse the request ──────────────────────────────────────
    # request.get_json() reads the JSON body of the POST request
    data = request.get_json()

    # Validate that a question was provided
    if not data or "question" not in data:
        # Return a 400 Bad Request error with a helpful message
        return jsonify({
            "error": "Missing 'question' field in request body."
        }), 400

    # Extract the question string
    question = data["question"].strip()

    # Don't process empty questions
    if not question:
        return jsonify({
            "error": "Question cannot be empty."
        }), 400

    # ── Run the RAG pipeline ───────────────────────────────────
    try:
        # This calls our full retrieve → rerank → generate pipeline
        result = rag_query(
            question,
            model,
            collection,
            all_chunks,
            reranker,
            groq_client
        )

        # ── Format the sources for the response ────────────────
        # Convert our internal (text, metadata, score) tuples
        # into clean JSON-friendly dicts for the API response
        sources = []
        for text, meta, score in result["sources"]:
            sources.append({
                "species":     meta["species"],      # which bird
                "chunk_index": meta["chunk_index"],  # which chunk
                "score":       round(score, 4),      # relevance score
                "preview":     text[:200]            # first 200 chars
            })

        # ── Return the response ────────────────────────────────
        return jsonify({
            "question": question,
            "answer":   result["answer"],
            "sources":  sources
        })

    # ── Handle errors gracefully ───────────────────────────────
    # If anything goes wrong in the pipeline, return a 500 error
    # with the error message rather than crashing the server
    except Exception as e:
        return jsonify({
            "error": f"Pipeline error: {str(e)}"
        }), 500


# ══════════════════════════════════════════════════════════════
# SPECIES ENDPOINT
#
# GET /species
#
# Returns a list of all bird species in the knowledge base.
# Useful for knowing what the system knows about.
# ══════════════════════════════════════════════════════════════

@app.route("/species", methods=["GET"])
def species():
    """
    Returns a list of all species in the knowledge base.
    """
    # Get all unique species names from the chunks metadata
    all_metadata = collection.get(include=["metadatas"])
    species_set = set()

    for meta in all_metadata["metadatas"]:
        # Add each species name to the set (sets auto-deduplicate)
        species_set.add(meta["species"])

    # Sort alphabetically and return as a list
    return jsonify({
        "species": sorted(list(species_set)),
        "count":   len(species_set)
    })


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting Birding RAG API on port {PORT}...")
    print(f"Health check : http://localhost:{PORT}/health")
    print(f"Ask endpoint : http://localhost:{PORT}/ask")
    print(f"Species list : http://localhost:{PORT}/species")
    print()

    # debug=False for stability
    # host="0.0.0.0" makes it accessible from other devices on your network
    app.run(host="0.0.0.0", port=PORT, debug=False)