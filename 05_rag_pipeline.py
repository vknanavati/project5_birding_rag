# ============================================================
# 05_rag_pipeline.py
#
# PURPOSE: The full end-to-end RAG pipeline.
# Takes a question, retrieves relevant chunks, re-ranks them,
# sends everything to the Groq LLM, and returns a cited answer.
#
# This is the heart of the project — all previous scripts
# were building toward this one.
# ============================================================

import os                          # file paths and environment variables
from dotenv import load_dotenv     # reads our .env file for the API key
from groq import Groq              # Groq API client for LLM generation

# ── Import our retrieval and re-ranking functions ─────────────
# We built these already — now we just plug them together
from retrieval_03 import load_resources, hybrid_search
from rerank_04 import load_reranker, rerank

# ── Load environment variables from .env ──────────────────────
# This reads GROQ_API_KEY from the .env file and makes it
# available via os.environ — keeps secrets out of code
load_dotenv()

# ── Settings ──────────────────────────────────────────────────

# How many candidates to retrieve before re-ranking
TOP_K_RETRIEVE = 10

# How many top chunks to pass to the LLM after re-ranking
# We don't want to pass all 10 — just the most relevant ones
# Too many chunks = LLM gets confused or loses focus
TOP_K_CONTEXT = 3

# The Groq model to use for generation
# llama-3.1-8b-instant is fast, free, and very capable
GROQ_MODEL = "llama-3.1-8b-instant"

# Maximum tokens the LLM can generate in its response
MAX_TOKENS = 1024


# ══════════════════════════════════════════════════════════════
# LOAD THE GROQ CLIENT
#
# The Groq client handles communication with the Groq API.
# It reads GROQ_API_KEY automatically from the environment.
#
# Analogy: this is like opening a phone line to the LLM.
# Once the client is loaded we can send it as many questions
# as we want without reconnecting each time.
# ══════════════════════════════════════════════════════════════

def load_groq_client():
    """
    Initialize and return the Groq API client.
    Reads GROQ_API_KEY from environment variables.
    """
    # Get the API key from environment
    api_key = os.environ.get("GROQ_API_KEY")

    # If the key is missing, give a helpful error message
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Make sure it is set in your .env file."
        )

    # Create and return the Groq client
    client = Groq(api_key=api_key)
    print("Groq client loaded.\n")
    return client


# ══════════════════════════════════════════════════════════════
# BUILD THE PROMPT
#
# The prompt is the full text we send to the LLM. It contains:
#   1. A system message — tells the LLM its role and rules
#   2. The retrieved chunks — the evidence it must use
#   3. The user's question
#
# The quality of the prompt directly affects answer quality.
# We instruct the LLM to:
#   - Only use the provided context (no hallucination)
#   - Cite which chunk each fact came from
#   - Admit when it doesn't have enough information
#
# Analogy: we're giving an expert witness a stack of documents
# and telling them: "Answer only from these documents.
# Tell us which document each fact comes from.
# If the answer isn't here, say so."
# ══════════════════════════════════════════════════════════════

def build_prompt(question, chunks):
    """
    Build the system prompt and user message for the LLM.

    Arguments:
        question -- the user's question (string)
        chunks   -- list of (text, metadata, score) tuples from re-ranking

    Returns (system_prompt, user_message) as a tuple of strings.

    Plain English: assemble the evidence into a clear briefing
    document for the LLM, with instructions on how to use it.
    """

    # ── Build the context block ────────────────────────────────
    # Format each chunk as a numbered source with its species label
    # This is what the LLM will read as its evidence
    context_parts = []
    for i, (text, meta, score) in enumerate(chunks):
        # Format: [Source 1 - american_robin] followed by the chunk text
        source_label = f"[Source {i+1} - {meta['species']}]"
        context_parts.append(f"{source_label}\n{text}")

    # Join all sources with a separator between them
    context = "\n\n---\n\n".join(context_parts)

    # ── System prompt ──────────────────────────────────────────
    # This tells the LLM who it is and what rules to follow.
    # The system prompt is separate from the user message —
    # it sets the overall behavior for the entire conversation.
    system_prompt = """You are an expert birding assistant specializing in North American birds, particularly species found in Connecticut and Pennsylvania backyards.

You will be given a question and a set of numbered source passages from a birding knowledge base. Your job is to answer the question using ONLY the information provided in those sources.

Rules you must follow:
1. Base your answer exclusively on the provided sources. Do not use outside knowledge.
2. Cite your sources inline using [Source N] notation after each fact.
3. If the sources do not contain enough information to answer the question, say: "I don't have enough information in my knowledge base to answer that fully." Then share what partial information you do have.
4. Be specific and helpful — a backyard birder is asking this question and wants practical, accurate information.
5. Write in clear, friendly prose. Not bullet points unless listing multiple distinct items."""

    # ── User message ───────────────────────────────────────────
    # This is the actual question plus the retrieved evidence.
    # We put the context BEFORE the question so the LLM reads
    # the evidence first, then knows what to answer.
    user_message = f"""Here are the relevant sources from the birding knowledge base:

{context}

---

Question: {question}

Please answer the question using only the sources above. Cite each source inline."""

    return system_prompt, user_message


# ══════════════════════════════════════════════════════════════
# GENERATE ANSWER
#
# Send the prompt to the Groq LLM and get back a response.
# The LLM reads the context and question and writes an answer.
#
# Analogy: hand the briefing document to the expert witness
# and ask them to answer the question. They read the evidence
# and respond in their own words with citations.
# ══════════════════════════════════════════════════════════════

def generate_answer(question, chunks, groq_client):
    """
    Send the question and retrieved chunks to the Groq LLM.
    Returns the generated answer as a string.

    Plain English: package up the evidence and question,
    send it to the LLM, and return what it writes back.
    """

    # Build the prompt from the question and chunks
    system_prompt, user_message = build_prompt(question, chunks)

    # Send the request to Groq
    # messages is a list of turns in the conversation:
    #   - "system" sets the LLM's role and rules
    #   - "user" is the actual question with context
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ]
    )

    # Extract the text of the response from the API response object
    # response.choices is a list of possible responses (we only ask for 1)
    # .message.content is the actual text the LLM generated
    answer = response.choices[0].message.content
    return answer


# ══════════════════════════════════════════════════════════════
# FULL RAG PIPELINE
#
# This is the main function that connects all three stages:
#   1. Hybrid retrieval — find top candidates
#   2. Re-ranking — re-score with CrossEncoder
#   3. Generation — send to LLM, get cited answer
#
# Everything we built in scripts 02, 03, and 04 feeds into here.
# ══════════════════════════════════════════════════════════════

def rag_query(question, model, collection, all_chunks, reranker, groq_client):
    """
    Run the full RAG pipeline for a single question.

    Arguments:
        question     -- the user's question (string)
        model        -- the bi-encoder (SentenceTransformer)
        collection   -- the ChromaDB collection
        all_chunks   -- all chunks loaded for BM25
        reranker     -- the CrossEncoder re-ranking model
        groq_client  -- the Groq API client

    Returns a dict with:
        answer   -- the LLM's generated answer (string)
        sources  -- the chunks used as context (list of tuples)

    Plain English: take a question all the way from raw text
    to a cited answer in one function call.
    """

    print(f"\nSearching knowledge base...")

    # ── Step 1: Hybrid retrieval ───────────────────────────────
    # Get top 10 candidates using semantic + keyword search
    candidates = hybrid_search(
        question, model, collection, all_chunks,
        top_k=TOP_K_RETRIEVE
    )
    print(f"Retrieved {len(candidates)} candidates.")

    # ── Step 2: Re-ranking ─────────────────────────────────────
    # Re-score the top 10 using the CrossEncoder
    # Returns top TOP_K_CONTEXT chunks sorted by relevance
    top_chunks = rerank(question, candidates, reranker, top_k=TOP_K_CONTEXT)
    print(f"Re-ranked to top {len(top_chunks)} chunks.")

    # Show which species were selected as context
    species_used = [meta['species'] for _, meta, _ in top_chunks]
    print(f"Context sources: {species_used}")

    # ── Step 3: Generation ─────────────────────────────────────
    # Send the top chunks + question to the LLM
    print(f"Generating answer with {GROQ_MODEL}...")
    answer = generate_answer(question, top_chunks, groq_client)

    return {
        "answer": answer,
        "sources": top_chunks
    }


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":

    print("Loading RAG pipeline...\n")

    # Load all resources once at startup
    model, collection, all_chunks = load_resources()
    reranker                       = load_reranker()
    groq_client                    = load_groq_client()

    print("="*60)
    print("BIRDING RAG SYSTEM READY")
    print("Type your birding questions below.")
    print("Type 'quit' to exit.")
    print("="*60)

    # ── Interactive question loop ──────────────────────────────
    # Keep asking for questions until the user types 'quit'
    while True:

        # Get a question from the user
        print()
        question = input("Your question: ").strip()

        # Exit condition
        if question.lower() in ["quit", "exit", "q"]:
            print("Happy birding!")
            break

        # Skip empty input
        if not question:
            continue

        # Run the full RAG pipeline
        result = rag_query(
            question, model, collection,
            all_chunks, reranker, groq_client
        )

        # Print the answer
        print(f"\n{'='*60}")
        print("ANSWER:")
        print('='*60)
        print(result["answer"])
        print('='*60)