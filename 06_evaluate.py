# ============================================================
# 06_evaluate.py
#
# PURPOSE: Compare RAG answers against plain LLM answers
# (no retrieval) on the same set of questions.
#
# This is the benchmark that proves RAG's value. We ask the
# same questions two ways and compare the results side by side.
#
# Key insight: a plain LLM answers from memory — it may be
# correct, vague, or hallucinated. A RAG system answers from
# your specific knowledge base — grounded and cited.
# ============================================================

import os                      # environment variables
from dotenv import load_dotenv # reads .env file
from groq import Groq          # Groq API client

# ── Import our full RAG pipeline ───────────────────────────────
from retrieval_03 import load_resources, hybrid_search
from rerank_04 import load_reranker, rerank
from rag_pipeline_05 import rag_query, build_prompt, load_groq_client

# ── Load environment variables ─────────────────────────────────
load_dotenv()

# ── Settings ───────────────────────────────────────────────────
GROQ_MODEL     = "llama-3.1-8b-instant"  # same model for fair comparison
MAX_TOKENS     = 1024                     # same token limit for both
TOP_K_RETRIEVE = 10                       # candidates before re-ranking
TOP_K_CONTEXT  = 3                        # chunks passed to LLM


# ══════════════════════════════════════════════════════════════
# PLAIN LLM ANSWER (NO RETRIEVAL)
#
# Ask the LLM the question directly with no context provided.
# This simulates what you'd get from a plain chatbot with no
# knowledge base — the LLM answers purely from training memory.
#
# Analogy: asking a knowledgeable friend your question off the
# top of their head, with no books or references in front of them.
# ══════════════════════════════════════════════════════════════

def ask_llm_only(question, groq_client):
    """
    Ask the LLM the question with no retrieved context.
    Returns the answer as a string.

    Plain English: send the question straight to the LLM
    with no supporting documents — just raw memory.
    """

    # A simple system prompt — no RAG instructions, no sources
    system_prompt = """You are a helpful birding assistant
    specializing in North American birds, particularly species
    found in Connecticut and Pennsylvania backyards."""

    # Just the question — no context chunks attached
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question}
        ]
    )

    return response.choices[0].message.content


# ══════════════════════════════════════════════════════════════
# RAG ANSWER
#
# Run the full pipeline: retrieve → rerank → generate.
# The LLM sees the same question but now has retrieved
# evidence from our knowledge base to work from.
#
# Analogy: asking the same knowledgeable friend but this time
# they have your specific field guides open in front of them
# and must cite which page each fact comes from.
# ══════════════════════════════════════════════════════════════

def ask_with_rag(question, model, collection, all_chunks, reranker, groq_client):
    """
    Run the full RAG pipeline for a question.
    Returns a dict with 'answer' and 'sources'.

    Plain English: retrieve relevant chunks, re-rank them,
    then generate a grounded cited answer.
    """
    result = rag_query(
        question, model, collection,
        all_chunks, reranker, groq_client
    )
    return result


# ══════════════════════════════════════════════════════════════
# DISPLAY COMPARISON
#
# Print both answers side by side in a clear format so we
# can read and compare them easily.
# ══════════════════════════════════════════════════════════════

def display_comparison(question, llm_answer, rag_result):
    """
    Print the plain LLM answer and RAG answer side by side.
    """

    print("\n" + "="*60)
    print(f"QUESTION: {question}")
    print("="*60)

    # ── Plain LLM answer ───────────────────────────────────────
    print("\n--- WITHOUT RAG (plain LLM, no retrieval) ---")
    print(llm_answer)

    # ── RAG answer ─────────────────────────────────────────────
    print("\n--- WITH RAG (retrieved context + citations) ---")

    # Show which species were used as sources
    species_used = [meta['species'] for _, meta, _ in rag_result['sources']]
    print(f"[Sources retrieved: {', '.join(species_used)}]")
    print()
    print(rag_result['answer'])
    print("="*60)


# ══════════════════════════════════════════════════════════════
# EVALUATION QUESTIONS
#
# These questions are designed to show where RAG helps most:
#
# 1. Questions specific to CT/PA — the LLM may answer
#    generically; RAG answers from your specific knowledge base
#
# 2. Comparison questions — RAG pulls the exact comparison
#    sections we wrote; the LLM may conflate the two species
#
# 3. Attracting birds — RAG gives specific feeder/plant advice
#    from your documents; the LLM gives generic advice
#
# 4. A trick question — neither should answer well, but the
#    RAG system should admit it more cleanly
# ══════════════════════════════════════════════════════════════

EVALUATION_QUESTIONS = [
    # Specific identification question — tests comparison retrieval
    "What is the difference between a Downy Woodpecker and a Hairy Woodpecker?",

    # Seasonal/geographic question — tests CT/PA specific knowledge
    "Which birds in my Connecticut backyard are only present in winter?",

    # Practical feeder question — tests attracting advice
    "What is the single best food I can put out to attract the most bird species?",

    # Behavior question — tests specific behavioral knowledge
    "Which backyard birds cache or hide their food for later?",
]


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":

    print("Loading evaluation pipeline...\n")

    # Load all resources
    model, collection, all_chunks = load_resources()
    reranker                       = load_reranker()
    groq_client                    = load_groq_client()

    print("\nRunning evaluation — comparing RAG vs plain LLM...\n")
    print("This will ask each question twice: once with RAG, once without.")
    print("="*60)

    # Run each evaluation question
    for question in EVALUATION_QUESTIONS:

        # Get plain LLM answer (no retrieval)
        print(f"\nAsking LLM only: {question[:50]}...")
        llm_answer = ask_llm_only(question, groq_client)

        # Get RAG answer (full pipeline)
        print(f"Running RAG pipeline...")
        rag_result = ask_with_rag(
            question, model, collection,
            all_chunks, reranker, groq_client
        )

        # Display both answers side by side
        display_comparison(question, llm_answer, rag_result)
        print()