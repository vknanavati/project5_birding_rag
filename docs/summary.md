# Project 5 — Summary

## What We Built

A production-style **Retrieval Augmented Generation (RAG)** system for identifying and learning about backyard birds in Connecticut and Pennsylvania. The system accepts natural language questions and returns grounded, cited answers drawn from a hand-crafted knowledge base of 21 East Coast bird species.

**Example questions it can answer:**
- *"What does a Carolina Wren sound like?"*
- *"How do I tell a House Finch from a Purple Finch?"*
- *"Which birds cache food for later?"*
- *"What should I plant to attract Cedar Waxwings?"*

---

## The Problem RAG Solves

Plain LLMs have two critical weaknesses: a knowledge cutoff date and a tendency to hallucinate. RAG fixes both by giving the LLM a real document collection to search before answering. Instead of relying on memory, the model reads retrieved evidence and cites exactly where each fact came from.

---

## The Full Pipeline

| Stage | Script | What It Does |
|---|---|---|
| Knowledge base | `01_build_knowledge_base.py` | Writes 21 detailed bird documents covering identification, habitat, behavior, diet, and attraction tips |
| Indexing | `02_chunk_and_embed.py` | Splits documents into overlapping chunks, embeds them with `all-MiniLM-L6-v2`, stores in ChromaDB |
| Retrieval | `retrieval_03.py` | Hybrid search combining semantic (bi-encoder) and keyword (BM25) results via Reciprocal Rank Fusion |
| Re-ranking | `rerank_04.py` | CrossEncoder re-scores top 10 candidates by reading query and chunk together simultaneously |
| Generation | `rag_pipeline_05.py` | Sends top chunks + question to Groq LLM with citation instructions, returns grounded answer |
| Evaluation | `06_evaluate.py` | Compares RAG answers against plain LLM answers on 4 benchmark questions |
| API | `app.py` | Flask API with `/ask`, `/health`, and `/species` endpoints |

---

## Key Technical Decisions

**Why hybrid search instead of semantic only?**
Semantic search finds conceptually related chunks even when exact words differ. Keyword search (BM25) finds exact species name matches that semantic search sometimes misses. Combining both via Reciprocal Rank Fusion catches more relevant content than either method alone.

**Why a CrossEncoder re-ranker?**
The bi-encoder used for retrieval encodes query and chunks independently — it never reads them together. The CrossEncoder reads the query and each candidate chunk simultaneously, catching relevance mismatches the bi-encoder misses. It's too slow to run on all 43 chunks, so it only re-scores the top 10 candidates.

**Why Groq instead of Anthropic or OpenAI?**
Free tier with no credit card required, very fast inference, and capable open-source models. Perfect for a learning project.

**Why `TOP_K_CONTEXT = 5` instead of 3?**
Initial evaluation with 3 chunks caused retrieval misses on questions requiring knowledge spread across multiple species. Increasing to 5 improved coverage while keeping the prompt focused.

---

## What the Evaluation Showed

| Question | RAG better? | Why |
|---|---|---|
| Downy vs Hairy Woodpecker | ✅ Yes | Specific cited facts, no hallucination |
| Winter-only birds in CT | ❌ No | Retrieval missed Dark-eyed Junco chunk |
| Best food to attract birds | ✅ Yes | Grounded in specific species preferences |
| Which birds cache food | ✅ Yes | Correctly cited Titmouse, Nuthatch, Jay, Crow |

RAG clearly wins on specific identification and feeder questions. It struggles when the right chunks don't get retrieved — confirming the key principle: **retrieval quality matters more than generation quality.**

---

## New Concepts Learned

**Embedding** — Converting text into a vector (list of numbers) that captures meaning. Similar meanings produce similar vectors, enabling search by concept rather than exact words.

**Vector store (ChromaDB)** — A database designed for storing and searching vectors by similarity rather than exact match.

**Hybrid search** — Combining semantic search (meaning-based) and keyword search (word-matching) for better retrieval coverage than either method alone.

**Reciprocal Rank Fusion (RRF)** — A method for combining two ranked lists by rank position rather than raw score, so results appearing highly in both lists rise to the top.

**Bi-encoder vs CrossEncoder** — Bi-encoders encode query and document independently (fast, scalable). CrossEncoders read both together (slower, more accurate). RAG uses both: bi-encoder for broad retrieval, CrossEncoder for precise re-ranking.

**Prompt engineering for RAG** — Instructing the LLM to answer only from provided sources, cite inline, and admit when context is insufficient — the key to grounded, honest answers.

---

## The Most Important Thing to Remember

> **The generator can only work with what the retriever gives it.**
>
> A perfect LLM with bad retrieval produces bad answers.
> A decent LLM with excellent retrieval produces excellent answers.
>
> When a RAG system gives a wrong answer, look at retrieval first — not the LLM.

---

## Knowledge Base

21 East Coast backyard bird species covering Connecticut and Pennsylvania:

American Robin, Black-capped Chickadee, Carolina Wren, Downy Woodpecker, American Goldfinch, Northern Cardinal, House Finch, Purple Finch, White-breasted Nuthatch, Tufted Titmouse, Dark-eyed Junco, Song Sparrow, Mourning Dove, American Crow, Blue Jay, European Starling, House Sparrow, Red-bellied Woodpecker, Cedar Waxwing, Eastern Towhee, Common Grackle