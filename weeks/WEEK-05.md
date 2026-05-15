# Week 5 — RAG Fundamentals

> **Month 2 · Building with LLMs**
> *"A model only knows what was in its training data. RAG is how you teach it the things it doesn't know — at inference time, without retraining."*

---

## Why this week

Retrieval-Augmented Generation (RAG) is the single most commonly deployed pattern in production AI. Every "Chat with your docs / your codebase / your knowledge base" feature is RAG underneath. Real engineering teams spend more time on RAG quality than on prompting.

The naive version is simple: chunk docs → embed → store → on query, embed + nearest-neighbor → stuff the top-k into the prompt. The hard version (Week 6) is *making it actually work*. This week is the simple version, done properly.

---

## Learning objectives

By Sunday night you should be able to:

1. Explain the full RAG pipeline end-to-end: ingestion, chunking, embedding, indexing, retrieval, generation
2. Choose a chunking strategy and defend it (fixed-size, recursive, semantic)
3. Read the MTEB leaderboard and pick a reasonable embedding model for English / code / multilingual
4. Run a vector DB (Qdrant or Chroma) locally and use it from Python
5. Build a working "chat with my docs" system
6. Identify the 3 most common failure modes of naive RAG (and why we'll fix them in Week 6)

---

## Day-by-day plan (≈ 2–3 hrs/day)

### Day 1 — The mental model

**Read (60 min):**
- [Pinecone — *Retrieval Augmented Generation*](https://www.pinecone.io/learn/retrieval-augmented-generation/) — best plain-English intro
- [LangChain — *Retrieval augmented generation (RAG)*](https://python.langchain.com/docs/concepts/rag/) — the conceptual page (don't dive into the code yet)

**Watch (30 min):**
- [DeepLearning.AI — *Building and Evaluating Advanced RAG Applications* (free course, ~1 hr)](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — watch the first 1–2 lessons today

**Reflect:** Draw the RAG architecture on paper. Mark every place a thing could go wrong (it's at least 5 places).

---

### Day 2 — Chunking

Chunking is more impactful on retrieval quality than the choice of embedding model. People skip this. Don't.

**Read (90 min):**
- [Pinecone — *Chunking strategies*](https://www.pinecone.io/learn/chunking-strategies/) — the classic taxonomy
- [Greg Kamradt — *5 Levels of Text Splitting*](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb) — go from character → recursive → semantic chunking, with code
- [Chroma Research — *Evaluating Chunking Strategies for Retrieval*](https://research.trychroma.com/evaluating-chunking) — the 2024 empirically-grounded study; modern default reference
- [Hamel Husain — *Mistakes I see people make doing RAG*](https://hamel.dev/notes/llm/rag/) — the most-cited practitioner take; read once now and again after Week 6

**Hands-on (30 min):**
- Pick a markdown doc you actually care about (e.g., one of your project READMEs, or the FastAPI docs)
- Apply 3 chunkers: fixed-size (512 tokens), recursive character splitter, markdown-aware
- Inspect the chunks. Which one preserves headings? Which one breaks mid-sentence?

---

### Day 3 — Embeddings: choosing one without ceremony

You don't need to fine-tune an embedding model in Week 5. You need to **pick a good one off the shelf**.

**Read (60 min):**
- [Hugging Face — *MTEB Leaderboard*](https://huggingface.co/spaces/mteb/leaderboard) — open it. Use the **MMTEB / MTEB v2** tab (Borda over 131 tasks, 250+ languages); the single-aggregate-English leaderboard is now obsolete. Focus on **retrieval** as the column most correlated with RAG quality.
- [Hugging Face — *MMTEB: A Massive Multilingual Text Embedding Benchmark*](https://huggingface.co/blog/mteb) — the official write-up of the v2 methodology
- [Sentence Transformers — *Pretrained Models*](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) — the reference doc

**Defaults for the rest of the roadmap (May 2026):**

| Use case | Recommended model |
|---|---|
| English, fast, free, ~600MB | `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0, current open SoTA at this size) |
| English, top quality, free | `Qwen/Qwen3-Embedding-4B` or `nvidia/NV-Embed-v2` or `mixedbread-ai/mxbai-embed-large-v1` |
| Multilingual or long-context | `BAAI/bge-m3` (8k context, dense+sparse+ColBERT multi-vector) |
| You'll pay a tiny per-query cost | `text-embedding-3-large` (OpenAI), `voyage-3` (Voyage), `embed-multilingual-v3` (Cohere) |
| Code-specific | `Salesforce/SFR-Embedding-Code-400M_R` or `Qodo/Qodo-Embed-1-1.5B` or `jinaai/jina-embeddings-v3` (Jina v2-code is officially **deprecated**) |

**Hands-on (60 min):**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
emb = model.encode(["How do I sort a list?", "list.sort() in Python", "Apple stock price"])
# cosine-similarity between (0,1) should be high; (0,2) should be low
```

> Note: many embedding models (BGE, E5, Qwen3-Embedding) want instruction prefixes (`query: ...` / `passage: ...`). Always check the model card.

---

### Day 4 — Vector databases

Vector DBs are a crowded market. The honest truth: for almost everything in this roadmap, **Qdrant** (open-source, local-or-cloud) or **Chroma** (laptop-friendly) is plenty. Move to Pinecone / Weaviate / pgvector when you have a real scaling reason.

**Read (60 min):**
- [Qdrant — *Quickstart*](https://qdrant.tech/documentation/quickstart/)
- [Chroma — *Getting started*](https://docs.trychroma.com/getting-started)
- [Pinecone — *Vector databases explained*](https://www.pinecone.io/learn/vector-database/) (conceptual)
- [Weaviate — *Vector index comparison*](https://weaviate.io/blog/vector-search-explained) (skim — HNSW vs IVF vs Flat is good to know)

**Hands-on (60 min):**
- Run Qdrant locally:
  ```bash
  docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
  ```
- Index 50 documents (any small markdown corpus), search for 5 queries, eyeball quality

---

### Day 5 — Build the RAG pipeline end-to-end (Part 1)

You're going to build a working RAG, *without LangChain or LlamaIndex*, because today is about understanding every step. We'll let LangChain shine in Week 14 when we do agents.

**The pipeline you're building today:**
```
docs/ (markdown)
   │
   ▼
[chunker]
   │
   ▼
[embed → Qwen3-Embedding-0.6B]
   │
   ▼
[Qdrant collection]
   │
   ▼      query ──► [embed query] ──► [top-k from Qdrant]
[generator: prompt = system + context + query]  ──►  answer
```

> **2026 preview:** late-interaction retrieval (ColBERTv2 via [RAGatouille](https://github.com/AnswerDotAI/RAGatouille) or PyLate) often beats bi-encoder retrieval out-of-the-box. We'll cover it as a first-class technique in Week 6; for now, build the bi-encoder baseline.

**Hands-on (~2 hr):**
- Ingest: walk a `docs/` folder, chunk each markdown file (recursive splitter, ~500 token chunks with 50 overlap), embed, upsert into Qdrant
- Query: embed → search top-5 → format prompt → call LLM → print answer + cited chunk IDs

You should have a working `python rag.py "How do I do X?"` by end of day.

---

### Day 6 — Build the RAG pipeline (Part 2): LlamaIndex / LangChain

You did it from scratch. Now do it again with a framework — so you know what they buy you (less code) and what they cost (less control).

**Read (45 min):**
- [LlamaIndex — *Build a Q&A app over your data*](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/)
- [LlamaIndex — *Building RAG from Scratch (Open-source)*](https://developers.llamaindex.ai/python/examples/low_level/oss_ingestion_retrieval/)

**Hands-on (~90 min):**
- Rebuild the same pipeline using LlamaIndex (or LangChain — pick one). Compare LOC.
- Most importantly: write a side-by-side note in your README about what you gained and what you lost vs the from-scratch version.

---

### Day 7 — Project + retro

Ship the weekly project (next section), write a short retro.

---

## Weekly Project — `docs-rag`

A "chat with my docs" CLI/API that:
- Ingests a folder of markdown / `.txt` / `.py` files
- Stores embeddings in Qdrant (or Chroma)
- Answers questions with cited source chunks

### Spec

```
$ docs-rag ingest ./fastapi-docs/
[+] Ingested 412 chunks from 87 files in 14.2s

$ docs-rag ask "How do I add CORS middleware?"
ANSWER:
Use the CORSMiddleware class. In your FastAPI app:

    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"])

SOURCES:
  [1] tutorial/cors.md  (similarity: 0.81)
  [2] advanced/middleware.md  (similarity: 0.74)
```

### Requirements

- Two versions in the same repo: `rag_scratch.py` (no framework) and `rag_llamaindex.py` (or `rag_langchain.py`)
- Chunking: recursive character splitter, 400–600 tokens, 50–100 overlap
- Embedding: `Qwen/Qwen3-Embedding-0.6B` (or `mixedbread-ai/mxbai-embed-large-v1` if you have a bit more VRAM)
- Vector DB: Qdrant (Docker) or Chroma (local persistent)
- Generator: any LLM (OpenAI, Anthropic, or local via Ollama / your Week 3 code-completer)
- Citations: every answer must list which chunks were retrieved + similarity scores
- A small eval (~10 questions where you know the right answer + which doc has it). For each, measure:
  - Retrieval@5 — did the correct chunk appear in the top-5?
  - Faithfulness — did the answer use only information present in the retrieved chunks? (do this manually for now, automate in Week 11)
- **A `failures.jsonl` log:** every question that failed retrieval or faithfulness, with reason. This becomes the test set for Week 6 upgrades.
- `README.md` with: setup, examples, a section called **"What naive RAG fails at"** listing the 3 worst failure cases you found

### Stretch

- Add markdown-aware splitting that respects headings
- Support PDF ingestion via `pymupdf` or `pypdf`
- Add an `--explain` flag that prints the embedding similarity for *every* candidate chunk (useful when debugging "why didn't it find that?")

---

## Curated resources

**Tutorials (your primary reading)**
- [Pinecone Learn — *Retrieval Augmented Generation*](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Pinecone Learn — *Chunking strategies*](https://www.pinecone.io/learn/chunking-strategies/)
- [Greg Kamradt — *5 Levels of Text Splitting* (notebook)](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)
- [Superlinked VectorHub — *Evaluation of RAG Retrieval Chunking Methods*](https://superlinked.com/vectorhub/articles/evaluation-rag-retrieval-chunking-methods)
- [LlamaIndex — *Getting Started*](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/)

**Embeddings**
- [MTEB Leaderboard (use the MMTEB / v2 tab)](https://huggingface.co/spaces/mteb/leaderboard)
- [HuggingFace — *MMTEB methodology blog*](https://huggingface.co/blog/mteb)
- [Sentence Transformers — *Pretrained Models*](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- [Qwen3-Embedding collection](https://huggingface.co/collections/Qwen/qwen3-embedding) — current SoTA open small/mid
- [Nomic — *nomic-embed-text-v2-moe*](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
- [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
- [BAAI/bge-m3 (dense + sparse + ColBERT multi-vector)](https://huggingface.co/BAAI/bge-m3)

**Vector DBs**
- [Qdrant docs — *Quickstart*](https://qdrant.tech/documentation/quickstart/)
- [Chroma docs — *Getting started*](https://docs.trychroma.com/getting-started)
- [Pinecone — *What is a vector database?*](https://www.pinecone.io/learn/vector-database/)
- [Weaviate — *Vector search explained*](https://weaviate.io/blog/vector-search-explained)

**Practitioner reading (don't skip)**
- [Hamel Husain — *Mistakes I see people make doing RAG*](https://hamel.dev/notes/llm/rag/) — the 2024–2026 practitioner reference
- [Jason Liu — *RAG writing*](https://jxnl.co/writing/category/rag/) — query understanding, evals, segmentation; widely cited
- [Chroma Research — *Evaluating Chunking Strategies*](https://research.trychroma.com/evaluating-chunking)

**Courses (free, optional)**
- [DeepLearning.AI — *Advanced Retrieval for AI with Chroma*](https://learn.deeplearning.ai/courses/advanced-retrieval-for-ai-with-chroma)
- [DeepLearning.AI — *Vector Databases: from Embeddings to Applications*](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/)

---

## "Done when…" checklist

- [ ] I can explain every component of the RAG pipeline and a failure mode each one has
- [ ] I built RAG once from scratch and once with LlamaIndex/LangChain and wrote down the tradeoff
- [ ] My `docs-rag` repo is on GitHub with ingestion + query + citations
- [ ] I evaluated retrieval@5 on at least 10 questions
- [ ] I can list the top-3 failure modes of naive RAG I personally experienced this week (we'll fix them in Week 6)
- [ ] I've written a retro

---

## Common pitfalls this week

1. **Chunk too small** → context fragments lose meaning. **Chunk too big** → embeddings get muddy, retrieval gets worse. Aim for 400–600 tokens to start.
2. **Embedding model mismatch** — using `bge` for English text is fine; using it for code or Japanese is not. Match model to data.
3. **Forgetting the embedding model's instruction prefix.** Many models (BGE, E5) want `"query: ..."` prepended to queries and `"passage: ..."` to documents. Check the model card.
4. **Cosine vs dot-product vs L2.** Sentence-Transformers normalize embeddings; use cosine. If you skip normalization, your similarities lie.
5. **Reaching for LangChain on Day 1.** You'll learn ten times more by doing it from scratch first.

---

← Previous: [Week 4 — Prompting & Structured Outputs](./WEEK-04.md) · → Next: [Week 6 — Advanced RAG (Hybrid Search, Reranking, Contextual Retrieval)](./WEEK-06.md)
