# Week 6 — Advanced RAG (Hybrid Search, Reranking, Contextual Retrieval)

> **Month 2 · Building with LLMs**
> *"Naive RAG breaks. This week is the four techniques that fix it and the eval method that proves it."*

---

## Why this week

Last week your RAG probably worked on 60–70% of your eval questions. The remaining 30–40% are why teams scrap demos when going to production. Four upgrades close most of that gap:

1. **Hybrid search** — vector similarity + BM25 keyword search (catches exact terms, acronyms, names)
2. **Reranking** — a cross-encoder reorders the top-N before sending to the LLM
3. **Contextual retrieval** (Anthropic, Sept 2024) — prepend a small context blurb to each chunk before embedding, so chunks aren't context-less
4. **Query rewriting / expansion** — rewrite the user query into something that retrieves better

You'll also learn how to **evaluate** RAG properly with Ragas, because shipping RAG without evals is shipping vibes.

---

## Learning objectives

By Sunday night you should be able to:

1. Implement hybrid retrieval (vector + BM25) with reciprocal rank fusion
2. Plug in a cross-encoder reranker (Cohere, BGE, or local) and measure the lift
3. Implement Anthropic's Contextual Retrieval and measure the lift
4. Use Ragas to compute retrieval precision/recall, faithfulness, and answer relevance
5. Explain when each technique helps and when it doesn't

---

## Day-by-day plan (≈ 2–3 hrs/day)

### Day 1 — Hybrid search

**Read (75 min):**
- [Pinecone — *Hybrid search*](https://www.pinecone.io/learn/hybrid-search-intro/)
- [Weaviate — *Hybrid search explained*](https://weaviate.io/blog/hybrid-search-explained)
- [Elastic — *RRF (Reciprocal Rank Fusion)*](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html) — the simplest, most common fusion algorithm

**Hands-on (60 min):**
- Add BM25 to your Week 5 pipeline using `rank_bm25` (or Qdrant's built-in sparse vectors via `bge-m3`)
- Implement reciprocal rank fusion:
  ```python
  def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
      scores = defaultdict(float)
      for ranking in rankings:
          for rank, doc_id in enumerate(ranking):
              scores[doc_id] += 1 / (k + rank)
      return sorted(scores, key=scores.get, reverse=True)
  ```
- Run your 10-question eval again with hybrid retrieval. Did retrieval@5 improve?

---

### Day 2 — Reranking

A bi-encoder (your embedding model) is fast but coarse. A **cross-encoder reranker** scores `(query, document)` pairs together, much more accurately. The pattern is universal: retrieve broad with the bi-encoder (top-50), rerank to top-5 with the cross-encoder.

**Read (60 min):**
- [Sentence Transformers — *Retrieve & Re-Rank*](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) — the canonical explainer
- [Cohere — *Rerank introduction*](https://docs.cohere.com/docs/rerank-2)
- [Pinecone — *Rerankers*](https://www.pinecone.io/learn/series/rag/rerankers/)

**Hands-on (90 min):**
- Add reranking to your pipeline. Pick one:
  - **Free / local:** `BAAI/bge-reranker-v2-m3` (cross-encoder) — runs on CPU for small batches
  - **Hosted:** Cohere Rerank-v3.5 (small free tier)
- Pipeline: retrieve top-50 (hybrid) → rerank to top-5 → send to LLM
- Re-run your eval. Lift?

---

### Day 3 — Contextual Retrieval (the Anthropic recipe)

A typical chunk like "It implements PageRank" is useless out of context — *what* implements PageRank? Anthropic's trick: for every chunk, prepend a short LLM-generated context blurb describing where in the doc it lives. Now the chunk reads "This section is part of Chapter 4 on graph algorithms. It implements PageRank…" and embeds far better.

**Read (60 min):**
- [Anthropic — *Introducing Contextual Retrieval*](https://www.anthropic.com/news/contextual-retrieval) — the original blog post
- [Anthropic — *Contextual Retrieval cookbook*](https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings)
- [DataCamp — *Contextual Retrieval: A Guide With Implementation*](https://www.datacamp.com/tutorial/contextual-retrieval-anthropic)

**Hands-on (90 min):**
- For each chunk, call Claude (or another cheap model) once at ingest time with:
  > *"Here is the full document. Here is one chunk. Write 1–2 sentences situating this chunk within the document."*
- Prepend the context to the chunk **before embedding**
- **Crucially**, use prompt caching on the document so this is cheap (Anthropic documents this in the post)
- Re-run your eval

According to Anthropic: contextual embeddings + contextual BM25 → 49% reduction in retrieval failures; with a reranker → 67% reduction.

---

### Day 4 — Query rewriting & expansion

Half of RAG failures are because the user's query doesn't look anything like the documents. *"How do I make it faster?"* doesn't retrieve documents that say *"optimize for low-latency inference."*

Techniques:
- **Query rewriting:** ask an LLM to rephrase the query in 2–3 different ways, retrieve for each, fuse
- **HyDE** (Hypothetical Document Embeddings): ask the LLM to *write a fake answer* and embed *that* instead of the query
- **Multi-query / sub-question decomposition:** decompose complex queries into atomic ones

**Read (60 min):**
- [LangChain — *Query construction & translation*](https://python.langchain.com/docs/concepts/retrieval/#query-translation) — overview of the techniques
- [HyDE paper — *Precise Zero-Shot Dense Retrieval without Relevance Labels*](https://arxiv.org/abs/2212.10496) (skim — the abstract + figure 1 are enough)
- [Anthropic — *Long-context prompting*](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/long-context-tips) (skim; useful when context is huge)

**Hands-on (60 min):**
- Implement one technique (HyDE is the easiest and works surprisingly well)
- Re-run your eval

---

### Day 5 — Evaluating RAG with Ragas

Up to this point you've been eyeballing. Today you actually measure.

**Read (60 min):**
- [Ragas — *Getting started*](https://docs.ragas.io/en/stable/getstarted/index.html)
- [Ragas — *Available metrics*](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/index.html) — focus on `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`
- [Ragas — *Test set generation*](https://docs.ragas.io/en/stable/concepts/test_data_generation/index.html) — auto-generates Q/A pairs from your docs

**Hands-on (~90 min):**
- Use Ragas to generate ~30 synthetic Q/A pairs from your corpus
- Hand-review them; toss the bad ones
- Run all 4 versions of your pipeline through Ragas:
  - Naive (Week 5)
  - + Hybrid + Rerank
  - + Contextual Retrieval
  - + Query rewriting
- Save the scores in a `eval/results.csv`

---

### Day 6 — Build the weekly project: integrate everything

See **Weekly Project** below.

---

### Day 7 — Polish, write retro

Today: clean up the repo, write the README with the headline chart, retro.

---

## Weekly Project — `advanced-rag`

Take your Week 5 `docs-rag` and turn it into a real-quality system. Make the gains measurable.

### Spec

A `compose.py`-style entrypoint that lets you toggle pipeline components:

```
$ advanced-rag eval --variants naive,hybrid,hybrid+rerank,contextual,contextual+rerank
variant                  context_precision  context_recall  faithfulness  answer_relevancy
naive                    0.62               0.51            0.78          0.71
hybrid                   0.71               0.66            0.82          0.78
hybrid+rerank            0.82               0.74            0.86          0.82
contextual               0.78               0.73            0.85          0.81
contextual+rerank        0.88               0.81            0.91          0.86  ← winner
```

### Requirements

- One codebase, 5 toggleable pipeline variants
- Hybrid retrieval: dense (BGE) + sparse (BM25 via `rank_bm25` or `bge-m3` sparse), fused with RRF
- Reranker: `bge-reranker-v2-m3` (free, local) **or** Cohere Rerank
- Contextual Retrieval implementation (per Anthropic recipe, with prompt caching if you use Claude)
- One query-rewriting variant (HyDE recommended)
- Ragas evaluation on at least 30 questions, results saved to `eval/results.csv`
- `report.md` with:
  - A bar chart of all variants on all metrics
  - **Where it broke** — 3 examples each variant got wrong
  - **What you'd do next** — what's the next thing to try?

### Stretch

- Add latency + cost to the eval table (an answer might be 0.05 better but 5× more expensive)
- Try a domain-specific embedding model (e.g., `jinaai/jina-embeddings-v2-base-code` if your corpus is code)
- Add a small Streamlit / Gradio UI

---

## Curated resources

**Anthropic Contextual Retrieval (your primary primary source)**
- [Anthropic — *Introducing Contextual Retrieval*](https://www.anthropic.com/news/contextual-retrieval) — read fully
- [Anthropic cookbook — *Contextual embeddings*](https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings) — runnable notebook
- [Together AI — *Implement contextual RAG from Anthropic*](https://docs.together.ai/docs/how-to-implement-contextual-rag-from-anthropic)
- [Towards Data Science — *Implementing Anthropic's Contextual Retrieval for Powerful RAG*](https://towardsdatascience.com/implementing-anthropics-contextual-retrieval-for-powerful-rag-performance-b85173a65b83/)

**Hybrid search & reranking**
- [Pinecone — *Hybrid search*](https://www.pinecone.io/learn/hybrid-search-intro/)
- [Pinecone — *Rerankers*](https://www.pinecone.io/learn/series/rag/rerankers/)
- [Weaviate — *Hybrid search explained*](https://weaviate.io/blog/hybrid-search-explained)
- [Sentence Transformers — *Retrieve & Re-Rank*](https://www.sbert.net/examples/applications/retrieve_rerank/README.html)
- [Cohere — *Rerank introduction*](https://docs.cohere.com/docs/rerank-2)
- [BAAI BGE Reranker collection](https://huggingface.co/collections/BAAI/bge-reranker-66c2c5dbc6e21d2dfedb1b87)

**Query rewriting / HyDE**
- [LangChain — *Query construction & translation*](https://python.langchain.com/docs/concepts/retrieval/#query-translation)
- [HyDE paper](https://arxiv.org/abs/2212.10496)

**Evaluation**
- [Ragas docs](https://docs.ragas.io/en/stable/)
- [Anthropic — *Evaluation* docs](https://docs.claude.com/en/docs/test-and-evaluate/eval-tool)
- [TruLens — *RAG triad*](https://www.trulens.org/getting_started/core_concepts/rag_triad/) — context relevance / groundedness / answer relevance

**Long-form**
- [DEV — *Advanced Retrieval Patterns That Actually Work*](https://dev.to/young_gao/rag-is-not-dead-advanced-retrieval-patterns-that-actually-work-in-2026-2gbo)
- [Analytics Vidhya — *Building Contextual RAG with Hybrid Search and Reranking*](https://www.analyticsvidhya.com/blog/2024/12/contextual-rag-systems-with-hybrid-search-and-reranking/)

---

## "Done when…" checklist

- [ ] Hybrid retrieval (dense + BM25) is in my pipeline and I can show its retrieval@5 lift vs vector-only
- [ ] A reranker is in place and I can show its lift vs no-reranker
- [ ] Contextual Retrieval is implemented and I can show its lift on at least one tricky chunk
- [ ] One query-rewriting technique (HyDE or multi-query) is in place
- [ ] Ragas runs end-to-end and I have a `results.csv` with at least 4 variants × 4 metrics
- [ ] My `report.md` has a chart and a failure analysis
- [ ] I have an opinion: which combination is best **for my data**, and why

---

## Common pitfalls this week

1. **Treating reranking as a universal cure.** It helps the most when your bi-encoder is weak or your top-50 is noisy. If your top-5 is already great, rerank lifts almost nothing.
2. **Contextual Retrieval without prompt caching.** Without caching, ingest costs explode. Cache the document.
3. **Believing one Ragas number.** Ragas has its own LLM-judge biases. Always look at the *examples* it scored low, not just the aggregate.
4. **Optimizing too early.** If your corpus is 200 pages of clean markdown, naive RAG is probably enough. Apply these upgrades when you've measured a real problem.

---

← Previous: [Week 5 — RAG Fundamentals](./WEEK-05.md) · → Next: [Week 7 — Fine-Tuning Theory + Dataset Preparation](./WEEK-07.md)
