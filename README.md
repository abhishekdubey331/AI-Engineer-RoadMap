# The 4-Month AI Engineer Roadmap

> **Deep, day-wise roadmap to go from "I know Python" to "I can ship production-grade LLM systems, code models, and AI agents."**

This is a compressed, deep-learning-focused version of the classic 6-month AI engineering path. It assumes you already have software-engineering fundamentals (Python, Git, Linux, FastAPI basics, a feel for ML). If you don't, start with [PREREQUISITES.md](./PREREQUISITES.md) before Week 1.

The roadmap is built around three principles:

1. **Build, don't just read.** Every week ends with a working artifact.
2. **Evaluate, don't vibe-check.** Every model change has a measurable eval.
3. **Top-tier resources only.** Every link in this roadmap has been picked from the best educators in the field — Karpathy, 3Blue1Brown, Sebastian Raschka, Lilian Weng, Anthropic, Hugging Face, vLLM, the original paper authors.

---

## Who this is for

You want to become the kind of AI engineer who can:

- Own a training-to-inference pipeline for a code or domain model
- Optimize inference with quantization, caching, and smart routing
- Fine-tune domain-specific LLMs (code generation, refactoring, reasoning)
- Design AI pipelines: prompting, retrieval, agents, evaluation
- Build agentic systems with multi-step tool calling
- Establish standards for benchmarking, evaluation, and observability
- Ship AI features end-to-end

This is **not** a "prompt engineer" track. It's a software-engineer-becomes-AI-engineer track.

---

## What you'll have built by the end

| # | Artifact | Week |
|---|---|---|
| 1 | A from-scratch BPE tokenizer | 1 |
| 2 | A nano-GPT trained on your own corpus | 2 |
| 3 | A local code-completion CLI on an open-weight model | 3 |
| 4 | A structured code-review bot (JSON schema, retries) | 4 |
| 5 | A docs-aware RAG assistant | 5 |
| 6 | An advanced RAG with hybrid search + reranking + contextual retrieval | 6 |
| 7 | An instruction dataset for a narrow code task | 7 |
| 8 | A LoRA-fine-tuned code model (with before/after eval) | 8 |
| 9 | A QLoRA vs LoRA vs base-model quantization benchmark | 9 |
| 10 | A vLLM-served inference API with streaming | 10 |
| 11 | A HumanEval-style code-eval harness with sandboxed execution | 11 |
| 12 | A production-style AI gateway (caching, routing, rate limits) | 12 |
| 13 | A tool-using code assistant (function calling, safety) | 13 |
| 14 | A LangGraph multi-step coding agent | 14 |
| 15 | An agent eval suite + Langfuse-style observability layer | 15 |
| 16 | **Capstone: Design-to-Code Agent** | 16 |

Pin these on GitHub. They are your portfolio.

---

## How the roadmap is structured

```
Month 1  Foundations of LLMs        →  Weeks 1–4
Month 2  Building with LLMs         →  Weeks 5–8
Month 3  Production LLM Systems     →  Weeks 9–12
Month 4  Agents, Eval, Capstone     →  Weeks 13–16
```

Each week has its own file in [`weeks/`](./weeks/) with:

- **Theme + learning objectives**
- **A day-by-day plan** (7 days, ~2–3 hrs/day, ~15–20 hrs/week)
- **Curated theory** — the best articles, papers, and videos
- **A hands-on lab** — short code exercises during the week
- **A weekly project** — the deliverable that goes on your GitHub
- **A "Done when…" checklist** so you know when to move on

---

## The 16-week map

### Month 1 — Foundations of LLMs

| Week | Topic | Project |
|------|-------|---------|
| [01](./weeks/WEEK-01.md) | Tokenization & Embeddings (BPE from scratch) | `minbpe`-style tokenizer + token-budget tool |
| [02](./weeks/WEEK-02.md) | The Transformer (attention, decoder-only) | Train a nano-GPT on a tiny corpus |
| [03](./weeks/WEEK-03.md) | Modern LLM internals + Hugging Face hands-on | Local code-completion CLI |
| [04](./weeks/WEEK-04.md) | Prompting, structured outputs, function calling | Structured code-review bot |

### Month 2 — Building with LLMs

| Week | Topic | Project |
|------|-------|---------|
| [05](./weeks/WEEK-05.md) | RAG fundamentals (chunking, embeddings, vector DBs) | Docs-aware code assistant |
| [06](./weeks/WEEK-06.md) | Advanced RAG (hybrid search, reranking, contextual retrieval) | Production-quality RAG over a real codebase |
| [07](./weeks/WEEK-07.md) | Fine-tuning theory + dataset preparation | Instruction dataset (1k–5k examples) |
| [08](./weeks/WEEK-08.md) | LoRA fine-tuning hands-on | Fine-tuned small code model + before/after eval |

### Month 3 — Production LLM Systems

| Week | Topic | Project |
|------|-------|---------|
| [09](./weeks/WEEK-09.md) | QLoRA + quantization (GPTQ, AWQ, GGUF) | LoRA vs QLoRA vs quantized benchmark |
| [10](./weeks/WEEK-10.md) | Inference servers (vLLM, PagedAttention, KV cache) | vLLM-backed streaming inference API |
| [11](./weeks/WEEK-11.md) | Code evaluation (HumanEval, SWE-bench, pass@k) | Sandboxed code-eval harness |
| [12](./weeks/WEEK-12.md) | Caching, routing, production API hardening | AI gateway with semantic cache + model router |

### Month 4 — Agents, Evaluation, Capstone

| Week | Topic | Project |
|------|-------|---------|
| [13](./weeks/WEEK-13.md) | Tool calling + agent design patterns | Tool-using code assistant with safety boundaries |
| [14](./weeks/WEEK-14.md) | Agent orchestration with LangGraph | Multi-step coding agent (plan→edit→test→reflect) |
| [15](./weeks/WEEK-15.md) | Agent evaluation + observability | Agent eval suite + Langfuse-style trace viewer |
| [16](./weeks/WEEK-16.md) | **Capstone — Design-to-Code Agent** | End-to-end system + README + demo |

---

## Weekly rhythm (2–3 hrs/day, ~15–20 hrs/week)

Each week file follows this shape:

```
Day 1   Theory primer (best video + foundational article)
Day 2   Deep dive #1 (paper or long-form article)
Day 3   Deep dive #2 + small hands-on lab
Day 4   Hands-on lab (read code, run notebooks)
Day 5   Build the weekly project — part 1
Day 6   Build the weekly project — part 2
Day 7   Polish, write the README, commit, reflect
```

The split is roughly:

```
40%  building
25%  reading / docs
20%  debugging / evaluation
10%  writing notes
 5%  sharing / publishing
```

If you spend 80% of your time watching videos, you are avoiding the real work. The course feeling-of-productivity is a trap.

---

## Recommended stack

**Core:** Python, Git, Linux shell, Docker, FastAPI, Pytest
**Modeling:** PyTorch, Hugging Face Transformers / Datasets / Accelerate, PEFT, TRL, Unsloth
**Inference:** vLLM, llama.cpp, Ollama, Text Generation Inference (TGI)
**Quantization:** bitsandbytes, auto-gptq, autoawq, GGUF
**Retrieval:** sentence-transformers, FAISS, Qdrant or Chroma, BM25 (rank-bm25), Cohere/BGE rerankers
**Agents:** LangGraph, OpenAI Agents SDK, Anthropic tool use
**Eval:** HumanEval, BigCodeBench, Ragas, custom unit-test harnesses, LLM-as-judge
**Observability:** Langfuse (open source) or LangSmith / Phoenix

You don't need all of these in week 1. The weekly files install only what you need that week.

---

## Hardware notes

The whole roadmap is doable on:

- A laptop + free Google Colab (T4) / Kaggle (P100/T4×2) for training weeks
- Or a personal GPU (16GB+ VRAM is comfortable)
- Or rented hourly GPUs (Runpod, Lambda, Vast.ai) — typically $0.30–$1/hr for an RTX 4090 / A10

A small model (1B–3B params) on QLoRA is what most weeks use. You do not need an H100.

---

## How to use this repo

1. Clone it.
2. Read [PREREQUISITES.md](./PREREQUISITES.md). If anything is unfamiliar, fix that first.
3. Each Monday morning, open `weeks/WEEK-XX.md`, skim the whole week, then start Day 1.
4. Keep your week's artifact in its own repo (or a sub-folder of a portfolio repo).
5. At the end of every week, write a short retro (what worked, what didn't, what failed) — this is what differentiates strong engineers.

---

## A note on "doing every video"

You will be tempted to watch every linked video and read every linked article. Don't. The roadmap is a buffet, not a checklist. Pick the format that matches how you learn best:

- **You're a reader?** Articles + papers, skip the videos.
- **You're a watcher?** Videos + code-along, skim the articles.
- **You learn by doing?** Jump to the project on Day 1 and use the readings as a reference.

The deliverable at the end of each week is the only thing that matters.

---

## Final Rule

> Every week must produce a working artifact.
> Every model change must have an eval.
> Every eval must have a report.
> Every report must include failure cases.

That mindset is what separates "someone who can call Claude" from "someone who ships AI products."

---

## License & contributions

This roadmap is open source. If you complete a week and have a better resource than what's listed, open a PR. If something is outdated, open an issue. Public roadmaps stay useful only when their community keeps them honest.
