<div align="center">

# 🧠 AI Engineer Roadmap

### **From "I know Python" to "I ship production LLM systems" — in 4 months, day by day.**

A free, open-source, deeply-researched 16-week roadmap covering **LLM internals, fine-tuning, RAG, inference optimization, agents, evaluation, and observability**. Every week is built on top-tier resources from Karpathy, Anthropic, OpenAI, Hugging Face, Sebastian Raschka, Lilian Weng, 3Blue1Brown, vLLM, and LangChain.

[![Stars](https://img.shields.io/github/stars/abhishekdubey331/AI-Engineer-RoadMap?style=for-the-badge&logo=github&color=fbbf24&logoColor=black)](https://github.com/abhishekdubey331/AI-Engineer-RoadMap/stargazers)
[![Forks](https://img.shields.io/github/forks/abhishekdubey331/AI-Engineer-RoadMap?style=for-the-badge&logo=github&color=60a5fa&logoColor=white)](https://github.com/abhishekdubey331/AI-Engineer-RoadMap/network/members)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e.svg?style=for-the-badge)](./LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-ec4899.svg?style=for-the-badge)](#-contributing)

[![Weeks](https://img.shields.io/badge/Weeks-16-8b5cf6?style=flat-square)](./weeks)
[![Months](https://img.shields.io/badge/Months-4-8b5cf6?style=flat-square)](#-the-16-week-map)
[![Projects](https://img.shields.io/badge/Portfolio_Projects-16-22c55e?style=flat-square)](#-what-youll-have-built)
[![Time](https://img.shields.io/badge/Time-2--3_hrs%2Fday-f97316?style=flat-square)](#-how-to-use-this-roadmap)
[![Updated](https://img.shields.io/github/last-commit/abhishekdubey331/AI-Engineer-RoadMap?style=flat-square&label=updated&color=06b6d4)](https://github.com/abhishekdubey331/AI-Engineer-RoadMap/commits)

[**🚀 Start Here**](./weeks/WEEK-01.md) ・ [**📋 Prerequisites**](./PREREQUISITES.md) ・ [**🗺️ 16-Week Map**](#-the-16-week-map) ・ [**🛠️ Stack**](#-the-stack-youll-learn) ・ [**📎 Beyond the 16 Weeks**](./APPENDIX.md) ・ [**🤝 Contribute**](#-contributing)

</div>

---

## 🎯 What this is

**A working AI engineer's training plan.** Not a course. Not a cheat sheet. Not "100 resources you should look at someday." A sequenced, opinionated, day-by-day path that — if you actually do the work — leaves you with **16 GitHub projects, 1 fine-tuned model published to Hugging Face, and a portfolio capstone** strong enough for serious AI-engineering interviews.

The roadmap is structured so that each week:

- 🎬 Starts with the **best video / article** for the topic (no generic surveys — specific picks)
- 📚 Has a **deep-dive reading list** (papers, blog posts, official docs)
- 💻 Includes a **hands-on lab** so concepts stick
- 🛠️ Ends with a **shippable project** that goes on your GitHub
- ✅ Has a **"Done when…" checklist** so you know when to move on

---

## 🔥 Why this exists

Most AI-engineering content online is one of three things:

1. **Toy tutorials** ("Build a ChatGPT clone in 10 lines!") — fine for an afternoon, useless for a career.
2. **Generic surveys** ("100 papers you should read in 2026") — paralysis-inducing, never finished.
3. **Paid bootcamps** charging $5K+ for what's freely available on YouTube and Hugging Face.

The free internet has *everything* an AI engineer needs to know. What's missing is **sequencing** — what to learn first, what to skip, what to build to make it stick, and how to know you've actually learned it.

That's what this roadmap is.

---

## 👥 Who this is for

✅ Software engineers transitioning to AI / ML engineering
✅ Backend / full-stack engineers who want to ship LLM-powered features
✅ ML engineers moving from classical ML to LLM systems
✅ Self-taught learners who already know Python and Git and want a serious path forward
✅ Bootcamp grads who want to go beyond "call the OpenAI API"

❌ Absolute beginners who don't know Python yet — start with [our prerequisites](./PREREQUISITES.md) first
❌ Researchers chasing SOTA papers — this is an *engineering* roadmap, not a research one
❌ Anyone looking for a 2-week shortcut — this is 4 months of real work

---

## 🏆 What you'll have built

By the end of the 16 weeks, you will have shipped a **complete portfolio**:

| # | Project | Skill it proves | Week |
|---|---|---|---|
| 01 | `token-budget` — BPE tokenizer from scratch + token-budget CLI | Tokenization, embeddings | [W1](./weeks/WEEK-01.md) |
| 02 | `mini-gpt` — train a tiny GPT on your own corpus | Transformer internals | [W2](./weeks/WEEK-02.md) |
| 03 | `code-completer` — local code-completion CLI w/ model benchmark | HF inference, KV cache | [W3](./weeks/WEEK-03.md) |
| 04 | `structured-code-reviewer` — JSON-schema-validated code review bot | Prompting, function calling | [W4](./weeks/WEEK-04.md) |
| 05 | `docs-rag` — chat-with-docs RAG with citations | RAG fundamentals | [W5](./weeks/WEEK-05.md) |
| 06 | `advanced-rag` — hybrid + rerank + Contextual Retrieval + Ragas | Production RAG | [W6](./weeks/WEEK-06.md) |
| 07 | `instruct-dataset-v1` — 1k+ examples for a narrow task | Dataset engineering | [W7](./weeks/WEEK-07.md) |
| 08 | `lora-codetune` — fine-tuned code model + adapter on HF Hub | LoRA + TRL + Unsloth | [W8](./weeks/WEEK-08.md) |
| 09 | `model-zoo-benchmark` — base / LoRA / QLoRA / GPTQ / AWQ / GGUF | Quantization | [W9](./weeks/WEEK-09.md) |
| 10 | `vllm-server` — vLLM + Prometheus + Grafana inference stack | Production serving | [W10](./weeks/WEEK-10.md) |
| 11 | `code-eval-harness` — sandboxed pass@k evaluation harness | Eval engineering | [W11](./weeks/WEEK-11.md) |
| 12 | `ai-gateway` — semantic cache + model router + rate limits | LLM platform engineering | [W12](./weeks/WEEK-12.md) |
| 13 | `code-helper-agent` — tool-using coding agent (hand-rolled + Agents SDK) | Tool calling, agent design | [W13](./weeks/WEEK-13.md) |
| 14 | `coding-agent-graph` — multi-step agent in LangGraph w/ HIL | Agent orchestration | [W14](./weeks/WEEK-14.md) |
| 15 | `agent-eval-and-obs` — eval suite + Langfuse traces + benchmarks | Agent eval + observability | [W15](./weeks/WEEK-15.md) |
| 16 | **🏁 `swe-bench-coding-agent`** — autonomous coding agent benchmarked on SWE-bench-Verified-Lite | **Everything above + public headline number** | [W16](./weeks/WEEK-16.md) |

Pin these on your GitHub. They *are* your portfolio.

---

## 🗺️ The 16-week map

<div align="center">

### Month 1 · Foundations of LLMs

</div>

| Week | Topic | Project | Key resources |
|:---:|---|---|---|
| **[01](./weeks/WEEK-01.md)** | Tokenization & Embeddings | BPE from scratch + `token-budget` CLI | Karpathy *minbpe*, HF NLP Course |
| **[02](./weeks/WEEK-02.md)** | The Transformer (attention, decoder-only) | Train a `mini-gpt` on a custom corpus | Karpathy *Let's build GPT*, Jay Alammar, 3Blue1Brown |
| **[03](./weeks/WEEK-03.md)** | Modern LLM internals + Hugging Face | Local `code-completer` w/ benchmark mode | HF docs, KV-cache deep dive |
| **[04](./weeks/WEEK-04.md)** | Prompting, structured outputs, function calling | `structured-code-reviewer` (Pydantic + retries) | Anthropic + OpenAI prompting docs, Instructor, Outlines |

<div align="center">

### Month 2 · Building with LLMs

</div>

| Week | Topic | Project | Key resources |
|:---:|---|---|---|
| **[05](./weeks/WEEK-05.md)** | RAG fundamentals (chunking, embeddings, vector DBs) | `docs-rag` (from scratch + LlamaIndex) | Pinecone Learn, Greg Kamradt, MTEB |
| **[06](./weeks/WEEK-06.md)** | Advanced RAG: hybrid search, reranking, contextual retrieval | `advanced-rag` w/ Ragas eval | Anthropic Contextual Retrieval, Sentence-Transformers |
| **[07](./weeks/WEEK-07.md)** | Fine-tuning theory + dataset preparation | `instruct-dataset-v1` (1k+ examples + dataset card) | Sebastian Raschka, Alpaca, Self-Instruct |
| **[08](./weeks/WEEK-08.md)** | LoRA fine-tuning hands-on (TRL + PEFT + Unsloth) | `lora-codetune` + adapter on HF Hub | TRL, PEFT, Unsloth, Raschka practical tips |

<div align="center">

### Month 3 · Production LLM Systems

</div>

| Week | Topic | Project | Key resources |
|:---:|---|---|---|
| **[09](./weeks/WEEK-09.md)** | QLoRA + quantization (GPTQ, AWQ, GGUF) | `model-zoo-benchmark` — full quant sweep | Maarten Grootendorst's visual guide, QLoRA paper |
| **[10](./weeks/WEEK-10.md)** | Inference servers (vLLM, PagedAttention, KV cache) | `vllm-server` + Prometheus + Grafana | vLLM docs, Anyscale on continuous batching |
| **[11](./weeks/WEEK-11.md)** | Code evaluation (HumanEval, SWE-bench, pass@k) | `code-eval-harness` — sandboxed runner | HumanEval, BigCodeBench, Inspect AI |
| **[12](./weeks/WEEK-12.md)** | Caching, routing, production API hardening | `ai-gateway` w/ semantic cache & router | GPTCache, LiteLLM, Anthropic prompt caching |

<div align="center">

### Month 4 · Agents, Evaluation & Capstone

</div>

| Week | Topic | Project | Key resources |
|:---:|---|---|---|
| **[13](./weeks/WEEK-13.md)** | Tool calling + agent design patterns | `code-helper-agent` (hand-rolled + Agents SDK) | Anthropic *Building Effective Agents*, OpenAI Agents SDK |
| **[14](./weeks/WEEK-14.md)** | Agent orchestration with LangGraph | `coding-agent-graph` w/ HIL + checkpoints | LangGraph docs |
| **[15](./weeks/WEEK-15.md)** | Agent evaluation + observability (Langfuse) | `agent-eval-and-obs` + SWE-bench-Lite run | Anthropic on agent evals, Langfuse, Inspect AI |
| **[16](./weeks/WEEK-16.md)** | 🏁 **Capstone: Autonomous Coding Agent on SWE-bench-Verified-Lite** | End-to-end system w/ public benchmark headline number + demo | All of the above |

---

## ⚙️ How to use this roadmap

> **The way you "do" this roadmap matters more than the roadmap itself.**

1. **Skim the [Prerequisites](./PREREQUISITES.md).** If anything is unfamiliar, fix that first — it's a self-audit, not gatekeeping.
2. **Open [Week 1](./weeks/WEEK-01.md) on a Monday morning.** Read the whole week first, then start Day 1.
3. **Build the project.** A week without a finished artifact is a week you didn't really do.
4. **Push every project to GitHub.** Pin the good ones.
5. **Write a 1-paragraph retro** at the end of each week — what worked, what failed, what surprised you.
6. **Move on.** Resist the temptation to "perfect" a week. Done > perfect.

### ⏱️ Time commitment

| Pace | Hours / week | What you'll get |
|---|---|---|
| 🟢 **Sustainable** | 15–20 hrs (≈ 2–3 hrs / day) | Default. Full coverage. Strong portfolio. |
| 🟡 **Aggressive** | 25–30 hrs | Finish in ~3 months. Some polish lost. |
| 🔴 **Sabbatical / bootcamp** | 40+ hrs | Finish in ~2 months. Bring snacks. |

### 🎯 Weekly rhythm

```
40%  building (the project)
25%  reading / docs
20%  debugging / evaluation
10%  writing notes
 5%  sharing / publishing
```

If you spend 80% of your time watching videos, you are *avoiding* the work, not doing it. The course-feeling-of-productivity is a trap.

---

## 🛠️ The stack you'll learn

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![vLLM](https://img.shields.io/badge/vLLM-FF6F00?style=for-the-badge)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-DC382D?style=for-the-badge)
![Langfuse](https://img.shields.io/badge/Langfuse-0A0A0A?style=for-the-badge)

</div>

| Category | Tools |
|---|---|
| **Core** | Python, Git, Linux shell, Docker, FastAPI, Pytest |
| **Modeling** | PyTorch, Hugging Face Transformers / Datasets / Accelerate, PEFT, TRL, Unsloth |
| **Inference** | vLLM, llama.cpp, Ollama, Text Generation Inference (TGI) |
| **Quantization** | bitsandbytes, auto-gptq, autoawq, GGUF |
| **Retrieval** | sentence-transformers, Qdrant / Chroma, FAISS, BM25, BGE rerankers |
| **Agents** | LangGraph, OpenAI Agents SDK, Anthropic tool use |
| **Evaluation** | HumanEval, BigCodeBench, Ragas, Inspect AI, LLM-as-judge |
| **Observability** | Langfuse (default), LangSmith, Arize Phoenix |

---

## 💻 Hardware notes

The whole roadmap is doable on:

- **🆓 Free tier:** a laptop + Google Colab T4 / Kaggle (T4×2) for training weeks
- **💻 Personal GPU:** 16GB+ VRAM is comfortable
- **☁️ Rented:** Runpod / Lambda / Vast.ai — typically **$0.30–$1/hr** for an RTX 4090 / A10

Most weeks use a small model (1B–3B params) on QLoRA. **You don't need an H100.**

---

## 🌟 Why "yet another roadmap"?

| | Other roadmaps | **This roadmap** |
|---|:---:|:---:|
| Generic ML survey | ✅ | ❌ |
| Built around 2025–2026 stack (vLLM, LangGraph, Contextual Retrieval) | ❌ | ✅ |
| Day-by-day plan | ❌ | ✅ |
| Every week ends with a shippable project | ❌ | ✅ |
| Curated by topic from named experts (Karpathy, Raschka, Weng…) | ❌ | ✅ |
| Includes evaluation & observability (not just "build & ship") | ❌ | ✅ |
| Free, open-source, MIT-licensed | ⚠️ | ✅ |

---

## 🤝 Contributing

This roadmap stays useful only if its community keeps it honest. **PRs are not just welcome — they're the point.**

Ways to contribute:

- 🔗 **Found a broken link?** → Open an issue or PR
- 📚 **Better resource than what's listed?** → Open a PR replacing it (must be at least as in-depth)
- 🐛 **Spotted a factual error?** → Open an issue
- ⚡ **Finished a week and have feedback?** → Open a discussion
- 🏗️ **Built a great Week-X project?** → Open a PR linking to it in a `community-projects/` section
- 🌍 **Want to translate a week?** → Open an issue first to coordinate

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) (coming soon) for the long version.

---

## ⭐ Star history

If this roadmap is useful, **starring the repo** is the single biggest signal you can send. It helps other learners find it and keeps the project visible enough for me (and contributors) to keep updating.

[![Star History Chart](https://api.star-history.com/svg?repos=abhishekdubey331/AI-Engineer-RoadMap&type=Date)](https://star-history.com/#abhishekdubey331/AI-Engineer-RoadMap&Date)

---

## 🙏 Acknowledgements

This roadmap stands on the shoulders of giants. The single best resources, by topic:

- **Tokenization & transformers** — [Andrej Karpathy](https://karpathy.ai/), [3Blue1Brown](https://www.3blue1brown.com/topics/neural-networks), [Jay Alammar](https://jalammar.github.io/)
- **Fine-tuning & PEFT** — [Sebastian Raschka](https://magazine.sebastianraschka.com/), [Hugging Face TRL/PEFT team](https://huggingface.co/docs/trl), [Unsloth](https://unsloth.ai/)
- **RAG** — [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval), [Pinecone Learn](https://www.pinecone.io/learn/), [Greg Kamradt](https://github.com/FullStackRetrieval-com/RetrievalTutorials)
- **Quantization** — [Maarten Grootendorst](https://newsletter.maartengrootendorst.com/), [Tim Dettmers (QLoRA)](https://arxiv.org/abs/2305.14314)
- **Inference** — [vLLM team](https://github.com/vllm-project/vllm), [Anyscale](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- **Agents** — [Anthropic *Building Effective Agents*](https://www.anthropic.com/research/building-effective-agents), [Lilian Weng](https://lilianweng.github.io/), [LangChain team](https://langchain-ai.github.io/langgraph/)
- **Evaluation** — [Anthropic *Demystifying evals*](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents), [Eugene Yan](https://eugeneyan.com/), [Inspect AI](https://inspect.ai-safety-institute.org.uk/)

If your work is referenced and you'd like a link changed or credit corrected, open an issue and I'll fix it the same day.

---

## 📬 Share this roadmap

If this helps you, please share it — it's the only "marketing" this project has:

[![Share on X](https://img.shields.io/badge/Share_on-X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/intent/tweet?text=A%20free%204-month%20day-by-day%20AI%20Engineer%20Roadmap%3A%20LLM%20internals%2C%20fine-tuning%2C%20RAG%2C%20vLLM%2C%20agents%2C%20evals%20%E2%86%92&url=https%3A%2F%2Fgithub.com%2Fabhishekdubey331%2FAI-Engineer-RoadMap)
[![Share on LinkedIn](https://img.shields.io/badge/Share_on-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Fgithub.com%2Fabhishekdubey331%2FAI-Engineer-RoadMap)
[![Share on HN](https://img.shields.io/badge/Share_on-Hacker_News-FF6600?style=for-the-badge&logo=ycombinator&logoColor=white)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fabhishekdubey331%2FAI-Engineer-RoadMap&t=AI%20Engineer%20Roadmap%3A%20a%20free%204-month%20day-by-day%20path)
[![Share on Reddit](https://img.shields.io/badge/Share_on-Reddit-FF4500?style=for-the-badge&logo=reddit&logoColor=white)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fabhishekdubey331%2FAI-Engineer-RoadMap&title=AI%20Engineer%20Roadmap%20%E2%80%94%20free%204-month%20day-by-day%20path)

---

## 📜 License

Released under the [**MIT License**](./LICENSE) — free for personal, educational, and commercial use. Attribution appreciated but not required.

---

## 📎 Beyond the 16 weeks

The roadmap is deliberately scoped to *LLM systems engineering*. Two areas are lightly covered on purpose, plus one reading habit that's mentioned everywhere but rarely taught explicitly. All three are covered in [**APPENDIX.md**](./APPENDIX.md):

- 🔧 **MLOps extensions** — dataset versioning (DVC), experiment tracking (MLflow / W&B), model registry, pipeline orchestration, drift monitoring, deployment patterns. Curated path through [Made With ML](https://madewithml.com/courses/mlops/) and [DataTalksClub MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp), plus Chip Huyen's [*Designing ML Systems*](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) and [*AI Engineering*](https://www.oreilly.com/library/view/ai-engineering/9781098166298/).
- 📄 **How to read AI papers** — Keshav's [three-pass method](https://www.cs.tufts.edu/~nr/cs257/archive/keshav/paper-reading.pdf), an LLM-augmented variant from Karpathy, plus the curated paper list (mapped to weeks) that backs the roadmap.
- 🧭 **The north-star sentence** to hold yourself to after the 16 weeks.

---

## 🎯 The final rule

> Every week must produce a working artifact.
> Every model change must have an eval.
> Every eval must have a report.
> Every report must include failure cases.

That mindset is what separates "someone who can call Claude" from "someone who ships AI products."

<div align="center">

**Built for engineers who want to ship, not just learn.**

If this saved you a year of self-study, [⭐ star the repo](https://github.com/abhishekdubey331/AI-Engineer-RoadMap/stargazers).

[**🚀 Start Week 1**](./weeks/WEEK-01.md)

</div>
