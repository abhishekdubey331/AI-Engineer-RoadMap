# Week 10 — Inference Servers (vLLM, PagedAttention, KV Cache)

> **Month 3 · Production LLM Systems**
> *"The difference between `model.generate()` in a script and a real serving system is roughly 30× in throughput. That's what this week is."*

---

## Why this week

You can squeeze 10–30 tokens/sec out of `transformers.generate()`. The same hardware running **vLLM** gives you 200–800 tokens/sec per request and serves dozens of concurrent requests through **PagedAttention** + **continuous batching**.

This week you go from "I run a model" to "I serve a model behind an OpenAI-compatible HTTP API with streaming, paged KV cache, continuous batching, and proper observability." You'll also poke at the alternatives — TGI, llama.cpp's server, Ollama — and know when to use which.

---

## Learning objectives

By Sunday night you should be able to:

1. Explain PagedAttention in your own words: virtual blocks, the block table, sharing across requests
2. Explain **continuous batching** vs static batching and why it tripled the field's throughput
3. Stand up a vLLM server with an OpenAI-compatible API and stream from it
4. Read and interpret vLLM metrics: TTFT, ITL (inter-token latency), throughput, KV-cache utilization
5. Compare vLLM, TGI, llama.cpp server, and Ollama on the same model + the same workload
6. Understand **speculative decoding** at a conceptual level and turn it on in vLLM
7. Decide which serving stack fits a given deployment

---

## Day-by-day plan (≈ 2–3 hrs/day)

### Day 1 — KV cache deep dive (the prerequisite)

You touched the KV cache in Week 3. This week we go deeper, because PagedAttention only makes sense once you can see why naive KV-cache management is wasteful.

**Read (90 min):**
- [HF blog — *KV cache quantization*](https://huggingface.co/blog/kv-cache-quantization)
- [Kwon et al. — *vLLM paper (PagedAttention)*](https://arxiv.org/abs/2309.06180) — the paper that defined modern KV-cache layout
- [Aleksa Gordić — *Inside vLLM: Anatomy of a High-Throughput LLM Inference System*](https://www.aleksagordic.com/blog/vllm) — the 2025/26 successor to the old vLLM design doc; covers scheduler, prefix caching, chunked prefill, V1 architecture

**Math check:** for Llama-3-8B (32 layers, 32 heads, `d_head=128`, GQA with 8 KV heads), how big is the KV cache per token in fp16? Per 8k-token request?

Approximately: `2 (K & V) × 32 layers × 8 KV heads × 128 d_head × 2 bytes ≈ 130 KB / token`. An 8k-token request ⇒ ~1 GB just of KV cache for one request. You see why memory rules everything.

---

### Day 2 — PagedAttention & vLLM internals

**Read (90 min):**
- [Aleksa Gordić — *Inside vLLM* (deep technical walkthrough)](https://www.aleksagordic.com/blog/vllm) — required; replaces the historical PagedAttention design doc
- [vLLM docs — *PagedAttention*](https://docs.vllm.ai/en/latest/design/paged_attention/) — historical reference (the page itself notes it no longer matches current code)
- [Hamza Elshafie — *Paged Attention from First Principles*](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles/) — clear "first principles" walk-through
- [vLLM — *Automatic Prefix Caching* design doc](https://docs.vllm.ai/en/stable/design/prefix_caching/) — the table-stakes 2026 optimization; 50–90% hit rates on RAG/agent workloads
- [vLLM — *Chunked Prefill*](https://docs.vllm.ai/en/latest/features/chunked_prefill/) — default-on in vLLM V1; the lever balancing TTFT vs throughput

**Watch (30 min, optional):**
- [Woosuk Kwon (vLLM creator) — *PagedAttention & vLLM* lecture slides (CMU LLM Systems 2025)](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-22-vLLM_woosuk_kwon-1f34697dbb1a1fb5b798daf6eff14b67.pdf)

**One-line takeaway:** the KV cache is broken into **fixed-size blocks** (e.g., 16 tokens). Each sequence has a *logical* block table → *physical* GPU blocks. Memory waste drops from 60–80% to ~4%. Shared prefixes (system prompts!) share physical blocks.

---

### Day 3 — Continuous batching

Static batching: you wait until N requests arrive, batch them, run one pass, return. Late arrivals get blocked. Short requests pay for long ones.

**Continuous batching:** new requests join the running batch mid-flight, and finished sequences drop out. Like an escalator instead of an elevator. This is what makes serving throughput possible.

**Read (60 min):**
- [Anyscale — *How Continuous Batching enables 23× throughput*](https://www.anyscale.com/blog/continuous-batching-llm-inference) — *the* canonical article
- [BentoML — *LLM Inference Handbook*](https://bentoml.com/llm/inference-optimization/) — modern, vendor-neutral chapters on batching, paging, caching, scheduling

---

### Day 4 — Bring up a real vLLM server

**Read (45 min):**
- [vLLM docs — *Quickstart*](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [vLLM docs — *OpenAI Compatible Server*](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [vLLM docs — *Engine Args*](https://docs.vllm.ai/en/latest/configuration/engine_args.html)

**Hands-on (~2 hr):**
```bash
pip install vllm

vllm serve Qwen/Qwen3-Coder-1.5B-Instruct \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16
```

Now call it like OpenAI:
```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(c.chat.completions.create(
    model="Qwen/Qwen3-Coder-1.5B-Instruct",
    messages=[{"role": "user", "content": "def fib(n):"}],
    stream=True,
))
```

Try also:
- Quantized: `--quantization awq` (or `gptq`) with your Week 9 quantized model
- Larger context: `--max-model-len 16384`
- Multiple replicas: `--tensor-parallel-size 2` if you have 2 GPUs

---

### Day 5 — Load testing & metrics

Now actually push it.

**Read (30 min):**
- [vLLM — *Production Metrics*](https://docs.vllm.ai/en/latest/serving/metrics.html)
- [vLLM — *Performance benchmarking*](https://docs.vllm.ai/en/latest/contributing/benchmarks.html)
- [Nebius — *Serving LLMs with vLLM: A practical inference guide*](https://nebius.com/blog/posts/serving-llms-with-vllm-practical-guide)

**Hands-on (90 min):**
- Use [`vllm bench serve`](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) or `wrk` / `oha` to generate concurrent requests
- Plot:
  - Throughput (req/s) vs concurrency
  - TTFT (time-to-first-token) p50 / p95
  - ITL (inter-token-latency) p50 / p95
  - KV-cache utilization (vLLM exposes this on `/metrics`)
- Find the **knee** — concurrency at which TTFT starts to climb sharply

---

### Day 6 — Speculative decoding (EAGLE-3 / Medusa) + the alternatives

**Read (60 min):**
- [vLLM docs — *Speculative Decoding*](https://docs.vllm.ai/en/latest/features/spec_decode.html) — current API
- [vLLM blog — *How Speculative Decoding Boosts vLLM Performance by up to 2.8×*](https://blog.vllm.ai/2024/10/17/spec-decode.html)
- [BentoML — *Speculative decoding* (LLM Inference Handbook)](https://bentoml.com/llm/inference-optimization/speculative-decoding) — the diminishing-returns-vs-batch story
- [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) — current SOTA speculative-decoding method (NeurIPS '25)
- [NVIDIA — *An Introduction to Speculative Decoding*](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)

**Hands-on (45 min):**
- Turn on speculative decoding in vLLM with **EAGLE-3** (best) or **Medusa** (also supported); fall back to ngram / prompt-lookup if you don't have a matching draft model
- Re-measure tokens/sec at batch size 1, 8, and 32. The lift narrows as batch grows — see the BentoML article.

**Read (30 min) — the 2026 alternatives tour:**
- [TGI (Text Generation Inference) — *Overview*](https://huggingface.co/docs/text-generation-inference/index)
- [Ollama — *Getting Started*](https://github.com/ollama/ollama#readme)
- [llama.cpp — *Server*](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [SGLang docs](https://docs.sglang.ai/) — genuinely competitive: ~29% throughput advantage on H100, up to 6.4× on prefix-heavy workloads via RadixAttention. Required name-check in 2026.
- [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — 1.4–2.1× faster than vLLM on H100/H200 for single-model production with engineering capacity. Know when it wins.
- [llm-d](https://github.com/llm-d/llm-d) — Kubernetes-native vLLM-based serving (good pointer for the K8s stretch goal)
- [LMCache](https://github.com/LMCache/LMCache) — production KV-cache layer; drops TTFT from 11s to 1.5s on 128k prompts at GKE/CoreWeave

---

### Day 7 — Project + retro

Today: ship the weekly project (next section).

---

## Weekly Project — `vllm-server`

A production-style inference server fronted by **vLLM**, packaged as Docker, with metrics, streaming, and a load-test report.

### Spec

```
vllm-server/
  docker-compose.yml          # vllm + Prometheus + Grafana
  prometheus.yml
  grafana/
    dashboard.json            # your TTFT / throughput / KV-utilization dashboard
  load_test/
    generate_traffic.py       # configurable concurrency
    workloads/
      short_prompts.jsonl
      long_prompts.jsonl
      mixed.jsonl
    run_bench.sh              # runs vllm bench serve
  reports/
    latency_vs_concurrency.png
    ttft_p50_p95.png
    backends_comparison.md    # vLLM vs TGI vs Ollama on the same workload
    final_report.md
  README.md
```

### Requirements

- vLLM serving your fine-tuned Week-8 model (or any small open-weight)
- OpenAI-compatible HTTP endpoint with streaming
- Docker Compose stack including Prometheus scraping `/metrics` and Grafana dashboard
- Load-test script that varies concurrency 1, 4, 16, 64 and reports:
  - Throughput (tok/s, req/s)
  - TTFT p50/p95
  - ITL p50/p95
- A `backends_comparison.md` running the same model on at least 2 of: vLLM / **SGLang** / TGI / llama.cpp server / Ollama. Tabulate throughput + p50 TTFT. **SGLang or TensorRT-LLM is required as one of the comparison points** — knowing only vLLM in 2026 dates you.
- **Prefix-cache hit-rate** measurements from `/metrics` (required, not stretch) — this is the most impactful single optimization in 2026
- A `final_report.md` with the knee-of-the-curve analysis and a 1-paragraph "what I'd change for prod"

### Stretch

- Speculative decoding with **EAGLE-3 or Medusa** (not just ngram) — report the lift at batch 1, 8, 32
- KV-cache fp8 (`--kv-cache-dtype fp8`) — measure the throughput delta
- A Kubernetes manifest (Deployment + Service + HPA) or an [llm-d](https://github.com/llm-d/llm-d) deploy — useful for resume signal even if you don't run it
- Disaggregated prefill/decode (P/D split) — see vLLM's [disagg docs](https://docs.vllm.ai/en/latest/features/disagg_prefill/)

---

## Curated resources

**vLLM (your primary stack)**
- [vLLM docs](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM — *PagedAttention design doc*](https://docs.vllm.ai/en/latest/design/paged_attention/)
- [vLLM — *Production Metrics*](https://docs.vllm.ai/en/latest/serving/metrics.html)
- [vLLM blog — *Speculative Decoding*](https://blog.vllm.ai/2024/10/17/spec-decode.html)

**Concepts**
- [Aleksa Gordić — *Inside vLLM*](https://www.aleksagordic.com/blog/vllm) — the SOTA-author deep walkthrough
- [Anyscale — *Continuous Batching*](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [Hamza Elshafie — *Paged Attention from First Principles*](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles/)
- [BentoML — *LLM Inference Handbook*](https://bentoml.com/llm/inference-optimization/) — vendor-neutral chapters on every topic this week
- [llm-d — *KV-Cache Wins You Can See*](https://llm-d.ai/blog/kvcache-wins-you-can-see) — real production prefix-cache hit-rate data

**Speculative decoding**
- [PyTorch — *A Hitchhiker's Guide to Speculative Decoding*](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)
- [BentoML — *Speculative decoding*](https://bentoml.com/llm/inference-optimization/speculative-decoding)
- [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) — current SOTA, NeurIPS '25
- [NVIDIA — *Intro to Speculative Decoding*](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)
- [Red Hat — *Performance improvements with speculative decoding in vLLM (Apr 2026)*](https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss)

**Alternatives & production stacks**
- [Hugging Face TGI](https://huggingface.co/docs/text-generation-inference/index)
- [Ollama](https://github.com/ollama/ollama)
- [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [SGLang](https://docs.sglang.ai/) — a strong vLLM competitor (RadixAttention)
- [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — production-grade single-model serving
- [llm-d](https://github.com/llm-d/llm-d) — Kubernetes-native vLLM
- [LMCache](https://github.com/LMCache/LMCache) — production KV-cache layer

**Papers (skim)**
- [Kwon et al. — *Efficient Memory Management for LLM Serving with PagedAttention*](https://arxiv.org/abs/2309.06180) — the vLLM paper
- [Leviathan et al. — *Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192)

---

## "Done when…" checklist

- [ ] I can explain PagedAttention and continuous batching without notes
- [ ] My vLLM server runs in Docker and streams tokens via an OpenAI-compatible endpoint
- [ ] I have throughput vs concurrency curves on disk
- [ ] I know my model's knee — concurrency where TTFT p95 doubles
- [ ] I've compared at least 2 backends and have an opinion
- [ ] Speculative decoding is on and I have a number for its lift on my workload
- [ ] I've written a retro

---

## Common pitfalls this week

1. **GPU OOM at startup.** vLLM preallocates KV cache by default. Lower `--gpu-memory-utilization` to 0.85 if your card is shared.
2. **TTFT vs throughput confusion.** Maximize throughput → batch hard. Minimize TTFT → don't. You can't have both for free. Decide which matters for your product.
3. **Benchmarking with `time curl`.** Use a real load tester (`vllm bench serve`, `oha`, `wrk`) that opens many concurrent connections.
4. **Forgetting `--max-model-len`.** Default may be the full model context, which preallocates a giant KV cache and limits batch size. Set it to the longest *you'll actually use*.
5. **Mixing chat templates.** vLLM applies the model's chat template when you use `/v1/chat/completions`. Don't pre-apply it.

---

← Previous: [Week 9 — QLoRA & Quantization](./WEEK-09.md) · → Next: [Week 11 — Code Evaluation (HumanEval, SWE-bench, pass@k)](./WEEK-11.md)
