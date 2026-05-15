# Week 3 — Modern LLM Internals + Hugging Face Hands-On

> **Month 1 · Foundations of LLMs**
> *"Last week you built a tiny model. This week you run a real one — and start to learn what makes it fast or slow, cheap or expensive."*

---

## Why this week

You've trained your own tiny transformer. That was the educational scaffold. Real-world AI engineering is mostly about working with **already-trained** open-weight models — Llama, Qwen, DeepSeek, Mistral, Phi — through the **Hugging Face** ecosystem.

This week you'll:

- Load and run a real open-weight code model locally
- Learn the generation parameters that matter (temperature, top_p, repetition_penalty, max_new_tokens)
- Understand the **KV cache** — the single most important inference optimization
- Build a small code-completion CLI you'll iterate on for the rest of the roadmap

This is the bridge between "I understand transformers" and "I can build with them."

---

## Learning objectives

By Sunday night you should be able to:

1. Use `AutoTokenizer` and `AutoModelForCausalLM` to load any HF model
2. Generate text with `.generate()` and explain every kwarg
3. Explain what the **KV cache** is, what it stores, why it's huge, and why it's the bottleneck
4. Explain prefill vs decode and why decode is memory-bandwidth-bound
5. Explain temperature, top-p, top-k, repetition penalty (mathematically, not vibes)
6. Stream tokens from a local model
7. Quantize a model to 4-bit on the fly with `bitsandbytes`
8. Compare 2–3 small open-weight code models head-to-head on a simple task

---

## Day-by-day plan (≈ 2–3 hrs/day)

### Day 1 — The Hugging Face ecosystem

**Watch (60 min):**
- [HuggingFace — *Welcome to the 🤗 LLM Course*](https://huggingface.co/learn/llm-course/chapter1/1) — work through Chapter 1 (it's short, mostly video clips + small code snippets)

**Read (30 min):**
- [Transformers — *Quick tour*](https://huggingface.co/docs/transformers/quicktour)
- [Transformers — *Pipelines*](https://huggingface.co/docs/transformers/main_classes/pipelines) (skim)

**Hands-on (30 min):**
```python
from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-Coder-0.5B-Instruct")
print(pipe("def fibonacci(n: int) -> int:", max_new_tokens=80)[0]["generated_text"])
```
Try 2–3 different small code models from the HF hub. Notice latency, memory, and output quality.

**Models to try (all small, all permissive):**
- `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- `bigcode/starcoder2-3b`
- `microsoft/Phi-3.5-mini-instruct`
- `HuggingFaceTB/SmolLM2-360M-Instruct` (tiniest, for laptops without a GPU)

---

### Day 2 — `AutoModelForCausalLM` and `.generate()`

Skip the `pipeline` abstraction today. You need to know what's underneath.

**Read (60 min):**
- [Transformers — *Generation strategies*](https://huggingface.co/docs/transformers/main/en/generation_strategies)
- [Transformers — *Text generation*](https://huggingface.co/docs/transformers/main_classes/text_generation) — the `generate()` API
- [HF blog — *How to generate text*](https://huggingface.co/blog/how-to-generate) — old but the canonical explainer for sampling strategies

**Hands-on (60 min):**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

inputs = tok("def quicksort(arr):", return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.05,
)
print(tok.decode(out[0], skip_special_tokens=True))
```

Now systematically vary one knob at a time:
- `temperature=0.0` vs `1.5` — what changes?
- `top_p=0.1` vs `0.95` — same prompt, what changes?
- `repetition_penalty=1.0` vs `1.3` — what changes?

Take notes. This is intuition you cannot get from reading.

---

### Day 3 — Streaming + chat templates

Real apps stream tokens as they generate. You also need to format inputs correctly for instruction-tuned models (which is most of them).

**Read (45 min):**
- [Transformers — *Chat templates*](https://huggingface.co/docs/transformers/main/en/chat_templating) — apply_chat_template is non-negotiable for instruct models
- [Transformers — *Streaming*](https://huggingface.co/docs/transformers/main/en/generation_strategies#streaming) — TextStreamer / TextIteratorStreamer

**Hands-on (60 min):**
- Build a CLI that takes a prompt and streams tokens to stdout in real-time
- Use `apply_chat_template` properly with `messages=[{"role": "user", "content": "..."}]`
- Notice the difference between the **base** model and the **instruct** model output on the same prompt

---

### Day 4 — KV cache: the most important concept this week

Inference is not just "run the model." It's prefill + decode + a giant cache. If you don't understand the KV cache, every later inference-optimization week (vLLM, paging, prefix caching) will be confusing.

**Read (75 min) — pick 2 of these 3:**
- [HuggingFace — *Best Practices for Generation with Cache*](https://huggingface.co/docs/transformers/main/en/kv_cache) — official deep dive
- [João Gante — *Generate: KV Cache strategies*](https://huggingface.co/blog/kv-cache-quantization) — practical, including quantizing the KV cache
- [Introl — *KV Cache Optimization*](https://introl.com/blog/kv-cache-optimization-memory-efficiency-production-llms-guide) — the production-engineer view

**Watch (optional, ~30 min):**
- [Efficient NLP — *Speeding up the GPT KV cache (visual)*](https://www.youtube.com/watch?v=80bIUggRJf4)

**Reflect:**
- For a 7B model with 32 layers, 32 heads, `d_head=128`, what's the per-token KV cache size in bytes (fp16)?
- Why is the KV cache so much memory at long context?
- What's the difference between **prefill** and **decode** time, and why is decode memory-bandwidth-bound while prefill is compute-bound?

---

### Day 5 — Quantize a model and notice nothing breaks

You don't need to fully understand quantization yet (Week 9 is the deep dive). But running a model in 4-bit so it fits on your hardware is a Day-1 production skill.

**Read (30 min):**
- [HF — *Quantization*](https://huggingface.co/docs/transformers/main/en/quantization/overview) — the overview
- [HF blog — *Making LLMs lighter with AutoGPTQ and transformers*](https://huggingface.co/blog/gptq-integration) (skim)

**Hands-on (90 min):**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    quantization_config=bnb,
    device_map="auto",
)
```

Measure for the same prompt:
- VRAM usage (`nvidia-smi`)
- First-token latency
- Tokens / second
- Subjective output quality on 5 prompts

Write the numbers down. You'll cite this in Week 9.

---

### Day 6 — Build the weekly project (Part 1)

See **Weekly Project** below. Today: get the CLI loading models, accepting prompts, and streaming output.

---

### Day 7 — Build the weekly project (Part 2) + benchmark + write-up

Today: add the benchmark mode, write the README, commit.

---

## Weekly Project — `code-completer`

A local code-completion CLI that wraps a small open-weight model, with a benchmark mode that lets you compare models.

### Spec

```
$ code-completer --model qwen-0.5b "def fibonacci(n: int) -> int:"
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

[12.4 tok/s, 4.2s, peak 1.1 GB VRAM]

$ code-completer bench --prompts ./benchmarks/prompts.jsonl --models qwen-0.5b,qwen-1.5b,phi-3.5
model       prompts  avg-tok/s  avg-first-token-ms  vram-peak  pass@1*
qwen-0.5b   20       38.2       180                 1.1 GB     0.40
qwen-1.5b   20       21.5       320                 3.2 GB     0.55
phi-3.5     20       18.9       410                 4.4 GB     0.65

* pass@1 here is exact-line match against expected completion (cheap proxy; we'll replace this in Week 11)
```

### Requirements

- Use `AutoTokenizer` + `AutoModelForCausalLM`, **not** `pipeline`
- Streaming output via `TextIteratorStreamer`
- `apply_chat_template` when the model is instruct-tuned
- Support at least 3 models from this list:
  - `Qwen/Qwen2.5-Coder-0.5B-Instruct`
  - `Qwen/Qwen2.5-Coder-1.5B-Instruct`
  - `bigcode/starcoder2-3b`
  - `microsoft/Phi-3.5-mini-instruct`
  - `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- `--load-4bit` flag using `BitsAndBytesConfig`
- Benchmark mode reads JSONL prompts and writes a `bench_report.md` with:
  - Average tokens/sec
  - Average first-token latency
  - Peak VRAM
  - Exact-line match accuracy on a small homemade test set (~20 prompts)
- 5+ `pytest` tests
- A clean `README.md` with: install steps, 3 example invocations, and a 1-paragraph note on which model gave the best quality/speed tradeoff *for you*

### Stretch

- Add `--device cpu` and benchmark CPU vs GPU
- Add `--stop` for stop sequences (e.g., `\n\ndef ` to stop at the next function)
- Add `--fill-in-middle` support if you pick a model that has FIM tokens (Qwen Coder and StarCoder2 do)

---

## Curated resources

**Official docs (your reference)**
- [Hugging Face LLM Course — *Chapter 1*](https://huggingface.co/learn/llm-course/chapter1/1)
- [Transformers — *Quick tour*](https://huggingface.co/docs/transformers/quicktour)
- [Transformers — *Generation strategies*](https://huggingface.co/docs/transformers/main/en/generation_strategies)
- [Transformers — *Chat templates*](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Transformers — *KV Cache*](https://huggingface.co/docs/transformers/main/en/kv_cache)
- [Transformers — *Quantization overview*](https://huggingface.co/docs/transformers/main/en/quantization/overview)

**Long-form articles**
- [HF blog — *How to generate text*](https://huggingface.co/blog/how-to-generate) — sampling strategies explained
- [HF blog — *KV cache quantization*](https://huggingface.co/blog/kv-cache-quantization)
- [Introl — *KV Cache Optimization for Production*](https://introl.com/blog/kv-cache-optimization-memory-efficiency-production-llms-guide)
- [Sebastian Raschka — *Understanding and Coding the Self-Attention Mechanism*](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) (great refresher)

**Models for the week**
- [Qwen2.5-Coder series](https://huggingface.co/collections/Qwen/qwen25-coder-66eaa22e6f99801bf65b0c2f)
- [Phi-3.5 mini instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- [SmolLM2 collection](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9) — laptop-friendly
- [StarCoder2](https://huggingface.co/bigcode/starcoder2-3b)

---

## "Done when…" checklist

- [ ] I have run at least 3 different open-weight code models locally
- [ ] I can explain (mathematically) what `temperature` and `top_p` do to the softmax distribution
- [ ] I can explain the KV cache to a teammate: what it caches, how big it gets, why it dominates memory
- [ ] I have a working `code-completer` CLI with a benchmark mode, on GitHub
- [ ] I have empirical numbers for 4-bit vs bf16 (latency, VRAM, quality on 5 prompts)
- [ ] I've tried `apply_chat_template` and seen what happens if I forget it (gibberish, or much worse output)
- [ ] I've written a retro

---

## Common pitfalls this week

1. **Loading a 7B model in fp32.** That's 28 GB. Use `torch_dtype=torch.bfloat16` or 4-bit. Always.
2. **Forgetting `apply_chat_template`.** Instruct models often degrade badly when given raw text. They want `<|im_start|>user\n...<|im_end|>`-style formatting.
3. **Benchmarking with warm runs only.** First-token latency is dominated by model load + prefill. Measure separately.
4. **Trusting one prompt.** Run your benchmark on 10–20 prompts before claiming "model X is better than model Y."

---

← Previous: [Week 2 — The Transformer](./WEEK-02.md) · → Next: [Week 4 — Prompting, Structured Outputs, Function Calling](./WEEK-04.md)
