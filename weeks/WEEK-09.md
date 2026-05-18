# Week 9 — QLoRA & Quantization (GPTQ, AWQ, GGUF)

> **Month 3 · Production LLM Systems**
> *"There are only two reasons a model doesn't fit on your hardware: it's actually too big, or you haven't quantized it yet. Usually the second."*

---

## Why this week

A 7B model is ~14 GB in `fp16`. The same model in 4-bit (`Q4_K_M` GGUF, or 4-bit AWQ) is ~4 GB and still gives you 95%+ of the quality on most tasks. Quantization is what makes "run an LLM on a laptop" or "fit a 70B model on a single H100" possible.

This week you go from "I clicked 4-bit in `BitsAndBytesConfig`" to "I can pick the right quantization for a given task and measure exactly what it costs in quality."

You'll also do **QLoRA** — LoRA on top of a 4-bit base — so you can fine-tune larger models on smaller GPUs.

---

## Learning objectives

By Sunday night you should be able to:

1. Explain what FP16, BF16, FP8, INT8, INT4 represent and when each is used
2. Explain how AWQ, GPTQ, and GGUF differ in algorithm and use-case
3. Quantize a model with `llm-compressor` (AWQ / GPTQ / FP8), `GPTQModel`, and `llama.cpp` (GGUF) and run it locally
4. Run a **QLoRA** fine-tune that wouldn't have fit in plain LoRA
5. Benchmark base / LoRA / QLoRA / GPTQ / AWQ / GGUF on quality, VRAM, latency, throughput
6. Recommend a quantization for a given deployment target (laptop, T4, A10, H100)

---

## Day-by-day plan (≈ 2–3 hrs/day)

### Day 1 — Numerical formats (the prerequisite)

**Read (60 min):**
- [HF blog — *A Gentle Introduction to 8-bit Matrix Multiplication for LLMs*](https://huggingface.co/blog/hf-bitsandbytes-integration) — the LLM.int8() paper made accessible
- [Maarten Grootendorst — *A Visual Guide to Quantization*](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) — outstanding visuals; absolute must-read
- [Sebastian Raschka — *Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments*](https://lightning.ai/pages/community/lora-insights/) — the LoRA/QLoRA quantization-tradeoff reference

> **2026 library landscape:** AutoAWQ was **archived May 2025**; AutoGPTQ has stopped development. The maintained replacements are [`vllm-project/llm-compressor`](https://github.com/vllm-project/llm-compressor) (AWQ, GPTQ, FP8, W8A8) and [`ModelCloud/GPTQModel`](https://github.com/ModelCloud/GPTQModel) (GPTQ specifically, being upstreamed into Transformers/Optimum/PEFT). The classic repos remain useful for understanding *the algorithms* historically, but ship with `llm-compressor`.

**Concepts to lock in:**

| Format | Bits | Range | Notes |
|---|---|---|---|
| FP32 | 32 | huge | Default in research / weights; almost never in serving |
| FP16 | 16 | ±65k | Original mixed-precision training |
| BF16 | 16 | huge (FP32-like) | Now standard; bigger range than FP16, fewer NaNs |
| FP8 | 8 | small | H100/B200; **the 2026 default for serving** when you have the hardware |
| INT8 | 8 | 0–255 | LLM.int8(), good baseline |
| INT4 | 4 | 16 levels | GPTQ / AWQ / NF4 / GGUF; the sweet spot for consumer serving |

---

### Day 2 — GPTQ, AWQ, GGUF compared

**Read (90 min):**
- [HF blog — *Overview of natively supported quantization schemes*](https://huggingface.co/blog/overview-quantization-transformers) — the vendor-neutral comparison table
- [Maarten Grootendorst — *A Visual Guide to Quantization*](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) — re-read with the algorithm chapter in mind
- [Kurt — *Which Quantization Should I Use? A Unified Evaluation on Llama-3.1-8B-Instruct*](https://arxiv.org/abs/2601.14277) — the only empirical apples-to-apples evaluation worth citing

**The three-line summary**

| Method | Where it shines | Where it loses |
|---|---|---|
| **GGUF** (`llama.cpp`) | CPU + consumer GPU, Apple Silicon, Ollama, LM Studio | GPU throughput at scale |
| **AWQ** | Quality-sensitive code/creative tasks; activation-aware → preserves the most important channels | A bit slower to quantize than GPTQ |
| **GPTQ** | Raw NVIDIA-GPU throughput | Code/long-context quality slightly behind AWQ |

**Default picks** (memorize):
- **Laptop / Mac:** GGUF `Q4_K_M` via Ollama or `llama.cpp`
- **Production GPU serving:** AWQ via vLLM (or FP8 if you have H100/H200)
- **Single 24GB GPU, fast experimentation:** AWQ or GPTQ via `transformers`

---

### Day 3 — Quantize a model with all three (Part 1: GGUF)

**Read (30 min):**
- [llama.cpp — *Quantization*](https://github.com/ggml-org/llama.cpp/blob/master/examples/quantize/README.md)
- [HF blog — *GGUF and interaction with Transformers*](https://huggingface.co/blog/gguf-transformers)
- [Ollama — *Model File Format*](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)

**Hands-on (~90 min):**
- Take a small model (1.5–3B) you fine-tuned in Week 8 (or any open-weight one)
- Convert to GGUF using `llama.cpp/convert_hf_to_gguf.py`
- Quantize: try `Q8_0`, `Q5_K_M`, `Q4_K_M`, `Q2_K`
- Run locally with `llama.cpp` server or import into Ollama:
  ```bash
  ollama create mymodel -f Modelfile
  ollama run mymodel
  ```
- Sample 5 prompts on each. Note VRAM and tokens/sec.

---

### Day 4 — Quantize with AWQ, GPTQ, and FP8 via `llm-compressor`

**Read (60 min):**
- [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) — the maintained 2026 path
- [llm-compressor — *AWQ example*](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a16/)
- [llm-compressor — *FP8 dynamic example*](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w8a8_fp8/)
- [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel) — maintained GPTQ fork, used by HF Transformers
- [vLLM — *AutoAWQ docs*](https://docs.vllm.ai/en/latest/features/quantization/auto_awq.html) (still useful as a how-to-load reference)

**Hands-on (~90 min):**
```python
# AWQ via llm-compressor (the maintained path)
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen3-Coder-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
tok = AutoTokenizer.from_pretrained(model_id)

oneshot(
    model=model, tokenizer=tok,
    dataset="open_platypus",  # tiny calibration set
    recipe=AWQModifier(bits=4, group_size=128),
    output_dir="./awq-4bit/",
)
```
- Load the resulting checkpoint with vLLM (`--quantization awq`) and sample
- For an FP8 run (if you have H100/L4/4090-class hardware), swap `AWQModifier` for the dynamic FP8 recipe in the examples linked above

> **Pre-quantized model hubs:** for production, you'll often pull pre-quantized checkpoints from [Red Hat AI's HF org](https://huggingface.co/RedHatAI) (FP8/AWQ/GPTQ) or community GGUFs. Browse before you re-quantize from scratch.

---

### Day 5 — QLoRA fine-tuning

QLoRA = LoRA on a base model quantized to 4-bit (typically NF4, from the bitsandbytes library). It's how people fine-tune 7B–70B on consumer GPUs.

**Read (45 min):**
- [Tim Dettmers et al. — *QLoRA: Efficient Finetuning of Quantized LLMs*](https://arxiv.org/abs/2305.14314) — abstract + section 3
- [HF blog — *Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA*](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- [Sebastian Raschka — *Practical Tips* (revisit) — the QLoRA section](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)

**Hands-on (~2 hr):**
- Take your Week 8 training script
- Switch the base from bf16 to 4-bit (NF4):
  ```python
  from transformers import BitsAndBytesConfig
  bnb = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True,
  )
  ```
- (Use Unsloth's `FastLanguageModel.from_pretrained(... load_in_4bit=True)` for the easy path)
- Retrain. Compare time, VRAM, final eval to LoRA.

Expected: QLoRA uses ~33% less memory but trains ~1.4× slower.

---

### Day 6 — Build the weekly project (Part 1)

See **Weekly Project** below. Today: set up the benchmark harness and run base + LoRA + QLoRA.

---

### Day 7 — Build the weekly project (Part 2) + write-up

Today: add the quantized variants (GPTQ / AWQ / GGUF) and write the report.

---

## Weekly Project — `model-zoo-benchmark`

Take *your* model from Weeks 7–8 and run a full quantization sweep. Produce one canonical comparison table you can drop into any future model README.

### Spec

```
$ python benchmark.py --model qwen3-coder-1.5b-mytask --eval test.jsonl

variant                       VRAM    1st-tok  tok/s   pass@1  size-on-disk
base (bf16)                   3.6 GB  280 ms   28.4    0.62    3.2 GB
LoRA-merged (bf16)            3.6 GB  280 ms   28.4    0.78    3.2 GB
QLoRA-merged (nf4)            1.4 GB  290 ms   30.1    0.77    1.1 GB
FP8 (llm-compressor)          1.9 GB  200 ms   46.0    0.78    1.7 GB ← if you have H100/L4/4090
GPTQ-4bit (GPTQModel)         1.5 GB  220 ms   42.5    0.74    1.0 GB
AWQ-4bit (llm-compressor)     1.5 GB  210 ms   44.2    0.76    1.0 GB
GGUF Q4_K_M (llama.cpp)       1.2 GB  180 ms   38.7    0.76    1.0 GB
GGUF Q2_K (llama.cpp)         0.9 GB  170 ms   45.1    0.55    0.7 GB ← quality drop
+ KV-cache fp8 on AWQ-4bit    1.1 GB  205 ms   47.1    0.76    1.0 GB ← cheap win
```

### Requirements

- One script that loads each variant and runs the same prompts
- Measure for each variant:
  - **VRAM** peak during eval
  - **First-token latency** (avg over 20 prompts)
  - **Decoding throughput** (tok/s)
  - **Eval pass-rate** on your Week-7 test set
  - **Size on disk**
- At least 6 variants must include base, LoRA-merged, **FP8** (required if you have H100/L4/4090-class hardware), and three INT4 quantizations (AWQ / GPTQ / GGUF)
- **One KV-cache quantization row** (`--kv-cache-dtype fp8` in vLLM, or equivalent) — this is the cheapest 2026 lever and the project should measure it explicitly
- A `report.md` with the table + 3 short paragraphs:
  - **Which variant I'd ship to laptop users** (and why)
  - **Which variant I'd ship to a GPU server** (and why)
  - **Where quality breaks down** (which variant first shows degradation, on what kind of task)

### Stretch

- Try **SmoothQuant**, **AQLM**, **HQQ**, or **AutoRound** for an additional Pareto point
- Pull a pre-quantized FP8 checkpoint from [Red Hat AI](https://huggingface.co/RedHatAI) and compare to your home-quantized version

---

## Curated resources

**Concepts**
- [Maarten Grootendorst — *A Visual Guide to Quantization*](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) — **the** introductory reference
- [HF blog — *A Gentle Introduction to 8-bit Matrix Multiplication*](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [HF blog — *4-bit quantization and QLoRA*](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

**Comparisons & evaluation**
- [HF blog — *Overview of natively supported quantization schemes*](https://huggingface.co/blog/overview-quantization-transformers) — canonical vendor-neutral table
- [Kurt et al. — *Which Quantization Should I Use? A Unified Evaluation on Llama-3.1-8B-Instruct*](https://arxiv.org/abs/2601.14277) — the honest empirical reference
- [Baseten — *33% faster LLM inference with FP8 quantization*](https://www.baseten.co/blog/33-faster-llm-inference-with-fp8-quantization/) — why FP8 on H100/B200 is the 2026 default

**Method docs (maintained 2026 path)**
- [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) — AWQ, GPTQ, FP8, W8A8 — the maintained library
- [llm-compressor — examples](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/) (AWQ, GPTQ, FP8 dynamic, KV-cache fp8)
- [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel) — maintained GPTQ fork
- [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index) — for NF4 / QLoRA
- [llama.cpp — *Quantize*](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [Qwen — *Using llama.cpp*](https://qwen.readthedocs.io/en/latest/quantization/llama.cpp.html)
- [HF blog — *GGUF and interaction with Transformers*](https://huggingface.co/blog/gguf-transformers)
- [Red Hat AI on Hugging Face](https://huggingface.co/RedHatAI) — pre-quantized FP8/AWQ/GPTQ checkpoints for production
- [vLLM — *FP8 docs*](https://docs.vllm.ai/en/latest/features/quantization/fp8/)

**Historical (now archived — read for algorithm understanding only)**
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) — archived May 2025
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) — development stopped

**Papers (skim)**
- [Dettmers et al. — *QLoRA: Efficient Finetuning of Quantized LLMs*](https://arxiv.org/abs/2305.14314)
- [Frantar et al. — *GPTQ: Accurate Post-Training Quantization*](https://arxiv.org/abs/2210.17323)
- [Lin et al. — *AWQ: Activation-aware Weight Quantization*](https://arxiv.org/abs/2306.00978)
- [Xiao et al. — *SmoothQuant*](https://arxiv.org/abs/2211.10438)

---

## "Done when…" checklist

- [ ] I can explain INT8 vs INT4 vs NF4 in one sentence each
- [ ] I have quantized a model in at least 3 formats (GGUF + AWQ + GPTQ ideally)
- [ ] I have run a QLoRA fine-tune and have a number for "I saved X GB of VRAM compared to LoRA"
- [ ] My `model-zoo-benchmark` repo runs end-to-end and produces the canonical table
- [ ] I can recommend a quantization for laptop vs GPU server vs Apple Silicon
- [ ] I've identified the first format where quality clearly degrades on my task
- [ ] I've written a retro

---

## Common pitfalls this week

1. **"Lower bit = always worse."** Often false. AWQ-4 sometimes beats GPTQ-8 on code. Trust measurements.
2. **Comparing quality on 3 prompts.** Use at least 50 to reach signal.
3. **Quantizing and forgetting to use the right loader.** For production, load AWQ / GPTQ / FP8 checkpoints with **vLLM** (`--quantization awq|gptq|fp8`) or via **Transformers' built-in integration with `GPTQModel` / `llm-compressor`**. The legacy `AutoAWQForCausalLM` / `AutoGPTQForCausalLM` classes still work but their parent libraries are archived — don't build new code around them.
4. **Disk size = VRAM.** Not quite — there's always overhead (KV cache, activations). Always measure VRAM in practice with `nvidia-smi`.
5. **Quantizing chat templates away.** Always save and ship the *original* tokenizer with the quantized model.

---

← Previous: [Week 8 — LoRA Fine-Tuning](./WEEK-08.md) · → Next: [Week 10 — Inference Servers (vLLM, PagedAttention, KV Cache)](./WEEK-10.md)
