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
3. Quantize a model with `auto-gptq`, `autoawq`, and `llama.cpp` (GGUF) and run it locally
4. Run a **QLoRA** fine-tune that wouldn't have fit in plain LoRA
5. Benchmark base / LoRA / QLoRA / GPTQ / AWQ / GGUF on quality, VRAM, latency, throughput
6. Recommend a quantization for a given deployment target (laptop, T4, A10, H100)

---

## Day-by-day plan (≈ 2–3 hrs/day)

### Day 1 — Numerical formats (the prerequisite)

**Read (60 min):**
- [HF blog — *A Gentle Introduction to 8-bit Matrix Multiplication for LLMs*](https://huggingface.co/blog/hf-bitsandbytes-integration) — the LLM.int8() paper made accessible
- [Maarten Grootendorst — *A Visual Guide to Quantization*](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) — outstanding visuals; absolute must-read
- [Lightning AI — *Understanding LLM Quantization*](https://lightning.ai/pages/community/tutorial/lora-llm/) (skim relevant section)

**Concepts to lock in:**

| Format | Bits | Range | Notes |
|---|---|---|---|
| FP32 | 32 | huge | Default in research / weights; almost never in serving |
| FP16 | 16 | ±65k | Original mixed-precision training |
| BF16 | 16 | huge (FP32-like) | Now standard; bigger range than FP16, fewer NaNs |
| FP8 | 8 | small | H100+; coming standard for serving |
| INT8 | 8 | 0–255 | LLM.int8(), good baseline |
| INT4 | 4 | 16 levels | GPTQ / AWQ / NF4 / GGUF; the sweet spot for consumer serving |

---

### Day 2 — GPTQ, AWQ, GGUF compared

**Read (90 min):**
- [Local AI Master — *GGUF vs GPTQ vs AWQ Compared: Best Quantization 2026*](https://localaimaster.com/blog/quantization-explained)
- [TensorRigs — *LLM Quantization Explained: GGUF vs GPTQ vs AWQ*](https://tensorrigs.com/blog/llm-quantization-guide/)
- [Maarten Grootendorst — *A Visual Guide to Quantization*](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) — re-read with the algorithm chapter in mind

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
- [llama.cpp — *Quantization*](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)
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

### Day 4 — Quantize with AWQ and GPTQ (Part 2)

**Read (45 min):**
- [autoawq README](https://github.com/casper-hansen/AutoAWQ)
- [auto-gptq README](https://github.com/AutoGPTQ/AutoGPTQ)
- [vLLM — *AutoAWQ*](https://docs.vllm.ai/en/latest/features/quantization/auto_awq.html)
- [vLLM — *GPTQ*](https://docs.vllm.ai/en/latest/features/quantization/gptq.html)

**Hands-on (~90 min):**
```python
# AWQ
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
model = AutoAWQForCausalLM.from_pretrained("your-base-or-merged-model")
tokenizer = AutoTokenizer.from_pretrained(...)
model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4})
model.save_quantized("./awq-4bit/")
```
- Load with `transformers` and sample
- (Bonus: load with vLLM if you have time — we'll do this properly in Week 10)

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
$ python benchmark.py --model qwen-coder-1.5b-design-to-code --eval test.jsonl

variant                      VRAM    1st-tok  tok/s   pass@1  size-on-disk
base (bf16)                  3.6 GB  280 ms   28.4    0.62    3.2 GB
LoRA-merged (bf16)           3.6 GB  280 ms   28.4    0.78    3.2 GB
QLoRA-merged (nf4)           1.4 GB  290 ms   30.1    0.77    1.1 GB
GPTQ-4bit                    1.5 GB  220 ms   42.5    0.74    1.0 GB
AWQ-4bit                     1.5 GB  210 ms   44.2    0.76    1.0 GB
GGUF Q4_K_M (llama.cpp)      1.2 GB  180 ms   38.7    0.76    1.0 GB
GGUF Q2_K (llama.cpp)        0.9 GB  170 ms   45.1    0.55    0.7 GB ← quality drop
```

### Requirements

- One script that loads each variant and runs the same prompts
- Measure for each variant:
  - **VRAM** peak during eval
  - **First-token latency** (avg over 20 prompts)
  - **Decoding throughput** (tok/s)
  - **Eval pass-rate** on your Week-7 test set
  - **Size on disk**
- At least 5 variants must include base, LoRA-merged, and three quantizations
- A `report.md` with the table + 3 short paragraphs:
  - **Which variant I'd ship to laptop users** (and why)
  - **Which variant I'd ship to a GPU server** (and why)
  - **Where quality breaks down** (which variant first shows degradation, on what kind of task)

### Stretch

- Add FP8 if you have H100 access (or use a hosted runtime)
- Compare KV-cache quantization (`--cache-quant int8` in vLLM) as well
- Try **SmoothQuant** or **AQLM** for an additional point on the Pareto frontier

---

## Curated resources

**Concepts**
- [Maarten Grootendorst — *A Visual Guide to Quantization*](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) — **the** introductory reference
- [HF blog — *A Gentle Introduction to 8-bit Matrix Multiplication*](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [HF blog — *4-bit quantization and QLoRA*](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

**Comparisons**
- [Local AI Master — *GGUF vs GPTQ vs AWQ Compared (2026)*](https://localaimaster.com/blog/quantization-explained)
- [TensorRigs — *LLM Quantization Explained (2026)*](https://tensorrigs.com/blog/llm-quantization-guide/)
- [Arxiv (2026) — *Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct*](https://arxiv.org/html/2601.14277v1)
- [VRLA Tech — *INT4, INT8, FP8, AWQ, and GPTQ (2026)*](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/)

**Method docs**
- [autoawq](https://github.com/casper-hansen/AutoAWQ)
- [auto-gptq](https://github.com/AutoGPTQ/AutoGPTQ)
- [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index)
- [llama.cpp — *Quantize*](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)
- [Qwen — *Using llama.cpp*](https://qwen.readthedocs.io/en/latest/quantization/llama.cpp.html)
- [HF blog — *GGUF and interaction with Transformers*](https://huggingface.co/blog/gguf-transformers)

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
3. **Quantizing and forgetting to use the right loader.** Quantized weights must be loaded with the matching library (`AutoAWQForCausalLM` for AWQ, `AutoGPTQForCausalLM` for GPTQ).
4. **Disk size = VRAM.** Not quite — there's always overhead (KV cache, activations). Always measure VRAM in practice with `nvidia-smi`.
5. **Quantizing chat templates away.** Always save and ship the *original* tokenizer with the quantized model.

---

← Previous: [Week 8 — LoRA Fine-Tuning](./WEEK-08.md) · → Next: [Week 10 — Inference Servers (vLLM, PagedAttention, KV Cache)](./WEEK-10.md)
