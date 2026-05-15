# Week 8 — LoRA Fine-Tuning Hands-On

> **Month 2 · Building with LLMs**
> *"Last week was the dataset. This week is `trainer.train()`. By Sunday night you have a fine-tuned model on the Hugging Face Hub."*

---

## Why this week

Fine-tuning a small model with LoRA is now genuinely doable on free Colab. You ship one this week — *with a measured before/after* — and you instantly understand:

- What's actually being trained (and not)
- Why hyperparameters matter (and which ones matter most)
- Why "loss went down" is not the same as "the model got better at the task"
- How to debug when fine-tuning makes things *worse* (it often does)

You'll use the **TRL + PEFT + Unsloth** stack. This is the modern path. Plain `transformers.Trainer` works but is verbose; Unsloth makes it 2× faster with 60% less VRAM.

---

## Learning objectives

By Sunday night you should be able to:

1. Run a LoRA SFT on a 1–3B model using TRL `SFTTrainer` (with or without Unsloth)
2. Pick reasonable hyperparameters (rank, alpha, lr, epochs, batch size) and defend the choice
3. Diagnose underfitting vs overfitting from a loss curve
4. Merge the LoRA adapter into the base model for deployment
5. Run a before/after eval and write a results report
6. Push the adapter (and optionally the merged model) to the Hugging Face Hub with a model card

---

## Day-by-day plan (≈ 2–3 hrs/day)

### Day 1 — TRL & PEFT walkthrough

**Read (75 min):**
- [TRL — *SFTTrainer*](https://huggingface.co/docs/trl/sft_trainer) — the whole page, twice
- [TRL — *Reducing Memory Usage*](https://huggingface.co/docs/trl/main/en/reducing_memory_usage) — gradient checkpointing, packing, accumulation
- [PEFT — *LoRA conceptual guide*](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [PEFT — *Quicktour*](https://huggingface.co/docs/peft/quicktour)

**Reflect:**
- For a 1B base model with LoRA rank 16, target modules = `["q_proj","k_proj","v_proj","o_proj"]`, what fraction of params are trainable? (Hint: typically < 1%.)

---

### Day 2 — Unsloth (faster, less VRAM)

Unsloth is a single import that makes TRL training 2× faster with much less memory, by replacing the attention kernels and fusing operations. It's free, open source, and works in Colab.

**Read (45 min):**
- [Unsloth — *Welcome / Quickstart*](https://docs.unsloth.ai/) (top of the docs)
- [HuggingFace — *Make LLM Fine-tuning 2x faster with Unsloth and 🤗 TRL*](https://huggingface.co/blog/unsloth-trl)
- [Stephen Diehl — *A Rapid Tutorial on Unsloth*](https://www.stephendiehl.com/posts/unsloth/) — short, no-nonsense

**Hands-on (60 min):**
- Open a fresh Colab T4 notebook
- Install:
  ```bash
  pip install -U unsloth trl peft bitsandbytes accelerate datasets
  ```
- Run a current Unsloth notebook end-to-end. Don't grab an old hardcoded link — pick from the **live index** at [docs.unsloth.ai/get-started/unsloth-notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks). For 2026, pick a Qwen3 or Llama-3.3 SFT notebook on a T4.
- Watch the loss come down. Generate a sample. Done.

---

### Day 3 — Run your first *real* fine-tune

Today: take your Week 7 dataset and SFT a small model on it.

**Recommended setup (Colab T4 / 16GB VRAM, May 2026)**:
- Base model — pick one (all current as of May 2026; check each model card for the exact `target_modules` list):
  - `Qwen/Qwen3-1.7B-Base` or `Qwen/Qwen3-Coder-1.5B-Instruct` (best for code tasks)
  - `meta-llama/Llama-3.3-3B-Instruct`
  - `microsoft/Phi-4-mini-instruct`
  - `google/gemma-3-1b-it`
- Format: chat template (you converted to this in Week 7)
- LoRA config: `r=16`, `alpha=32`, `dropout=0.05`, `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` *(default for Llama-3/Qwen3-style architectures; if you pick Phi-4-mini or Gemma-3, check the model card — module names differ)*
- Training: 1–3 epochs, LR `2e-4`, batch size 2 + grad accumulation 4 (effective batch 8)
- `bf16=True`, gradient checkpointing on

**Run it. Save the loss curve. Save the adapter to `./outputs/lora-v1/`.**

**Hands-on (~3 hr including training time):**
- Expect training to take 30–90 min on a T4 for ~1k examples × 3 epochs
- If it OOMs: switch to QLoRA (4-bit base), reduce batch size, reduce sequence length

---

### Day 4 — Evaluate, compare, iterate

This is the day most beginners skip and where the actual learning lives.

**Steps:**
1. Generate outputs for every row in `test.jsonl` with both base and fine-tuned models
2. Compute the same metric as your Week-7 baseline
3. Tabulate:
   ```
   Model          test_accuracy  test_pass_rate  avg_tokens_out  avg_latency
   base           0.30           0.22            180             1.4s
   lora_v1        0.62           0.55            145             1.4s
   ```
4. Inspect 20 failure cases from `lora_v1`. Categorize:
   - "Right idea, wrong format" → instruction-following issue → maybe more epochs / better prompts
   - "Confidently wrong" → likely data quality issue
   - "Garbage" → catastrophic forgetting or bad LR

**Read (45 min) on debugging:**
- [Sebastian Raschka — *Practical Tips for Finetuning LLMs Using LoRA* — section on rank/alpha](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [Hamel Husain — *LLM Evals FAQ*](https://hamel.dev/blog/posts/evals-faq/) — the practitioner reference for whether your fine-tune actually worked
- [Hamel Husain — *LLM-as-a-Judge*](https://hamel.dev/blog/posts/llm-judge/) — for the parts of your task where no clean metric exists

> **Catastrophic-forgetting sanity check:** before you call `lora_v1` "better", give it 10 *off-task* prompts (e.g., "write a haiku", "summarize this paragraph"). If it produces gibberish or drops into the fine-tuning shape no matter what, your LR is too high or you trained too many epochs. Always include this check.

---

### Day 5 — Iterate (v2)

Now that you've seen v1 fail in specific ways, you change *one thing at a time* and retrain.

**A short list of high-leverage changes:**
- **More epochs** (2 → 5) — usually the first knob
- **Higher rank** (16 → 32 or 64) — for harder tasks; watch for overfitting
- **Lower learning rate** (2e-4 → 5e-5) — if loss is noisy or training diverges
- **More target modules** — apply LoRA to MLP layers too (`gate_proj`, `up_proj`, `down_proj`)
- **Better data** — add 100 hand-written examples covering the failure cases you saw on Day 4
- **Preference tuning on top.** Once SFT is decent, run **ORPO** or **DPO** with 100–500 `(preferred, dispreferred)` pairs you handcraft from v1's failures. This is the highest-leverage modern upgrade. See the [Unsloth RL guide](https://docs.unsloth.ai/basics/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto).
- **NEFTune** — adds Gaussian noise to embeddings during training. The original 2023 result has *mixed* replication on Llama-3/Qwen3 bases — try it, measure it, don't trust it on faith.

Retrain. Re-evaluate. Tabulate `lora_v2` next to `lora_v1` and `base`.

**Read (45 min):**
- [Unsloth — *RL / DPO / ORPO / KTO guide*](https://docs.unsloth.ai/basics/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto) — the canonical 2026 reference for preference tuning on top of SFT
- [OpenAI cookbook — *DPO guide*](https://cookbook.openai.com/examples/fine_tuning_direct_preference_optimization_guide)
- [HF blog — *NEFTune: Noisy Embeddings Improve Instruction Finetuning*](https://huggingface.co/papers/2310.05914) — read skeptically

---

### Day 6 — Merge, save, publish

**Merge the adapter** (so the deployed model is a single artifact):
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Coder-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "./outputs/lora-v2/")
merged = model.merge_and_unload()
merged.save_pretrained("./merged/")
```

**Push to the Hub:**
```bash
huggingface-cli login
huggingface-cli upload your-username/qwen3-coder-1.5b-mytask ./merged/
```

**Write a model card** (`README.md` in the model repo):
- Base model + license
- Training data (link to your Week-7 dataset)
- Hyperparameters
- Evaluation results vs baseline
- Known limitations + failure cases
- Intended use

**Read (15 min):**
- [HuggingFace — *Model Cards*](https://huggingface.co/docs/hub/model-cards)

---

### Day 7 — Reflect, retro, prepare for Week 9

Today: polish the repo. Write a results blog-post-style retro:

- What changed v1 → v2?
- What surprised you?
- What would v3 do?

---

## Weekly Project — `lora-codetune`

A reproducible LoRA fine-tune on **your Week-7 dataset**, with measurable before/after results.

### Spec

```
lora-codetune/
  README.md
  train.py                  # Unsloth + TRL SFTTrainer
  eval.py                   # base vs lora_v1 vs lora_v2 head-to-head
  configs/
    lora_v1.yaml
    lora_v2.yaml
  outputs/
    lora_v1/                # adapter weights + tokenizer
    lora_v2/
  reports/
    loss_curves.png
    eval_v1.md
    eval_v2.md
    final_report.md         # the headline document
  model_card.md             # what you'll publish on the Hub
```

### Requirements

- One reproducible `train.py` script driven by a YAML config
- At least two configs (`v1`, `v2`) where `v2` is a justified iteration on `v1`
- Eval script that runs base, v1, v2 on the same `test.jsonl` and produces a markdown table
- **Catastrophic-forgetting trace:** for each run, generate outputs on 10 *off-task* prompts; log them in the report so anyone can see the model didn't get worse at general capability
- Loss curves saved to PNG
- Final report includes:
  - Hyperparameters for each run
  - Aggregate metrics + **inference latency and peak VRAM** at serve-time (a hiring manager actually cares about these)
  - 5 side-by-side example outputs (base / v1 / v2)
  - At least 3 failure cases with analysis
- Adapter pushed to the HF Hub with a model card

### Stretch

- Add a third run: **QLoRA** (4-bit base) — compare quality, VRAM, training time
- Train a second model (e.g., `Phi-4-mini` or `Gemma-3-1b`) on the same dataset and report which base learns the task best
- **Run one preference-tuning step (ORPO or DPO)** on top of v2 with `(preferred, dispreferred)` pairs you handcraft from v2 failures — this is the 2026 progression that separates a serious project from a tutorial-grade one
- (Advanced) Try **GRPO** for a verifiable reward (e.g., test pass/fail on a code task) — DeepSeek-R1-style

---

## Curated resources

**Official docs**
- [TRL — *SFTTrainer*](https://huggingface.co/docs/trl/sft_trainer)
- [TRL — *Reducing Memory Usage*](https://huggingface.co/docs/trl/main/en/reducing_memory_usage)
- [TRL — *PEFT integration examples*](https://huggingface.co/docs/trl/main/en/peft_integration)
- [PEFT — *Quicktour*](https://huggingface.co/docs/peft/quicktour)
- [PEFT — *LoRA conceptual guide*](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [Unsloth docs](https://docs.unsloth.ai/)
- [Unsloth — *Notebooks index (live, maintained)*](https://docs.unsloth.ai/get-started/unsloth-notebooks)
- [Unsloth — *RL / DPO / ORPO / KTO / GRPO guide*](https://docs.unsloth.ai/basics/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto)

**Tutorials**
- [HF blog — *Make LLM Fine-tuning 2× faster with Unsloth and TRL*](https://huggingface.co/blog/unsloth-trl)
- [Stephen Diehl — *A Rapid Tutorial on Unsloth*](https://www.stephendiehl.com/posts/unsloth/) (2024 — fundamentals still apply)
- [Stephen Diehl — *Fine-tuning with ORPO and Unsloth*](https://www.stephendiehl.com/posts/orpo/)
- [Sebastian Raschka — *Practical Tips for Finetuning LLMs Using LoRA*](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — *the* tuning-numbers reference
- [Sebastian Raschka — *Parameter-Efficient LLM Finetuning With LoRA*](https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html)
- [Hamel Husain — *LLM Evals FAQ*](https://hamel.dev/blog/posts/evals-faq/) — required reading before claiming "v2 is better"
- [OpenAI cookbook — *DPO guide*](https://cookbook.openai.com/examples/fine_tuning_direct_preference_optimization_guide)

**Papers (skim)**
- [Hu et al. — *LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685)
- [Jiang et al. — *NEFTune: Noisy Embeddings Improve Instruction Finetuning*](https://arxiv.org/abs/2310.05914)

**Free Colab notebooks (don't sleep on these)**
- [Unsloth — live notebooks index](https://docs.unsloth.ai/get-started/unsloth-notebooks) — covers Qwen3, Llama-3.3, Gemma-3, Phi-4, and GRPO recipes; pick the one matching your base
- [Hugging Face Open-Source AI Cookbook](https://huggingface.co/learn/cookbook/index) — alternative recipes if you prefer plain TRL

---

## "Done when…" checklist

- [ ] I have a reproducible `train.py` that finishes a run on Colab T4
- [ ] Loss curves are saved and look sensible (smooth-ish, going down)
- [ ] `eval.py` runs base / v1 / v2 head-to-head and produces a markdown table
- [ ] I can identify the failure modes of v1 by category
- [ ] v2 is a *justified* iteration on v1 (I changed one or two things on purpose)
- [ ] Adapter is pushed to the Hub with a model card
- [ ] Final report is honest about what didn't work
- [ ] I've written a retro

---

## Common pitfalls this week

1. **Looking only at loss.** A model whose loss went from 1.5 → 0.4 can be *worse* at your task. Always run the eval.
2. **Catastrophic forgetting.** Too-aggressive fine-tuning (especially with high lr or too many epochs on a tiny dataset) can wreck the base capabilities. Always sanity-check by giving the model an *off-task* prompt and ensuring it doesn't produce gibberish.
3. **Wrong target modules.** LoRA on only `q_proj/v_proj` (the original paper) is suboptimal for instruction tuning. Apply LoRA to MLP modules too.
4. **No eval on the test set during dev.** It's OK to peek at val loss; it is **not** OK to keep tweaking until test numbers go up — that's overfitting to test.
5. **Forgetting `tokenizer.padding_side = "right"`** (or left for some models) — gets you mystery loss explosions.
6. **Not saving the tokenizer with the adapter.** When you reload weeks later, you need both.

---

## End-of-Month-2 checkpoint

You now have **8 GitHub repos** and **1 published model** on Hugging Face:
1. `token-budget`
2. `mini-gpt`
3. `code-completer`
4. `structured-code-reviewer`
5. `docs-rag`
6. `advanced-rag`
7. `instruct-dataset-v1`
8. `lora-codetune` (with adapter on the Hub)

This is already a credible portfolio. Month 3 is where you make the system "feel like production."

---

← Previous: [Week 7 — Fine-Tuning Theory + Dataset Prep](./WEEK-07.md) · → Next: [Week 9 — QLoRA & Quantization (GPTQ, AWQ, GGUF)](./WEEK-09.md)
