# Week 7 — Fine-Tuning Theory + Dataset Preparation

> **Month 2 · Building with LLMs**
> *"Fine-tuning fails 90% of the time. 80% of those failures are caused by the dataset, not the model. So we spend a full week on the dataset."*

---

## Why this week

Most beginners want to *fine-tune the model* in week one and *prepare the dataset* in week zero. They get the order exactly wrong. The skill is not running `trainer.train()`. The skill is shipping a dataset where:

- Examples reflect the exact task you want the model to do
- Inputs and outputs are clean, deduplicated, and licensed
- The format matches what your training framework expects (chat template, completion, instruction-input-output)
- There's a clean train/val/test split with no leakage
- You've measured baseline performance *before* training, so you'll know if training helped

This week is theory + dataset. **Next week** (Week 8) you actually fine-tune.

---

## Learning objectives

By Sunday night you should be able to:

1. Pick the right fine-tuning approach for a problem: full fine-tune vs LoRA vs QLoRA vs instruction tuning vs DPO/RLHF
2. Explain when *not* to fine-tune (often the right answer is "use RAG")
3. Build an instruction dataset (1k–5k rows) for a narrow task
4. Apply data quality controls: dedup, PII scrub, license check, train/val/test split
5. Format the dataset for SFTTrainer (with chat templates) or completion-style training
6. Score a baseline on the test set before any training happens

---

## Day-by-day plan (≈ 2–3 hrs/day)

### Day 1 — When to fine-tune (and when not to)

This is the single most underused skill: knowing *not* to fine-tune. Fine-tuning is expensive, fragile, and often the wrong answer. Read the decision tree below carefully.

```
Problem: model isn't good enough on my task

├── Is it a knowledge problem? (model doesn't know X)
│   └── RAG. Don't fine-tune knowledge — knowledge goes stale fast.
│
├── Is it a style / format problem? (right answer, wrong shape)
│   └── Try prompting / few-shot first. If that fails, instruction-tune.
│
├── Is it a capability problem? (model can't reason this way at all)
│   ├── Small task → LoRA / instruction fine-tune (Week 8)
│   └── Big task → almost certainly out of scope; pick a stronger base model
│
└── Is it a preference problem? (the answer is wrong-ish; I want it to be more X)
    └── Preference fine-tuning: DPO / KTO. Hard mode; later in roadmap.
```

**Read (75 min):**
- [Sebastian Raschka — *When to fine-tune (and when to use prompting/RAG/agents)*](https://magazine.sebastianraschka.com/p/llm-research-insights-instruction) — read parts of the long-form review covering fine-tuning vs prompting
- [Hamel Husain — *Fine-tuning notes index*](https://hamel.dev/notes/llm/finetuning/) — the most-cited modern practitioner reference; read the high-level overview today and bookmark the specific notes for Days 4–5
- [BigDataBoutique — *Fine-Tuning LLMs in 2026: When RAG Isn't Enough*](https://bigdataboutique.com/blog/fine-tuning-llms-when-rag-isnt-enough) — current-base-model practitioner perspective (replaces the 2023 Anyscale-Llama-2 piece, whose hyperparameter advice is now misleading)
- [OpenAI — *Fine-tuning best practices*](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset) — even if you're not using OpenAI, their dataset advice is universal

**Reflect:** For your Week 16 capstone (an autonomous coding agent benchmarked on SWE-bench-Verified-Lite), is there a piece that fine-tuning genuinely helps? Could you get there with prompting + RAG? Or could a small fine-tuned model for one narrow sub-step (e.g., patch generation or critique) materially improve cost/latency? Write 1 paragraph answering this. Be honest.

---

### Day 2 — The taxonomy: SFT, instruction tuning, LoRA, QLoRA, DPO

**Read (90 min):**
- [Sebastian Raschka — *Parameter-Efficient Finetuning With LoRA*](https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html) — the foundational explainer
- [Sebastian Raschka — *Practical Tips for Finetuning LLMs Using LoRA*](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — empirical results, hyperparameter advice
- [Cameron R. Wolfe — *Easily Train a Specialized LLM: PEFT, LoRA, QLoRA, LLaMA-Adapter, and More*](https://cameronrwolfe.substack.com/p/easily-train-a-specialized-llm-peft) — the broader landscape

**Terminology to lock in:**
- **Pretraining** — train on a huge corpus, predict the next token, no task. You almost never do this. (Cost: $$$$$)
- **Continued pretraining / domain adaptation** — pretrain more on your domain corpus (e.g., medical, legal, code). Sometimes useful, rarely necessary.
- **SFT (Supervised Fine-Tuning)** — train on `(input, output)` pairs. The default fine-tuning.
- **Instruction tuning** — SFT, but where the input is an instruction. Pretty much synonymous in practice.
- **LoRA** — instead of updating all weights, learn low-rank update matrices A and B such that `ΔW ≈ BA`. ~0.1–1% trainable params, much faster, smaller adapters.
- **QLoRA** — LoRA on top of a base model quantized to 4-bit. Saves another ~60% VRAM. Slightly slower.
- **DPO / KTO / ORPO** — preference fine-tuning. You give `(prompt, preferred, dispreferred)` instead of `(input, output)`. In 2026:
  - **ORPO** = SFT + preference in one stage. Simplest; works well on small data; the modern default for "just align the output style after SFT."
  - **DPO** = two-stage (SFT, then preference). More flexible, slightly more compute, well-supported across frameworks.
  - **KTO** = binary thumbs-up/down per example instead of pairs. Use when you have implicit user feedback but no pairwise comparisons.
  - **GRPO** (DeepSeek-R1 style) = reward-model-free RL on verifiable rewards (e.g., test pass/fail). Eats coding tasks; expensive but powerful.

  Required references when you do this in Week 8: [Unsloth — *RL / DPO / ORPO / KTO guide*](https://docs.unsloth.ai/basics/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto) · [OpenAI cookbook — *DPO guide*](https://cookbook.openai.com/examples/fine_tuning_direct_preference_optimization_guide)

---

### Day 3 — How to source data

**Read (75 min):**
- [Stanford Alpaca — *A Strong, Replicable Instruction-Following Model*](https://crfm.stanford.edu/2023/03/13/alpaca.html) — **historical foundation** (March 2023). The recipe still teaches the right pattern, but its 52k-from-davinci-003 quality bar is below 2026 standards. Read for the *method*, not as a quality target.
- [Self-Instruct paper (light skim)](https://arxiv.org/abs/2212.10560) — abstract + Figure 1 + section 3 — also a historical foundation
- [Magpie paper (Xu et al., 2024)](https://arxiv.org/abs/2406.08464) — current state-of-the-art for synthetic instruction-data generation; the recipe that replaced Self-Instruct
- [Hamel Husain — *Curating LLM training data*](https://hamel.dev/notes/llm/finetuning/data_cleaning.html) — the modern practitioner playbook
- [HuggingFace blog — *SmolLM3*](https://huggingface.co/blog/smollm3) — peek behind a worked example of dataset curation at scale (FineWeb-Edu, data mix decisions)

**Strategies (pick what fits your project):**

| Source | Cost | Quality | When to use |
|---|---|---|---|
| Hand-write | $$$ (your time) | Best | Always include some — start with 50–200 you write yourself |
| Scrape from existing datasets | $ | Mixed | Find permissively-licensed datasets on HF Hub; filter ruthlessly |
| Synthetic via larger LLM | $$ | Surprisingly good | Generate (input, output) pairs by prompting GPT-4 / Claude. Mix with hand-written. |
| Self-instruct | $$ | Good | LLM generates new instructions from a small seed set |
| Distillation from a teacher | $$ | Very good | Use a strong model's outputs as training data for a smaller model |
| Real production traffic | $ | Best | Once you have a working product, log + label the failures |

For this roadmap your default is **synthetic + hand-curated** — that's how Alpaca-Code, Dolly, and most domain-specific models started.

---

### Day 4 — Build your dataset, Part 1: sourcing

Today you start the actual project (Weekly Project below). Pick a task — *one* narrow task. Examples:

- **Issue text → top-3 candidate files to edit** (the file-localiser sub-task for the Week 16 SWE-bench capstone)
- **Stack trace → likely root cause**
- **Code → docstring**
- **Code → unit tests**
- **Natural language → SQL** for a specific schema
- **Bug report → fix patch**

Start hand-writing 30–50 (input, output) pairs.

Yes, by hand. This is the most valuable thing you'll do all month. You'll surface ambiguities in your task definition that you couldn't have noticed any other way.

---

### Day 5 — Build your dataset, Part 2: scale + clean

**Scale with a teacher model:**
- Take 10 hand-written examples as seeds
- Prompt GPT-4 / Claude to generate 50 more in the same format
- Mix and shuffle

**Clean (75 min):**
- [HuggingFace — *Filter*, *Map*, *Deduplicate* with `datasets`](https://huggingface.co/docs/datasets/process)
- Deduplicate exact matches
- Near-dedup with `datasketch` MinHash + LSH (or `text-dedup` library)
- PII / secret scrub with `detect-secrets` or simple regex (emails, API keys, names)
- Length filter: drop the longest/shortest 1% (likely junk)
- Format check: every row has the expected keys and types
- **Contamination check** — for any test row, search the base model's pretraining-proxy data (or use [LLM Decontaminator](https://github.com/lm-sys/llm-decontaminator) / [Min-K%++](https://github.com/zjysteven/mink-plus-plus)) to confirm it didn't leak in. A 2026 dataset without a contamination report is incomplete.
- **Diversity audit** — embed every prompt, cluster (`hdbscan` or k-means), confirm no single cluster has >5–10% of the data. Synthetic generation tends to collapse into a few templates; this catches it.

**Read (60 min):**
- [HuggingFace — *Data preparation* in the LLM Course](https://huggingface.co/learn/llm-course/chapter11/3) — current best practices
- [BigScience — *Dataset documentation* (model card / dataset card)](https://huggingface.co/docs/hub/datasets-cards)

---

### Day 6 — Format for SFTTrainer + train/val/test split

The format matters because TRL's SFTTrainer wants either:
- **Chat format:** `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`
- **Text completion format:** `{"text": "..."}` — already containing the full templated prompt + response

**Read (45 min):**
- [TRL — *SFTTrainer dataset formats*](https://huggingface.co/docs/trl/sft_trainer#dataset-format-support) — *required reading*
- [HuggingFace — *Chat templates*](https://huggingface.co/docs/transformers/main/en/chat_templating)

**Hands-on (60 min):**
- Convert your dataset to chat format
- 90/5/5 train/val/test split — but *stratified* if your data has categories (e.g., difficulty levels, languages)
- Save to JSONL: `train.jsonl`, `val.jsonl`, `test.jsonl`
- **Test set hygiene:** no row from train should be a near-duplicate of test (run MinHash check)

---

### Day 7 — Baseline + dataset card + retro

**Run a baseline (60 min):**
- For every row in `test.jsonl`, generate an output from the *base* (un-fine-tuned) model
- Compute a simple metric. For code: do the outputs compile? Pass unit tests? For text: BLEU, ROUGE, or human-eval on 20 samples.
- Save `baseline_results.csv`

**Write the dataset card (30 min):**
- Source(s) of data
- Annotation / generation method
- Train / val / test counts
- Known limitations & biases
- License

Now you're ready for Week 8.

---

## Weekly Project — `instruct-dataset-v1`

A high-quality instruction dataset for **one narrow task** — the same one you'll fine-tune on in Week 8.

### Spec

```
instruct-dataset-v1/
  README.md
  DATASET_CARD.md
  data/
    seeds/                 # your hand-written 30-50
    generated/             # LLM-generated examples
    raw/                   # before dedup/cleaning
    processed/
      train.jsonl
      val.jsonl
      test.jsonl
  scripts/
    generate.py            # synthetic generation
    dedup.py
    pii_scrub.py
    split.py
    baseline_eval.py
  baseline_results.csv
  reports/
    quality_audit.md       # 20 random rows, manually graded
```

### Requirements

- At least **1,000** total examples (after dedup), at least **50 hand-written**
- 90/5/5 split (or 80/10/10), stratified if applicable
- Dedup pass: exact + near-duplicate (MinHash)
- PII / secret scrub
- Each example follows a strict schema (`pydantic` or `jsonschema` validated)
- A `DATASET_CARD.md` following [HF dataset card structure](https://huggingface.co/docs/hub/datasets-cards)
- Baseline metric on the test set with the *unfine-tuned* model
- **Contamination check** — log how many test rows had near-matches against the base model's pretraining-proxy data (or run an LLM Decontaminator pass)
- **Diversity audit** — cluster prompt embeddings; confirm no single cluster holds >10% of the data (synthetic generation tends to collapse)
- **Dataset published to the Hugging Face Hub** as a baseline (was stretch, now required — it's 10 lines and makes the artifact public)

### Stretch

- Add an automated **AlpaGasus-style** quality scoring step: prompt a strong LLM to grade each row 1–5; drop the bottom decile
- Add a [Magpie](https://arxiv.org/abs/2406.08464)-style self-synthesis pipeline: generate new instructions by sampling from an aligned model
- Write a `pytest` suite that validates the schema + dedup + diversity invariants of any dataset version

---

## Curated resources

**Concept / decision-making**
- [Sebastian Raschka — *When to fine-tune* (instruction insights)](https://magazine.sebastianraschka.com/p/llm-research-insights-instruction)
- [Hamel Husain — *Fine-tuning notes index*](https://hamel.dev/notes/llm/finetuning/) — the modern practitioner reference
- [BigDataBoutique — *Fine-Tuning LLMs in 2026: When RAG Isn't Enough*](https://bigdataboutique.com/blog/fine-tuning-llms-when-rag-isnt-enough)
- [Cameron R. Wolfe — *Easily Train a Specialized LLM*](https://cameronrwolfe.substack.com/p/easily-train-a-specialized-llm-peft)
- [Sebastian Raschka — *Practical Tips for Finetuning LLMs Using LoRA*](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [Sebastian Raschka — *Parameter-Efficient LLM Finetuning With LoRA*](https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html)
- [Unsloth — *RL / DPO / ORPO / KTO guide*](https://docs.unsloth.ai/basics/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto)
- [OpenAI cookbook — *DPO guide*](https://cookbook.openai.com/examples/fine_tuning_direct_preference_optimization_guide)

**Datasets / data generation**
- [Hamel Husain — *Curating LLM training data*](https://hamel.dev/notes/llm/finetuning/data_cleaning.html) — **the** modern data-curation reference
- [Magpie paper (2024)](https://arxiv.org/abs/2406.08464) — current state-of-the-art for synthetic instruction generation
- [Stanford Alpaca blog](https://crfm.stanford.edu/2023/03/13/alpaca.html) — historical foundation
- [Self-Instruct paper](https://arxiv.org/abs/2212.10560) — historical foundation
- [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) — modern reference SFT dataset
- [argilla/distilabel-intel-orca-dpo-pairs](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs) — modern reference DPO dataset
- [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) — 10k high-quality human-written, the gold standard for tiny datasets
- [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- [OpenAssistant Conversations](https://huggingface.co/datasets/OpenAssistant/oasst1)

**Tooling**
- [Hugging Face `datasets` — Process docs](https://huggingface.co/docs/datasets/process)
- [`text-dedup` (Chenghao Mou)](https://github.com/ChenghaoMou/text-dedup) — production-grade dedup
- [`detect-secrets`](https://github.com/Yelp/detect-secrets)
- [`distilabel`](https://github.com/argilla-io/distilabel) — synthetic data generation pipelines
- [`argilla`](https://argilla.io/) — human-in-the-loop dataset labeling

**Dataset cards & licensing**
- [HF — *Dataset Cards*](https://huggingface.co/docs/hub/datasets-cards)
- [BigCode — *The Stack* and license filtering for code](https://www.bigcode-project.org/docs/about/the-stack/)

---

## "Done when…" checklist

- [ ] I picked one narrow task and wrote down its success criteria
- [ ] I have at least 50 hand-written and 1,000 total examples
- [ ] Train/val/test are split with no leakage, stratified if needed
- [ ] Every example validates against a strict schema
- [ ] Dedup + PII scrub were run, audit log committed
- [ ] I ran a baseline (base model, no fine-tuning) on the test set and have a CSV
- [ ] DATASET_CARD.md is in the repo
- [ ] I've manually graded 20 random rows and 90%+ are "I'd be happy to train on this"

---

## Common pitfalls this week

1. **Skipping the hand-written seeds.** If you can't hand-write 50 good examples, your task definition is unclear and you'll never know if training "worked."
2. **Train/test leakage.** Near-duplicates of test rows in train inflate your eval to 100% while the real-world performance stays the same. Use MinHash.
3. **Generating with one model.** If you generate 5k examples with GPT-4, you're partly teaching your model to mimic GPT-4 *and its biases*. Mix sources.
4. **No baseline.** Without a base-model number you cannot prove fine-tuning helped. The baseline is the first measurement, not the last.
5. **Optimizing for size.** 1k clean examples > 50k noisy examples. Always.

---

← Previous: [Week 6 — Advanced RAG](./WEEK-06.md) · → Next: [Week 8 — LoRA Fine-Tuning Hands-On](./WEEK-08.md)
