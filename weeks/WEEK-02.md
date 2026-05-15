# Week 2 — The Transformer (Attention, Decoder-Only)

> **Month 1 · Foundations of LLMs**
> *"If you can't draw a transformer block on a napkin from memory by Sunday, you didn't finish this week."*

---

## Why this week

Every modern LLM — GPT-4o, Claude, Llama-3, Qwen, DeepSeek — is a stack of transformer decoder blocks. If you understand one block in detail, you understand all of them. The differences between models are mostly:

- How many blocks
- How wide each block is
- What flavor of attention (vanilla, GQA, MQA, FlashAttention)
- What positional encoding (absolute, RoPE, ALiBi)
- What normalization (LayerNorm vs RMSNorm)
- What activation in the MLP (GELU vs SwiGLU)

This week you go from "I've heard of attention" to "I've trained a tiny GPT on my own data and I know what every line does."

This is the **single most important week** in the roadmap. Don't rush it.

---

## Learning objectives

By Sunday night you should be able to:

1. Draw the transformer block from memory: residual stream, multi-head attention, MLP, two LayerNorms
2. Explain Q/K/V and why the scaled dot-product is divided by `√d_k`
3. Explain causal masking and why decoder-only models can't see the future
4. Explain why multi-head attention exists (and isn't just one big head)
5. Read every line of `nanoGPT`'s `model.py` and explain what it does
6. Train a tiny character-level GPT on your own corpus and sample from it
7. Explain the differences between encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) models

---

## Day-by-day plan (≈ 2–3 hrs/day)

### Day 1 — Visual intuition

**Watch (45 min):**
- [3Blue1Brown — *But what is a GPT? Chapter 5*](https://www.youtube.com/watch?v=wjZofJX0v4M) (27 min)
- [3Blue1Brown — *Attention in transformers, step-by-step — Chapter 6*](https://www.youtube.com/watch?v=eMlx5fFNoYc) (26 min)

**Read (45 min):**
- [Jay Alammar — *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) — read it slowly, top to bottom. This is the most famous transformer explainer for a reason.

**Reflect:**
- Open your notes and sketch the transformer block. Then sketch the data flow for **one token** through one block. Compare with Alammar's diagrams.

---

### Day 2 — More visual intuition, then the math

**Watch (30 min):**
- [3Blue1Brown — *How might LLMs store facts? Chapter 7*](https://www.youtube.com/watch?v=9-Jl0dxWQs8) (the MLP intuition — surprisingly underrated)

**Read (90 min):**
- [Jay Alammar — *The Illustrated GPT-2*](https://jalammar.github.io/illustrated-gpt2/) — focuses on decoder-only and masked self-attention
- [The Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/2018/04/03/attention.html) — the original paper as runnable PyTorch. Skim, don't read every line; you'll come back to it.

**Reflect:**
- Why is causal (masked) attention strictly weaker than bidirectional attention, but better for generation?
- What's the difference between `nn.Linear` and a "projection" in attention? (Hint: nothing.)

---

### Day 3 — Karpathy, Part 1

The single most important video you will watch in this roadmap.

**Watch (Part 1, ~60 min):**
- [Andrej Karpathy — *Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) — watch the first 60 minutes today (up to and including the self-attention section)

**Code along.** Pause, type the code, run it. Do not copy from the repo.

**Reference code:**
- [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture) — the companion repo

---

### Day 4 — Karpathy, Part 2

**Watch (~60 min):**
- Finish [*Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY) (multi-head attention, MLP, residual streams, LayerNorm, dropout, scaling up)

**By end of Day 4 you should have:**
- A working ~200-line `gpt.py` file
- It trains on the tiny Shakespeare dataset
- It samples (probably nonsense, but recognizable as Shakespeare-shaped nonsense)
- Loss curve that goes down

---

### Day 5 — Read the production version

Karpathy's tiny GPT is the educational version. `nanoGPT` is the slightly-more-real version. Read it.

**Read (60 min):**
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — read [`model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py) line by line. It's ~300 lines. Annotate it with comments in your own words.

**Then watch (~2h, can split across Day 5 and 6):**
- [Andrej Karpathy — *Let's reproduce GPT-2 (124M)*](https://www.youtube.com/watch?v=l8pRSuU81PU) — the "real" tutorial, takes you from nano-GPT to actually training GPT-2 124M on FineWeb. Watch the first hour today.

**Companion code:**
- [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt)

---

### Day 6 — Modern transformer variants (RoPE, GQA, RMSNorm, SwiGLU)

The original "Attention Is All You Need" transformer is 8 years old. Modern LLMs have replaced some of its components. Today is a quick tour.

**Read (60 min):**
- [Eleuther — *Rotary Embeddings: A Relative Revolution*](https://blog.eleuther.ai/rotary-embeddings/) — RoPE is what Llama, Qwen, DeepSeek all use
- [Sebastian Raschka — *Understanding the Llama 3 Architecture*](https://magazine.sebastianraschka.com/p/understanding-the-llama-architecture) — covers RoPE, GQA, RMSNorm, SwiGLU concretely
- [Lilian Weng — *The Transformer Family Version 2.0*](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — broad survey, skim

**Optional skim (paper time, only if you're hungry):**
- [Vaswani et al. — *Attention Is All You Need*](https://arxiv.org/abs/1706.03762) — the original; read sections 3 and 5

---

### Day 7 — Project + write-up

Today you finalize and ship the weekly project (next section), plus write a short retro.

---

## Weekly Project — `mini-gpt`

Train a small character-level transformer **on your own corpus** and generate samples.

### Spec

Pick a corpus that is interesting to **you**:
- Your own writing / blog / notes
- A subreddit you scraped
- Your own tweets / DMs export
- A favorite author's complete works (out of copyright: Tolstoy, Austen, Shakespeare)
- All your git commit messages from a past project
- A specific subset like all `print()` statements from a big Python repo

### Requirements

- Tokenize the corpus (you can reuse the BPE you wrote in Week 1, or use a character-level tokenizer for simplicity)
- Implement (or adapt) a tiny decoder-only transformer:
  - 4–6 layers
  - 4–6 heads
  - `d_model` between 128 and 384
  - Context length 64–256 tokens
- Train for at least 5,000 steps on a single GPU (Colab T4 is enough)
- Save the loss curve (matplotlib chart in the repo)
- Generate 5 samples of ≥ 200 tokens each, save to `samples.md`
- README explaining: dataset, hyperparameters, total parameters, training time, sample quality

### Stretch (pick one or more)

- Add a **second** corpus (very different from the first), retrain, and compare samples
- Replace your absolute positional embedding with **RoPE** and confirm samples are at least as good
- Replace LayerNorm with **RMSNorm**
- Replace ReLU/GELU MLP with **SwiGLU**
- Plot per-head attention patterns for one sample and write a paragraph about what you see

### Deliverable

```
mini-gpt/
  data/         # your corpus
  model.py      # the transformer (your annotated version of nanoGPT model.py)
  train.py
  sample.py
  samples.md
  loss_curve.png
  README.md
```

This repo goes on your GitHub. Pin it.

---

## Curated resources (your reference shelf)

**Videos (in priority order)**
- [Andrej Karpathy — *Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) — **mandatory**
- [Andrej Karpathy — *Let's reproduce GPT-2 (124M)*](https://www.youtube.com/watch?v=l8pRSuU81PU) — the bigger, more "real" version
- [3Blue1Brown — Deep Learning chapters 5 & 6 (GPT + Attention)](https://www.3blue1brown.com/topics/neural-networks)
- [Stanford CS25 — *Introduction to Transformers with Andrej Karpathy*](https://www.youtube.com/watch?v=XfpMkf4rD6E) (optional, slightly more advanced lecture)

**Articles (in priority order)**
- [Jay Alammar — *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)
- [Jay Alammar — *The Illustrated GPT-2*](https://jalammar.github.io/illustrated-gpt2/)
- [Sebastian Raschka — *Understanding the Llama 3 Architecture*](https://magazine.sebastianraschka.com/p/understanding-the-llama-architecture)
- [The Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/2018/04/03/attention.html) — the paper as code
- [Lilian Weng — *The Transformer Family Version 2.0*](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
- [Eleuther — *Rotary Embeddings: A Relative Revolution*](https://blog.eleuther.ai/rotary-embeddings/)

**Code**
- [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture) — companion for *Let's build GPT*
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — the slightly-more-real version
- [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) — companion for *Let's reproduce GPT-2*

**Papers (light skim only)**
- [Vaswani et al. — *Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
- [Radford et al. — GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

## "Done when…" checklist

- [ ] I can draw a transformer decoder block from memory (residual stream, MHA, MLP, two norms)
- [ ] I can answer: *"Why √d_k?"* and *"What does the causal mask actually look like?"*
- [ ] I have read `nanoGPT/model.py` line by line and annotated it
- [ ] My `mini-gpt` repo trains, loss goes down, and the samples sound vaguely like the corpus
- [ ] I can explain Q, K, V to a non-ML friend in under 3 minutes
- [ ] I know the difference between encoder-only, decoder-only, and encoder-decoder
- [ ] I know what RoPE, GQA, RMSNorm, and SwiGLU are (1-sentence definitions)
- [ ] I've written a retro

---

## Common pitfalls this week

1. **Watching videos passively.** Karpathy's video looks easy when you watch it. It's hard when you type it. Do not skip the typing.
2. **Trying to train a 7B model.** You can't. Stay in the 100k–10M parameter range this week. Bigger comes later, when you fine-tune (Week 8).
3. **Getting stuck on the math.** You do not need to derive backprop through attention by hand. You need to know what each tensor *shape* is at each step.
4. **Reading the original paper first.** It's not a tutorial. Read it after you've watched Karpathy and Alammar.

---

← Previous: [Week 1 — Tokenization & Embeddings](./WEEK-01.md) · → Next: [Week 3 — Modern LLM internals + Hugging Face hands-on](./WEEK-03.md)
