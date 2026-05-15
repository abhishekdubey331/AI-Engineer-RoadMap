# Week 1 — Tokenization & Embeddings

> **Month 1 · Foundations of LLMs**
> *"Before you can train, fine-tune, or prompt an LLM, you have to understand how it sees text."*

---

## Why this week

Tokenization is the single most underrated topic in LLM engineering. It silently determines:

- **Cost** — you're billed by the token
- **Latency** — prompt + decode time scales with token count
- **Quality** — bad tokenization breaks math, code, and non-English text
- **Context windows** — "128k tokens" is not "128k characters"
- **Fine-tuning** — your dataset gets tokenized before training; a mismatched tokenizer silently corrupts everything

Everyone wants to skip straight to transformers. The engineers who skip tokenization spend the next year debugging things that turn out to be tokenizer issues. We're not going to be those engineers.

By the end of this week you'll have **written BPE from scratch**, and you'll have a small CLI tool you'll use for the rest of the roadmap to estimate token counts and costs.

---

## Learning objectives

By Sunday night you should be able to:

1. Explain the difference between character-, word-, and subword-tokenization, and why subword (BPE) won
2. Implement Byte-Pair Encoding (BPE) from scratch in ~200 lines of Python
3. Explain how GPT-2, GPT-4, Llama, and Claude tokenizers differ
4. Use `tiktoken` and Hugging Face `tokenizers` to count tokens, estimate cost, and inspect token IDs
5. Explain what an **embedding** is, and run a sentence-similarity demo with `sentence-transformers`
6. Diagnose a "weird LLM output" caused by tokenization (the infamous `SolidGoldMagikarp` story)

---

## Day-by-day plan (≈ 2–3 hrs/day)

### Day 1 — What is a token, really?

**Watch (90 min):**
- [Andrej Karpathy — *Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE) (2h13m, watch first 60–90 min today)

**Read (30 min):**
- [Hugging Face NLP Course — Chapter 6: The 🤗 Tokenizers library — *Introduction* + *Training a new tokenizer*](https://huggingface.co/learn/llm-course/chapter6/1)

**Reflect:**
- Open [OpenAI's Tokenizer playground](https://platform.openai.com/tokenizer) and paste:
  - A simple English sentence
  - A Python snippet with indentation
  - A long number like `1234567890123`
  - A non-English sentence (Hindi, Mandarin, Arabic, etc.)
  - The word `SolidGoldMagikarp`
- Count the tokens for each. Notice the asymmetry.

---

### Day 2 — Finish Karpathy + understand BPE

**Watch (60–90 min):**
- Finish Karpathy's tokenizer video

**Read (45 min):**
- [Sebastian Raschka — *Implementing a Byte Pair Encoding (BPE) Tokenizer From Scratch*](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)
- [Hugging Face NLP Course — *Byte-Pair Encoding tokenization*](https://huggingface.co/learn/llm-course/chapter6/5)

**Hands-on (30 min):**
- Clone Karpathy's repo: `git clone https://github.com/karpathy/minbpe`
- Run the included `train.py` on a tiny text file (e.g., paste a chapter of *Alice in Wonderland*)
- Inspect the vocab that comes out

---

### Day 3 — Modern tokenizers in practice

**Read (60 min):**
- [Karpathy `minbpe` README + `lecture.md`](https://github.com/karpathy/minbpe) — the lecture.md is a written companion to the video, very dense
- [Answer.AI — *How I created the Karpathy Tokenizers book chapter*](https://www.answer.ai/posts/2025-10-13-video-to-doc.html) (skim — context for the next reading)
- [Fast.ai — *Let's Build the GPT Tokenizer: A Complete Guide to Tokenization in LLMs*](https://www.fast.ai/posts/2025-10-16-karpathy-tokenizers) (the book-chapter version of the video — the best written reference on this topic)

**Hands-on (45 min):**
- `pip install tiktoken sentence-transformers transformers`
- In a notebook, compare GPT-2, GPT-4o, Llama-3, and (if you have an Anthropic key) Claude tokenizers on the same input:
  ```python
  import tiktoken
  from transformers import AutoTokenizer

  text = "def fibonacci(n: int) -> int: ..."
  for enc in ["gpt2", "cl100k_base", "o200k_base"]:
      print(enc, len(tiktoken.get_encoding(enc).encode(text)))

  for hf in ["meta-llama/Meta-Llama-3-8B", "Qwen/Qwen2.5-Coder-7B"]:
      tok = AutoTokenizer.from_pretrained(hf)
      print(hf, len(tok.encode(text)))
  ```
- Note the differences — they matter for cost and for choosing which model is "cheaper" per task.

---

### Day 4 — Build BPE from scratch (Part 1)

Today you start writing your own BPE. You're going to follow Karpathy's `minbpe` structure but type every line yourself — no copy-paste. Resist the urge to just run his code.

**Goal:** Implement the **byte-level** tokenizer:
- `train(text, vocab_size)` — learns merges
- `encode(text)` — applies them
- `decode(ids)` — reverses

**Pseudo-structure:**
```python
class BasicTokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int): ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
```

**References (use only when stuck):**
- [`karpathy/minbpe` — `base.py` and `basic.py`](https://github.com/karpathy/minbpe/tree/master/minbpe)
- [`minbpe/exercise.md`](https://github.com/karpathy/minbpe/blob/master/exercise.md) — Karpathy's own exercise track

---

### Day 5 — Build BPE from scratch (Part 2)

Today you upgrade to a **regex-based** tokenizer (the GPT-style variant) which pre-splits by whitespace/punctuation patterns before merging. This is what `tiktoken` actually does internally.

**Goal:** Add a `RegexTokenizer` that uses GPT-4's pattern:
```python
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|"""
    r"""\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
```

**Stretch goal:** Match `tiktoken`'s `cl100k_base` encoding for at least one sample sentence.

---

### Day 6 — From tokens to embeddings

Now that you understand tokens, the next layer up: **what does the model do with them?**

**Read (45 min):**
- [Hugging Face NLP Course — Chapter 1: *How Transformers work?*](https://huggingface.co/learn/llm-course/chapter1/4) (the embeddings + transformer overview)
- [Cohere — *What are embeddings?*](https://docs.cohere.com/docs/embeddings) (very short, very clear)

**Watch (15 min):**
- [3Blue1Brown — *Visualizing Attention, a Transformer's Heart* (first 5 min on embeddings)](https://www.youtube.com/watch?v=eMlx5fFNoYc)

**Hands-on (60 min):**
- `pip install sentence-transformers`
- Run this and explore:
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer("BAAI/bge-small-en-v1.5")

  sents = [
      "How do I sort a list in Python?",
      "What's the syntax for sorting an array in Python?",
      "How to bake sourdough bread",
      "def quicksort(arr): ...",
  ]
  emb = model.encode(sents)
  # Compute cosine similarity between all pairs and print the matrix
  ```
- See how the first two are very close, and code is also closer to them than the bread question.

This is the foundation of RAG, which we hit in Week 5.

---

### Day 7 — Project, write-up, and the "weird LLM output" debug

Today is build + polish + a small detective story.

**Read (30 min) — *the cautionary tale*:**
- [LessWrong — *SolidGoldMagikarp (plus, prompt generation)*](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) — the canonical "tokenization broke the model" story. Skim it.
- Bonus skim: [Riley Goodside on tokenization quirks](https://x.com/goodside) — search his posts for tokenizer-related threads

**Build & finalize the weekly project (next section).**

---

## Weekly Project — `token-budget`

A small but useful CLI you will actually reuse for the next 15 weeks.

### Spec

```
$ token-budget --model gpt-4o ./prompts/system.md ./prompts/user.md
file                tokens     cost ($/1M in × tokens)
prompts/system.md   1,243      $0.0062
prompts/user.md     412        $0.0021
TOTAL               1,655      $0.0083

$ token-budget --model llama-3.1-8b ./repo/
src/main.py         523
src/utils.py        212
docs/README.md      891
...
TOTAL              4,237 tokens, fits in 8k context ✓
```

### Requirements

- A `bpe.py` containing your own `BasicTokenizer` and `RegexTokenizer` (from Days 4–5)
- A CLI (`argparse` or `click`) that accepts:
  - One or more file paths, OR a directory (recursively scans `.py`, `.md`, `.txt`, `.json`)
  - `--model` flag with at least 3 options: `gpt-4o`, `llama-3.1-8b`, `your-own-bpe`
- For `gpt-4o` use `tiktoken`'s `o200k_base`
- For `llama-3.1-8b` use `AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")` (or a permissively-licensed equivalent if you can't access the Llama gate)
- For `your-own-bpe` use your own trained tokenizer
- Output: per-file token count, total, cost estimate (use today's published $/1M token prices, hardcoded with a comment)
- 5+ `pytest` tests
- A `README.md` explaining what it does, how to install, and **one paragraph** on what surprised you about tokenization

### Stretch (optional)

- Add a `--diff` mode that compares two tokenizers on the same input and prints the delta
- Add a `--longest-token` mode that finds the rarest tokens in a corpus (this is how you find `SolidGoldMagikarp`-style glitch tokens)

---

## Curated resources (your reference shelf)

**Videos**
- [Andrej Karpathy — *Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE) — **the** canonical resource. 2h13m, every minute earned.
- [3Blue1Brown — *But what is a GPT? — Chapter 5*](https://www.youtube.com/watch?v=wjZofJX0v4M) — context for what tokens flow into
- [3Blue1Brown — *Attention in transformers, step-by-step*](https://www.youtube.com/watch?v=eMlx5fFNoYc) — preview of next week

**Articles**
- [Fast.ai — *Let's Build the GPT Tokenizer: A Complete Guide*](https://www.fast.ai/posts/2025-10-16-karpathy-tokenizers) — written version of Karpathy's video, page-equivalent to the entire lecture
- [Sebastian Raschka — *BPE from scratch*](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)
- [Hugging Face NLP Course — Chapter 6](https://huggingface.co/learn/llm-course/chapter6/1) — official, comprehensive
- [Cohere — *What are embeddings?*](https://docs.cohere.com/docs/embeddings)

**Code**
- [karpathy/minbpe](https://github.com/karpathy/minbpe) — read every line
- [karpathy/rustbpe](https://github.com/karpathy/rustbpe) — the same in Rust, optional
- [openai/tiktoken](https://github.com/openai/tiktoken) — production tokenizer; check `tiktoken/_educational.py`
- [huggingface/tokenizers](https://github.com/huggingface/tokenizers) — production reference

**Papers (light skim only, do not get stuck here)**
- [Sennrich et al. (2016) — *Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909) — the original BPE paper for NMT
- [Radford et al. (2019) — GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — see the tokenization section

**Cautionary tale**
- [LessWrong — *SolidGoldMagikarp*](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)

---

## "Done when…" checklist

Before moving to Week 2, you should be able to truthfully tick all of these:

- [ ] I can explain in two sentences why BPE beats word- and character-level tokenization
- [ ] My `bpe.py` trains, encodes, and decodes correctly on a small corpus, with `encode(decode(x)) == x` round-tripping on UTF-8 inputs including emoji
- [ ] I have a working `token-budget` CLI on GitHub with tests and a README
- [ ] I can predict (within ±20%) how many tokens a given code snippet will be in GPT-4o
- [ ] I know what a glitch token is and why they exist
- [ ] I can answer: *"Why do LLMs struggle with arithmetic on long numbers?"* with a tokenization-flavored answer
- [ ] I've written a 1-paragraph retro: what surprised me, what was hard, what I'd do differently

---

## Common pitfalls this week

1. **Spending 3 days reading and 0 days coding.** The video is great. The code is what makes it stick. If by Day 4 you haven't typed `class BasicTokenizer:`, you are off-track.
2. **Trying to beat `tiktoken`.** You won't. The goal is *understanding*, not benchmark-chasing.
3. **Skipping the regex tokenizer.** The plain BPE is fine, but every real-world LLM uses regex pre-splitting. You need to see why.
4. **Forgetting UTF-8.** A character is not a byte. BPE operates on bytes. This trips up everyone the first time.

---

→ Next: [Week 2 — The Transformer (Attention, Decoder-only)](./WEEK-02.md)
