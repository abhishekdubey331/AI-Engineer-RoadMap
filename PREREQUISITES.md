# Prerequisites

> **What you should be comfortable with before Week 1.**

The 4-month roadmap assumes you already have basic software-engineering skills. Below is a short audit. If a section says **"comfortable"**, you can move on. If it says **"shaky"** or **"never seen it"**, spend a few days fixing that before Week 1 — otherwise the LLM-specific weeks will feel like quicksand.

You do **not** need to be an expert. You need to be unblocked.

---

## 1. Python (intermediate)

You should be comfortable with:

- Functions, classes, type hints (`def f(x: int) -> str:`)
- Exceptions, `try/except/finally`, custom exception classes
- Context managers (`with open(...) as f:`)
- File handling, JSON, CSV
- List/dict comprehensions, generators
- `argparse` or `click` for CLIs
- `pathlib`, `os`, `subprocess` basics
- Virtual environments (`venv` / `uv`) and `pip` / `pyproject.toml`
- `pytest` (writing and running tests)

**Best refresher:**
- [The official Python tutorial](https://docs.python.org/3/tutorial/) — skim the parts you're rusty on
- [Real Python — Python type checking guide](https://realpython.com/python-type-checking/)
- [uv](https://docs.astral.sh/uv/) for fast venv + package management (this is what modern Python projects use)

---

## 2. Git & GitHub

You should be comfortable with:

- `clone`, `add`, `commit`, `push`, `pull`, `fetch`
- Branches, merging, rebasing (basic)
- Pull requests on GitHub
- Resolving merge conflicts
- `.gitignore` for not committing model weights or `.env` files

**Best refresher:**
- [Pro Git book, chapters 1–3](https://git-scm.com/book/en/v2) — free, the canonical reference
- [Oh My Git!](https://ohmygit.org/) — visual game for learning git

---

## 3. Linux shell

You should be comfortable with:

- Navigating (`cd`, `ls`, `pwd`, `find`)
- Piping (`|`, `>`, `>>`, `<`)
- `grep`, `sed`, `awk` at a basic level
- Environment variables, `.env` files
- `ssh`, `scp` for remote machines (you'll rent GPUs at some point)
- Process management (`ps`, `kill`, `htop`)

**Best refresher:**
- [The Missing Semester of Your CS Education (MIT)](https://missing.csail.mit.edu/) — Lectures 1–4 are gold

---

## 4. APIs & FastAPI

You should understand:

- HTTP verbs, status codes, JSON bodies
- REST endpoints (`GET`, `POST`)
- How to write a minimal FastAPI app with one or two endpoints
- Server-Sent Events (SSE) at a conceptual level (you'll use these for streaming LLM tokens)
- Async Python basics (`async def`, `await`)

**Best refresher:**
- [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/) — work through the first half
- [MDN — HTTP basics](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview)

---

## 5. Docker (basics)

You should be able to:

- Write a `Dockerfile` for a Python app
- Build an image, run a container
- Understand `docker-compose.yml` at a glance
- Mount volumes, expose ports

**Best refresher:**
- [Docker — Get started guide](https://docs.docker.com/get-started/) (official, ~2 hours)

---

## 6. ML & PyTorch (very light)

## 6. Math foundations (light but non-negotiable)

You do **not** need a math degree. You do need a working intuition for the operations LLMs are made of, because by Week 2 you'll be staring at attention (matmul + softmax) and by Week 5 you'll be tuning cosine-similarity thresholds. If matmul feels magical, the rest of the roadmap is brittle.

You should be comfortable with:

- **Vectors** — what they are, addition, scalar multiplication, magnitude/norm
- **Dot product** and **cosine similarity** (you'll use cosine constantly in RAG)
- **Matrices** and **matrix multiplication** (this is the engine of every transformer)
- **Softmax** — turns logits into a probability distribution
- **Loss functions** — especially **cross-entropy** (next-token prediction's bread and butter)
- **Gradient descent** intuition — partial derivatives, the chain rule (one-line version), `loss.backward()` updates weights
- **Probability basics** — joint, conditional, log-probabilities

**Best refresher (pick one path):**

- 🎥 **Visual path** — [3Blue1Brown — *Essence of Linear Algebra*](https://www.3blue1brown.com/topics/linear-algebra) (12 videos, ~3 hrs). The single best intro to vectors / matmul / eigenstuff in any medium. Pair with [3Blue1Brown — *Neural Networks*](https://www.3blue1brown.com/topics/neural-networks) chapters 1–4 for the loss-and-gradient picture.
- 📚 **Practice-first path** — [Khan Academy — *Linear Algebra*](https://www.khanacademy.org/math/linear-algebra) (free, exercises included). Use as a workbook alongside 3Blue1Brown.
- 🐍 **Code-first path** — [Pablo Caceres — *Introduction to Linear Algebra for Applied Machine Learning with Python*](https://pabloinsente.github.io/intro-linear-algebra) — every concept paired with NumPy code; closest to how you'll actually use the math.
- 📖 **Reference book (free)** — [Boyd & Vandenberghe — *Introduction to Applied Linear Algebra*](https://web.stanford.edu/~boyd/vmls/) — the most beginner-friendly applied linear algebra textbook; PDF free online.

**Bare-minimum signal** that you're ready: open a Python REPL and write, without Googling, (a) cosine similarity between two NumPy vectors, and (b) the softmax of a logit vector. If both come out in under 5 minutes, you're set.

---

## 7. PyTorch fundamentals (Week 2 assumes these)

Earlier drafts of this roadmap said "you've used `.backward()` once" — that's not enough. Week 2 jumps into building a tiny GPT, which assumes you're fluent with **`nn.Module`**, **forward passes**, **training loops**, and **GPU device placement**. The fix: do *one* of the resources below *before* Week 2.

You should be able to:

- Move tensors between CPU and GPU (`tensor.to("cuda")`)
- Subclass `nn.Module` with `__init__` and `forward`
- Run an end-to-end training loop: forward → loss → `loss.backward()` → `optimizer.step()` → `optimizer.zero_grad()`
- Use a `DataLoader` over a `Dataset`
- Save and load model weights (`state_dict`)
- Run inference in `torch.no_grad()` / `model.eval()` mode

**Best refresher (pick one, do it end-to-end):**

- 🚀 **Fastest path (~1 hr)** — [Sebastian Raschka — *PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs*](https://sebastianraschka.com/teaching/pytorch-1h/) — dense, recent, exactly the right level.
- 📘 **Official path (~2 hrs)** — [PyTorch — *Learn the Basics*](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) — the canonical 8-section tutorial; ends with the FashionMNIST training loop you'll mirror in Week 2.
- 🎬 **Deepest path (~10 hrs)** — [Andrej Karpathy — *Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html), parts 1–4 (micrograd → makemore → wavenet). If you do this *one* thing, you'll fly through Week 2.
- 🧑‍🏫 **Course path** — [Daniel Bourke — *Zero to Mastery: Learn PyTorch for Deep Learning*](https://www.learnpytorch.io/) — free, comprehensive, code-along.

**Bare-minimum signal** that you're ready: write a 50-line PyTorch script that trains a 2-layer MLP on the MNIST or FashionMNIST dataset, saves the weights, reloads them, and runs inference. If you can do that on a fresh notebook in under 30 minutes from scratch, you're set.

---

## 8. Classical ML refresher (light)

You should have at least once:

- Trained a simple model (logistic regression, MLP, or small CNN)
- Understood loss, gradient descent, train/val/test split, overfitting
- Used PyTorch tensors and `.backward()` once

You do **not** need to know transformers yet — that's literally Week 2.

**Best refresher (pick one):**
- [Andrej Karpathy — Neural Networks: Zero to Hero, Part 1 (micrograd)](https://www.youtube.com/watch?v=VMj-3S1tku0) — the single best intro to backprop in existence
- [PyTorch — 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [fast.ai — Practical Deep Learning, Lessons 1–2](https://course.fast.ai/)

---

## 9. Data handling

You should be able to:

- Use `pandas` for loading, filtering, grouping CSV/JSONL
- Read & write JSONL line-by-line
- Deduplicate a dataset
- Do a basic train/val/test split

**Best refresher:**
- [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Hugging Face Datasets — quick tour](https://huggingface.co/docs/datasets/quickstart)

---

## A 1-week self-test project (optional)

If you're not sure whether you're ready, build this in a weekend. If you can finish it, you're ready for Week 1.

> **CLI Code Analyzer.** Build a Python CLI that takes a folder path, scans `.py` files, extracts function/class names + line counts, finds TODO comments, and outputs JSON. Wrap it in Docker. Write 5 pytest tests. Push it to GitHub.

This was the original Week 1 project of the 6-month version of this roadmap — if you can build it in 2–3 days, the prerequisites are covered.

---

## Hardware & accounts to set up before Week 1

- A GitHub account
- A Hugging Face account ([huggingface.co](https://huggingface.co)) — you'll need an access token
- A Google Colab account (free tier is enough for early weeks)
- (Optional) An OpenAI **or** Anthropic API key with $5–$20 of credit for prompting weeks
- (Optional) A Runpod / Lambda / Vast.ai account for renting GPUs in fine-tuning weeks

Don't preinstall everything. Each weekly file lists exactly what to install that week.

---

You're ready when you can read this list and nod along to ~80% of it. Don't aim for 100% — you'll learn the rest along the way. Open [`weeks/WEEK-01.md`](./weeks/WEEK-01.md) and start.
