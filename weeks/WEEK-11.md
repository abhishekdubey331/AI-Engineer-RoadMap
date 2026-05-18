# Week 11 — Code Evaluation (HumanEval, SWE-bench, pass@k)

> **Month 3 · Production LLM Systems**
> *"Vibes will get you a demo. Evals get you a production system. This is the week you stop guessing."*

---

## Why this week

Every model change you make from now on — quantize, fine-tune, swap a base — needs an eval. Without one, you're shipping vibes.

For code models specifically, the eval that matters is **functional correctness**: does the generated code actually pass tests? Not BLEU. Not ROUGE. Not "looks right." Tests.

This week you build a **sandboxed code-evaluation harness** you can plug *any* model into and get a real number out. It becomes the backbone of every later experiment.

---

## Learning objectives

By Sunday night you should be able to:

1. Explain functional correctness vs textual similarity for code
2. Compute `pass@k` correctly (the hypergeometric estimator, not the naive one)
3. Stand up a sandboxed execution environment (Docker + resource limits + timeout) that safely runs generated code
4. Run HumanEval, MBPP, and at least one harder benchmark (BigCodeBench or LiveCodeBench) against your model
5. Read a SWE-bench-Verified trajectory and explain what makes it hard
6. Build a custom mini-eval for *your* task (the Week-7 dataset's test split, scored properly)
7. Use **LLM-as-judge** correctly: when it works, when it lies

---

## Day-by-day plan (≈ 2–3 hrs/day)

### Day 1 — The metric: pass@k, properly

**Read (75 min):**
- [Michael Brenndoerfer — *HumanEval: Functional Code Generation Evaluation with Pass@k*](https://mbrenndoerfer.com/writing/humaneval-code-generation-benchmark-pass-at-k) — clearest pass@k explainer
- [Michael Brenndoerfer — *Code Evaluation: Functional Correctness and pass@k*](https://mbrenndoerfer.com/writing/code-evaluation-functional-correctness-pass-at-k-benchmarks)
- [DataCamp — *HumanEval Benchmark*](https://www.datacamp.com/tutorial/humaneval-benchmark-for-evaluating-llm-code-generation-capabilities)

**The pass@k formula:**

For `n` total samples per problem, of which `c` pass:

```
pass@k = 1 − C(n−c, k) / C(n, k)
```

(probability that at least one of `k` randomly drawn samples passes, hypergeometric estimator).

Naive `pass@k` = `c/n` is biased. The hypergeometric version is what HumanEval uses. Burn this into memory.

---

### Day 2 — Read the HumanEval & MBPP code

You're going to build *like* HumanEval. Read it first.

**Read (60 min):**
- [openai/human-eval](https://github.com/openai/human-eval) — read `execution.py` line by line; this is how they sandbox
- [google/mbpp](https://github.com/google-research/google-research/tree/master/mbpp)
- [bigcode/bigcodebench](https://github.com/bigcode-project/bigcodebench) — newer, much harder

**Hands-on (60 min):**
- Run HumanEval on a small open-weight code model:
  ```bash
  pip install human-eval
  # Generate completions from your model into samples.jsonl
  evaluate_functional_correctness samples.jsonl
  ```
- Open the output. Some completions will fail in obvious ways. Take notes.

---

### Day 3 — Sandboxing (the safety part)

Generated code can do *anything*. You must run it in isolation.

**Read (60 min):**
- [HumanEval — *execution.py*](https://github.com/openai/human-eval/blob/master/human_eval/execution.py) — the resource-limited subprocess pattern
- [Docker docs — *Resource constraints*](https://docs.docker.com/config/containers/resource_constraints/)
- [BigCodeBench — *evaluate*](https://github.com/bigcode-project/bigcodebench/blob/main/bigcodebench/evaluate.py)
- [Inspect AI — *Sandboxing*](https://inspect.aisi.org.uk/agents/sandboxing.html) — modern, eval-framework-grade

**Sandbox checklist:**
- Run in a subprocess (never `exec()` in your test process)
- Strict timeout (e.g., 10s)
- No network (Docker `--network=none`)
- Resource limits (memory, CPU, file descriptors)
- Ephemeral filesystem (tmpfs / Docker volume that's destroyed after run)
- Restrict to a Python image with only the libraries the test needs

---

### Day 4 — Build the harness, Part 1

**Hands-on (~2.5 hr):**
- Create `code_eval/` repo
- Define a `Task` schema:
  ```python
  class Task(BaseModel):
      task_id: str
      prompt: str           # what's shown to the model
      entry_point: str      # the function name to test
      canonical_solution: str  # reference (for unit tests)
      test: str             # pytest-style or HumanEval-style test code
  ```
- Implement `generate(model, task, n_samples) -> list[Completion]`
- Implement `execute(completion, task, timeout, sandbox=docker_runner) -> Result(passed: bool, error: str | None)`
- Compute `pass@1`, `pass@10`, `pass@100` from samples
- Run on the first 10 HumanEval tasks to verify your harness matches the official numbers

---

### Day 5 — Build the harness, Part 2 + your dataset

**Hands-on (~2.5 hr):**
- Make the harness accept **multiple datasets**:
  - HumanEval (Python, function-writing) — saturated and contamination-suspect by 2026; include as a baseline only
  - **LiveCodeBench** (continuously refreshed from LeetCode/AtCoder/CodeForces, contamination-free) — required in 2026
  - **BigCodeBench-Hard** (148-task subset) — the actually-discriminating split
  - MBPP (Python, easier function-writing)
  - At least 10 hand-written tasks from *your* Week-7 task domain — with tests
- Plug in 3 models behind a common interface (could be OpenAI client API → vLLM server / OpenAI / Anthropic / Ollama)
- Run the matrix: 3 models × 3 datasets

**Contamination check (required):** for any HumanEval/MBPP number, run a sanity check — count what % of model outputs are token-level near-matches to the canonical solution. If >25%, the number is almost certainly contamination-inflated and you should report LiveCodeBench / BigCodeBench-Hard alongside.

---

### Day 6 — SWE-bench-Verified, Aider Polyglot: the next-level evals

HumanEval tests function-writing. The 2026 benchmarks that actually decide what frontier labs report:

- **SWE-bench-Verified (500 human-validated GitHub issues)** — the canonical "real-world coding" eval. Every Claude / GPT / Gemini coding-agent paper reports against this. Pass rates climbed from 40% → 80%+ between 2024 and 2026.
- **Aider Polyglot (225 Exercism problems, 6 languages)** — tests file-editing in an agentic loop; closer to actual coding than HumanEval.
- **Terminal-Bench 2.0** — agentic-shell tasks.

**Read (60 min):**
- [OpenAI — *Introducing SWE-bench Verified*](https://openai.com/index/introducing-swe-bench-verified/) — the 500-task human-cleaned subset
- [SWE-bench — *official site* + Verified leaderboard](https://www.swebench.com/) + [Verified page](https://www.swebench.com/verified.html)
- [SWE-bench — *Submission guide*](https://www.swebench.com/SWE-bench/guides/submissions/)
- [Aider Polyglot benchmark](https://github.com/Aider-AI/polyglot-benchmark) + [Aider leaderboard](https://aider.chat/docs/leaderboards/)
- [Runloop — *Understanding LLM Code Benchmarks*](https://runloop.ai/blog/understanding-llm-code-benchmarks-from-humaneval-to-swe-bench)

**Hands-on (60 min):**
- Don't try to run full SWE-bench (it's huge and costly). Instead, run **SWE-bench-Verified** on ~10 instances using the official harness; pick easy/medium difficulty. Inspect one trajectory in detail: identify the right file, edit the right region, don't break other tests, produce a patch that applies cleanly.
- You will return to this in Week 16 — your capstone is **a coding agent benchmarked on SWE-bench-Verified-Lite**, so this week's work is directly load-bearing.

---

### Day 7 — LLM-as-judge & the project write-up

When you don't have tests (e.g., "is this code readable?"), an LLM judge is the next best thing. Done badly, judges are biased and noisy. Done well, they're shockingly useful.

**Read (60 min):**
- [Hamel Husain — *LLM Evals FAQ*](https://hamel.dev/blog/posts/evals-faq/) — the single best practitioner reference; mandatory
- [Hamel Husain — *LLM-as-a-Judge*](https://hamel.dev/blog/posts/llm-judge/) — paired companion read
- [Eugene Yan — *LLM-as-judge for evaluation*](https://eugeneyan.com/writing/llm-evaluators/) — the original deep guide
- [Hugging Face — *Evaluation cookbook*](https://huggingface.co/blog/llm-as-a-judge)
- [Anthropic — *Evaluating outputs*](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests)

**Patterns that work**
- Pairwise comparison (A vs B), not absolute scoring
- Multiple judges, majority vote
- Include the rubric and one "anchor" example of each rating
- Always sanity-check the judge against human labels on a sample

**Add to your harness:**
- An `--llm-judge` mode that grades non-runnable outputs (e.g., docstring quality)

---

## Weekly Project — `code-eval-harness`

A reusable code-eval system. By the end of the roadmap, every model decision you make goes through this.

### Spec

```
$ code-eval run --model qwen-coder-1.5b --tasks humaneval --n-samples 10
[######################] 164/164 tasks
pass@1 = 0.41
pass@10 = 0.58

$ code-eval run --model my-finetuned-v2 --tasks tasks/mytask.jsonl --n-samples 5
[##############################] 50/50 tasks
pass@1 = 0.66
pass@5 = 0.84

$ code-eval report --output report.md
# generates a multi-model, multi-dataset summary
```

### Requirements

- A clean `Task` schema (`pydantic` or `attrs`)
- Sandboxed execution (Docker `--network=none`, resource limits, timeout)
- pass@k with the correct hypergeometric estimator
- Backends: at least one local (Ollama or vLLM) + at least one hosted (OpenAI or Anthropic)
- Datasets: HumanEval (baseline) + **at least one of LiveCodeBench or BigCodeBench-Hard** (contamination-resistant; required for any reported HumanEval number to be credible) + your custom Week-7 task dataset
- A `report.md` generator that produces a model × dataset table
- LLM-as-judge mode for tasks without runnable tests, with **two judges from different model families** + a calibration step against ≥10 human-labeled examples
- Robust failure handling: timeouts, syntax errors, import errors all bucketed in the report

### Stretch

- Add MBPP
- Add a SWE-bench-Verified mini run (~10 instances; the full set is expensive) — load-bearing for Week 16
- Add Aider Polyglot
- Cache completions to disk so re-runs are free
- Export to W&B or MLflow for nicer dashboards

---

## Curated resources

**Concepts**
- [Michael Brenndoerfer — *HumanEval pass@k* (interactive)](https://mbrenndoerfer.com/writing/humaneval-code-generation-benchmark-pass-at-k)
- [Michael Brenndoerfer — *Functional Correctness and pass@k*](https://mbrenndoerfer.com/writing/code-evaluation-functional-correctness-pass-at-k-benchmarks)
- [DataCamp — *HumanEval Benchmark*](https://www.datacamp.com/tutorial/humaneval-benchmark-for-evaluating-llm-code-generation-capabilities)
- [DeepEval — *HumanEval reference*](https://deepeval.com/docs/benchmarks-human-eval)
- [Runloop — *HumanEval → SWE-bench*](https://runloop.ai/blog/understanding-llm-code-benchmarks-from-humaneval-to-swe-bench)

**Benchmarks (the catalog)**
- [SWE-bench Verified](https://www.swebench.com/verified.html) + [submission guide](https://www.swebench.com/SWE-bench/guides/submissions/) — the canonical 2026 coding-agent eval
- [Aider Polyglot](https://github.com/Aider-AI/polyglot-benchmark) + [Aider leaderboard](https://aider.chat/docs/leaderboards/)
- [LiveCodeBench](https://livecodebench.github.io/) — contamination-free, continuously refreshed
- [BigCodeBench](https://github.com/bigcode-project/bigcodebench) (use the Hard subset)
- [openai/human-eval](https://github.com/openai/human-eval) — saturated and contamination-suspect; baseline only
- [google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp)
- [TerminalBench](https://www.tbench.ai/) (Terminal-Bench 2.0)
- [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) — the still-discriminating subset (BIG-bench is legacy)

**Practitioner reading**
- [Hamel Husain — *LLM Evals FAQ*](https://hamel.dev/blog/posts/evals-faq/) — required
- [Hamel Husain — *LLM-as-a-Judge*](https://hamel.dev/blog/posts/llm-judge/)
- [Eugene Yan — *LLM Evaluators*](https://eugeneyan.com/writing/llm-evaluators/)

**Eval frameworks**
- [Inspect AI](https://inspect.aisi.org.uk/) — modern, sandboxed, agent-aware
- [EleutherAI — *lm-evaluation-harness*](https://github.com/EleutherAI/lm-evaluation-harness)
- [DeepEval](https://docs.confident-ai.com/)
- [promptfoo](https://www.promptfoo.dev/) — fast eval for prompts, lightweight

**LLM-as-judge**
- [Hamel Husain — *LLM-as-a-Judge*](https://hamel.dev/blog/posts/llm-judge/)
- [Eugene Yan — *LLM Evaluators*](https://eugeneyan.com/writing/llm-evaluators/)
- [HF blog — *LLM as a Judge*](https://huggingface.co/blog/llm-as-a-judge)
- [Anthropic — *Evaluating outputs*](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests)

**Papers (skim)**
- [Chen et al. — *Evaluating Large Language Models Trained on Code* (HumanEval)](https://arxiv.org/abs/2107.03374)
- [Austin et al. — *Program Synthesis with Large Language Models* (MBPP)](https://arxiv.org/abs/2108.07732)
- [Jimenez et al. — *SWE-bench*](https://arxiv.org/abs/2310.06770)

---

## "Done when…" checklist

- [ ] My harness runs HumanEval and gets within ±5% of the published numbers for a known model
- [ ] Generated code runs in a sandbox with timeout + no network
- [ ] pass@k uses the correct hypergeometric estimator
- [ ] I have evaluated my Week-8 fine-tuned model on a custom task dataset and have a number
- [ ] I have read one SWE-bench-Verified trajectory in detail and can explain what made it hard
- [ ] I have an LLM-as-judge mode and have validated it against human grades on a sample
- [ ] I have a `report.md` showing at least 2 models × 2 datasets
- [ ] I've written a retro

---

## Common pitfalls this week

1. **Using naive `c/n` for pass@k.** That's biased. Use the hypergeometric.
2. **No sandbox.** A single bad completion can `rm -rf` your machine. Always sandbox.
3. **Generating once.** Models are stochastic. `n_samples >= 5` for any pass@k > 1.
4. **Confusing "syntax valid" with "passes."** Many evals score "did the code run without exception?" — that's not the same as correct. Always run the tests.
5. **Trusting one judge model.** If GPT-4 is your judge for evaluating GPT-4 outputs, your numbers are inflated. Use a different family.
6. **Comparing across non-identical prompts.** If the prompt template differs between two models, you can't fairly compare. Standardize.

---

← Previous: [Week 10 — Inference Servers](./WEEK-10.md) · → Next: [Week 12 — Caching, Routing, Production API Hardening](./WEEK-12.md)
