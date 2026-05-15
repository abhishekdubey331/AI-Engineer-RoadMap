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
5. Read a SWE-bench-Lite trajectory and explain what makes it hard
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
- [Inspect AI — *Sandboxing*](https://inspect.ai-safety-institute.org.uk/agents/sandboxing.html) — modern, eval-framework-grade

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
  - HumanEval (Python, function-writing)
  - MBPP (Python, easier function-writing)
  - At least 10 hand-written tasks from *your* Week-7 task domain (e.g., design-to-code components, SQL queries, etc.) — with tests
- Plug in 3 models behind a common interface (could be OpenAI client API → vLLM server / OpenAI / Anthropic / Ollama)
- Run the matrix: 3 models × 3 datasets

---

### Day 6 — SWE-bench: the next-level eval

HumanEval tests function-writing. **SWE-bench** tests "given a real GitHub issue + the repo, generate a patch that resolves it." This is the eval that separates models from agents. Pass rates on full SWE-bench are still in the 30–60% range as of 2026, even for the strongest agents.

**Read (60 min):**
- [SWE-bench — *official site*](https://www.swebench.com/)
- [Runloop — *Understanding LLM Code Benchmarks: From HumanEval to SWE-bench*](https://runloop.ai/blog/understanding-llm-code-benchmarks-from-humaneval-to-swe-bench)
- [Adnan Masood — *Code Generation and Repository-Level Software Engineering Benchmarks*](https://medium.com/@adnanmasood/code-generation-repository-level-software-engineering-benchmarks-a-field-guide-to-llm-benchmarks-330bc3015d80) (field guide)

**Hands-on (60 min):**
- Don't try to run full SWE-bench (it's huge). Instead, run **SWE-bench-Lite** (300 instances) on a couple of cheap problems. Inspect one trajectory in detail. Notice everything that has to go right: identify the right file, edit the right region, not break other tests, produce a patch that applies cleanly.
- You will return to this in Week 14 when you build an actual agent.

---

### Day 7 — LLM-as-judge & the project write-up

When you don't have tests (e.g., "is this code readable?"), an LLM judge is the next best thing. Done badly, judges are biased and noisy. Done well, they're shockingly useful.

**Read (45 min):**
- [Eugene Yan — *LLM-as-judge for evaluation*](https://eugeneyan.com/writing/llm-evaluators/) — the best practical guide on the internet
- [Hugging Face — *Evaluation cookbook*](https://huggingface.co/blog/llm-as-a-judge)
- [Anthropic — *Evaluating outputs*](https://docs.claude.com/en/docs/test-and-evaluate/develop-tests)

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

$ code-eval run --model my-finetuned-v2 --tasks tasks/design-to-code.jsonl --n-samples 5
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
- Datasets: HumanEval **and** a custom dataset from your Week-7 task
- A `report.md` generator that produces a model × dataset table
- LLM-as-judge mode for tasks without runnable tests
- Robust failure handling: timeouts, syntax errors, import errors all bucketed in the report

### Stretch

- Add MBPP and BigCodeBench
- Add SWE-bench-Lite (just a handful of instances; full runs are expensive)
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
- [openai/human-eval](https://github.com/openai/human-eval)
- [google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp)
- [BigCodeBench](https://github.com/bigcode-project/bigcodebench)
- [LiveCodeBench](https://livecodebench.github.io/)
- [SWE-bench](https://www.swebench.com/) — and [SWE-bench-Lite](https://www.swebench.com/lite.html)
- [TerminalBench](https://www.tbench.ai/) — agentic-shell benchmark, increasingly cited
- [BIG-bench, MMLU, etc.](https://github.com/google/BIG-bench) — for general LLMs, not code

**Eval frameworks**
- [Inspect AI](https://inspect.ai-safety-institute.org.uk/) — modern, sandboxed, agent-aware
- [EleutherAI — *lm-evaluation-harness*](https://github.com/EleutherAI/lm-evaluation-harness)
- [DeepEval](https://docs.confident-ai.com/)
- [promptfoo](https://www.promptfoo.dev/) — fast eval for prompts, lightweight

**LLM-as-judge**
- [Eugene Yan — *LLM Evaluators*](https://eugeneyan.com/writing/llm-evaluators/)
- [HF blog — *LLM as a Judge*](https://huggingface.co/blog/llm-as-a-judge)
- [Anthropic — *Evaluating outputs*](https://docs.claude.com/en/docs/test-and-evaluate/develop-tests)

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
- [ ] I have read one SWE-bench-Lite trajectory in detail and can explain what made it hard
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
