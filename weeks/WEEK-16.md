# Week 16 — Capstone: Autonomous Coding Agent on SWE-bench-Verified

> **Month 4 · Agents, Evaluation, Capstone**
> *"This is the week the 15 pieces become one product, scored against a public benchmark a hiring manager already knows. Everything you've built is on trial."*

---

## Why this capstone, in May 2026

The original 6-month version of this roadmap had a *Design-to-Code Agent* capstone. In 2024 that was a credible portfolio piece. By May 2026, "given a prompt, output a React component" has been commoditised by v0, Bolt, Lovable, Magic Patterns, Figma Make, Stitch, and Vercel AI SDK's generative UI. A 4-month-junior submitting a worse v0 to an interview will get a *worse-v0* impression in return.

The 2026 capstone has to put a number on the page that recruiters and senior engineers immediately understand:

> *"My agent solves N / 50 SWE-bench-Verified-Lite instances at $X/instance with p95 latency Y."*

SWE-bench-Verified is the eval **every frontier lab reports on** (Claude, GPT, Gemini, Qwen-Coder, DeepSeek). Frontier-model agents are now in the 80–90%+ range; a junior building from scratch and clearing 20–35% with honest evals is genuinely impressive, because the bar is "you ran the harness end-to-end and you can talk about the failure modes." That's the rare skill.

> **Want a different domain?** Skip to **"Alternative capstones"** at the bottom of the file. TAU2-bench (customer-service agents), Terminal-Bench 2.0 (agentic shell), WebArena (browser agents), GAIA (general assistants), and a focused MCP server are all valid swaps. The architecture below carries over.

---

## What you're building

A single-agent (or one-orchestrator-plus-a-few-subagents-only-if-you-can-defend-it) **autonomous coding agent** that:

- Takes a SWE-bench-Verified instance: a real GitHub issue + the repo at a specific commit
- Reads the code, identifies the relevant files, generates a patch, runs the project's tests
- Reflects on test failures and revises
- Emits a final unified-diff patch + a structured report + a Langfuse trace

Inherits every prior week:

| Roadmap piece | How the capstone uses it |
|---|---|
| W1: tokenization | Token-budgeting the prompt under real context-window limits |
| W2: transformers | Understanding only (you don't train from scratch) |
| W3: HF inference | The base inference path for hosted-model fallbacks |
| W4: structured outputs / function calling | Tool calls + a structured `PatchPlan` schema |
| W5–6: RAG | Retrieval over the *repo* — the right files, the right snippets, the right tests |
| W7–8: fine-tuning | (Optional) a small LoRA-tuned **critic** or **file-localiser** model that picks where to edit; **don't** fine-tune the patcher itself |
| W9: quantization | Serve the small specialised model as AWQ or FP8 |
| W10: vLLM serving | The model server behind the gateway |
| W11: code evals | The eval harness — you already built it; SWE-bench drops in as another dataset |
| W12: AI gateway | Caching shared system prompts, routing simple/hard subtasks, cost tracking |
| W13: tools | `read_file`, `list_dir`, `grep`, `write_patch`, `run_tests`, `git_diff` — exposed via MCP |
| W14: LangGraph | The orchestration: locate → plan → patch → test → reflect |
| W15: eval + obs | Langfuse trace on every run; the eval suite from W15 is the regression baseline |

---

## Day-by-day plan (bias toward more hours than your usual week)

### Day 1 — Lock the spec, the architecture, the eval. **No code today.**

Write `ARCHITECTURE.md` with:

1. **Problem statement** in two sentences
2. **Input contract** — what does the harness pass in? (a SWE-bench instance: repo URL + base commit + issue text + test patch + golden patch hidden)
3. **Output contract** — a `model_patch` unified diff + structured `report.json` (chosen files, attempted strategy, test trajectory, confidence)
4. **System diagram** (mermaid is fine) — every box & arrow you'll build
5. **Eval criteria** — your headline number is "% solved on SWE-bench-Verified-Lite (50 instances)", with `$/instance`, `p95 wall-clock`, `avg iterations`, `tool-call validity`
6. **Non-goals** — for example: no GUI, no human-in-the-loop during the SWE-bench run, no multi-repo, no agents above 3
7. **Failure modes you expect** + what the system does about them
8. **Decision log** for the contested design choices (single vs multi-agent? fine-tune the localiser?). Two sentences each.

**Read (~90 min):**
- [SWE-bench — *official site* + Verified tab](https://www.swebench.com/verified.html)
- [OpenAI — *Introducing SWE-bench Verified*](https://openai.com/index/introducing-swe-bench-verified/) — the 500-task human-cleaned subset
- [SWE-bench — *Submission guide*](https://www.swebench.com/SWE-bench/guides/submissions/) — how to format submissions
- [princeton-nlp/SWE-agent](https://github.com/princeton-nlp/SWE-agent) — the canonical reference architecture for "general LLM → coding agent"; read the README + the `ACI` (agent-computer-interface) docs
- [Anthropic — *2026 Agentic Coding Trends Report (PDF)*](https://resources.anthropic.com/hubfs/2026%20Agentic%20Coding%20Trends%20Report.pdf) — current state-of-the-art landscape
- [AI21 — *How to scale agentic evaluation: 200,000 SWE-bench runs*](https://www.ai21.com/blog/scaling-agentic-evaluation-swe-bench/)

---

### Day 2 — Build the repo-retrieval layer

The single hardest part of a SWE-bench agent is **finding the right file to edit.** Most failures aren't bad patches — they're patches in the wrong place.

Plug your Week-6 advanced-RAG pipeline against the *repo*:

- Index: chunk every `.py` (or whatever language) file, embed, store in Qdrant. Use [`Salesforce/SFR-Embedding-Code-400M_R`](https://huggingface.co/Salesforce/SFR-Embedding-Code-400M_R) or [`Qodo/Qodo-Embed-1-1.5B`](https://huggingface.co/Qodo/Qodo-Embed-1-1.5B) — code-specialised embedding models materially outperform general-purpose ones on retrieval here.
- Query: embed the issue text, retrieve top-20, **rerank** with `bge-reranker-v2-m3` or `mxbai-rerank-large-v2`
- Pair with **BM25** (via [`bm25s`](https://github.com/xhluca/bm25s)) for exact symbol matches — function/class names defeat semantic similarity
- Consider [RAGatouille](https://github.com/AnswerDotAI/RAGatouille) (ColBERTv2) — often wins on code retrieval

**Optional fine-tune (Week 7–8 reused):** train a tiny *file-localiser* — given `(issue_text, candidate_file_list)`, return the top-3 files. 1k synthetic examples from past SWE-bench instances + a Qwen3-1.7B base. This is the legitimate use of fine-tuning in this capstone.

---

### Day 3 — Build the tool layer (Week 13 reused) + MCP wrap

The tools, schemas, and safety boundaries from Week 13 carry over verbatim. Tighten:

- `read_file(path, lines=None)` — must support line ranges; otherwise context blows up
- `list_dir(path)`, `grep(pattern, path)`
- `apply_patch(diff: str)` — *not* `write_file`; SWE-bench wants unified diffs and your agent shouldn't bypass that abstraction
- `run_tests(targets: list[str] | None = None)` — runs the project's own test command in a sandbox
- `view_test_failure(test_id: str)` — focused failure summary; not the whole stack trace
- `git_diff()` — current staged diff

Wrap them in an MCP server so the same tool layer drives both your SWE-bench runner and, optionally, Claude Desktop interactively while you're debugging.

---

### Day 4 — The orchestrator (LangGraph from Week 14)

```
                       issue_in
                          │
                          ▼
                      localise_files  ◄────── RAG (Day 2) [+ optional fine-tune]
                          │
                          ▼
                       plan_patch    (LLM call; emits a structured PatchPlan)
                          │
                          ▼
                       apply_patch
                          │
                          ▼
                       run_tests
                          │
                          ├── pass ──► emit_diff
                          │
                          └── fail
                                │
                                ▼
                          diagnose_failure (LLM call w/ test output)
                                │
                                ▼
                          revise_patch  ──► back to apply_patch  (max 3 revisions)
```

- Use `SqliteSaver` checkpointing so a crash mid-run doesn't lose state
- Stream node events to stdout *and* Langfuse spans
- Cap total iterations + total cost per instance
- A **loop-detector** node: if `apply_patch` is called with a near-identical diff twice, abort

---

### Day 5 — Gateway + serving + cost discipline (Weeks 9–12 reused)

- Front model calls with the Week-12 gateway: semantic cache the repo summary, prompt-cache the system prompt (Anthropic 1-hour TTL), router decides simple-vs-complex
- If you trained a small file-localiser, serve it via vLLM (Week 10) with FP8 / AWQ (Week 9); Claude / GPT for the patch-generation step
- Hardening: timeouts, retries with backoff, circuit breaker on the test sandbox

---

### Day 6 — Run the eval

```bash
# Use the official SWE-bench harness:
pip install swebench
python -m swebench.harness.run_evaluation \
  --predictions_path predictions/swe_bench_verified_mini.jsonl \
  --max_workers 4 \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --instance_ids <50 hand-picked instances spanning difficulty>
```

Build the headline table:

```
config                                  solved/50   pass%   $/inst   p95 wall   avg iters
single agent, frontier only             14          28%     $0.42    11m        4.8
+ RAG (BGE-M3 + bm25s, no rerank)       21          42%     $0.39    12m        4.4
+ rerank (bge-reranker)                 26          52%     $0.41    13m        4.2
+ small fine-tuned file-localiser       29          58%     $0.31    11m        3.9   ← winner
+ revise-on-fail loop                   31          62%     $0.36    14m        4.3
```

Real frontier-model agents are >80% in May 2026. **A junior building this from scratch and landing 25–50% honestly is a strong portfolio piece.** Don't fake the number. The reproducible methodology is the deliverable.

---

### Day 7 — The README, the demo, the retro

**Capstone README** — write it as a senior reviewer would skim it (give specifics, not adjectives):

```markdown
# Autonomous Coding Agent · SWE-bench-Verified-Lite

## Headline number
58% (29/50) solved at $0.31/instance, p95 wall-clock 11 min.

## Problem
## Architecture (with diagram)
## Stack (and why each)
## Retrieval pipeline
## (Optional) Fine-tuned components
## Inference & serving
## Agent design (with a graph image)
## Evaluation methodology
## Observability
## Results
## Failure modes (top 5 with examples)
## What I'd do next
## How to reproduce locally
## Acknowledgements + sources used
```

**Demo (3–5 min screen recording):**

- Show **one easy instance** end-to-end (locate → patch → tests pass → exit)
- Show **one hard instance** with reflect → revise → finally succeeds
- Show **one failure**, walk through the trace in Langfuse, name the failure mode
- Show the eval dashboard with the headline table

Upload to YouTube unlisted. Link from the README.

**Interview prep — be ready to answer:**

- Why this base model / this fine-tune / no fine-tune?
- Why this retrieval stack (and what loses if you remove rerank)?
- Why single-agent and not multi-agent?
- How do you know your eval isn't contamination?
- How do you handle infinite loops / pathological patches?
- What's your $/instance? How would you cut it 5×?
- What's your top failure mode, and what fix would move the number 5%?
- How would you scale this to 1000× the instances?
- What didn't work?

The failure analysis is the most important section in your README. Engineers respect honest weaknesses more than glossy successes.

---

## Deliverable — `swe-bench-coding-agent`

```
swe-bench-coding-agent/
  ARCHITECTURE.md
  README.md
  app/
    api/                       # FastAPI entry (for the interactive mode)
    agent/                     # LangGraph orchestrator
    tools/                     # the tool layer (MCP-wrapped)
    models/                    # adapters: vLLM + frontier (Claude / GPT)
    rag/                       # repo retrieval (BGE-M3 / BM25 / reranker / optional ColBERT)
  models/
    file_localiser/            # (optional) LoRA adapter + training script
  serving/
    docker-compose.yml         # vllm + gateway + langfuse + qdrant
    gateway/                   # Week-12 gateway
  knowledge_base/
    eval_instances/            # the 50 SWE-bench-Verified-Lite instances you ran on
  evals/
    runners/
      run_swebench.py
      run_custom.py
    reports/
      headline_table.md
      failure_modes.md
      cost_breakdown.md
      regression_history.csv
      final_report.md
  observability/
    langfuse/                  # local stack + dashboards
  demo/
    recording.mp4              # (or link)
    screenshots/
  graph.png                    # the LangGraph rendered as PNG
```

---

## Curated resources

This week is mostly **integration**, not new concepts. Re-read the strongest pieces from earlier weeks; the new reading is the SWE-bench / SWE-agent canon.

**The capstone-specific canon (new this week)**
- [SWE-bench Verified leaderboard + page](https://www.swebench.com/verified.html)
- [OpenAI — *Introducing SWE-bench Verified*](https://openai.com/index/introducing-swe-bench-verified/)
- [SWE-bench — *Submission guide*](https://www.swebench.com/SWE-bench/guides/submissions/)
- [princeton-nlp/SWE-agent](https://github.com/princeton-nlp/SWE-agent) — reference architecture + the agent-computer-interface (ACI) paper
- [Anthropic — *2026 Agentic Coding Trends Report* (PDF)](https://resources.anthropic.com/hubfs/2026%20Agentic%20Coding%20Trends%20Report.pdf)
- [AI21 — *How to scale agentic evaluation: 200,000 SWE-bench runs*](https://www.ai21.com/blog/scaling-agentic-evaluation-swe-bench/)
- [Aider leaderboards](https://aider.chat/docs/leaderboards/) — comparison points for context

**Re-read from earlier weeks**
- [Anthropic — *Building Effective Agents*](https://www.anthropic.com/research/building-effective-agents) *(W13)*
- [Anthropic — *Writing effective tools*](https://www.anthropic.com/engineering/writing-tools-for-agents) *(W13)*
- [Anthropic — *Demystifying evals for AI agents*](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) *(W15)*
- [Hamel Husain — *Your AI Product Needs Evals*](https://hamel.dev/blog/posts/evals/) *(W15)*
- [LangGraph — concepts](https://langchain-ai.github.io/langgraph/concepts/) *(W14)*
- [Anthropic — *Contextual Retrieval*](https://www.anthropic.com/news/contextual-retrieval) *(W6)*
- [Sebastian Raschka — *Practical Tips for Finetuning LLMs Using LoRA*](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) *(W8)*
- [vLLM docs](https://docs.vllm.ai/) *(W10)*

**Competitive analysis (study, don't clone)**
- [Cursor](https://cursor.sh/), [Claude Code](https://claude.ai/code), [OpenAI Codex CLI](https://github.com/openai/codex), [Continue.dev](https://continue.dev/), [Cline](https://github.com/cline/cline), [Aider](https://aider.chat/)
- Write a 300-word README section on what each does better than your agent. **This honesty is what separates a portfolio piece from a tutorial.**

---

## "Done when…" checklist

- [ ] `ARCHITECTURE.md` exists and a senior engineer can read it in 5 minutes
- [ ] The full stack runs from one `docker compose up` (or `make demo`)
- [ ] At least one SWE-bench-Verified-Lite instance runs end-to-end without manual intervention
- [ ] Eval ran on ≥ 50 instances; **the headline pass rate is in the README** (whatever it is — be honest)
- [ ] At least 3 configurations are compared in the report (e.g., naive / +RAG / +rerank / +fine-tune / +revise-loop)
- [ ] Langfuse shows clean traces for every run with per-node tokens / cost / latency
- [ ] A 3–5 minute demo video exists
- [ ] README has the 13 sections listed above, including the explicit "What didn't work" section
- [ ] I can pitch this in 60 seconds and answer the 9 interview questions above
- [ ] I've written the longest retro of the roadmap

---

## Common pitfalls

1. **Trying to clone Cursor in a week.** Cut ruthlessly. A working v1 on 50 instances with an honest pass-rate beats a half-built "full IDE."
2. **Cheating the eval.** The agent must not see test contents at planning time; it sees them after `run_tests`. SWE-bench-Verified's test patches are designed to catch this. Cheating gets discovered, and the discovery is permanent.
3. **No failure analysis.** Engineers respect honest weaknesses more than glossy claims. The failure section is the most-read part of the README.
4. **A "production-ready" README for a 50-instance demo.** Avoid hype. Specific numbers ("+18% pass-rate with the localiser at +0.04/$instance") beat any adjective.
5. **Skipping the demo video.** A 3-minute screen recording does more for your portfolio than 3 weeks of README polish.
6. **Multi-agent because it sounds impressive.** Re-read [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents). Default to single-agent. Add subagents only with a written justification.

---

## Alternative capstones (if coding isn't your domain)

Architecture stays the same; swap the dataset + tool layer:

- **TAU2-bench customer-service agent** — [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench). Tool-agent-user evaluation; voice optional. Strong for ML / applied roles in customer-facing products.
- **WebArena / BrowseComp browser agent** — computer-use territory. Requires a browser-automation tool layer; otherwise inherits everything else.
- **GAIA general-assistant agent** — [GAIA](https://huggingface.co/papers/2311.12983). Broader research-style agent eval; benchmarks against ChatGPT Agent / Operator.
- **A focused MCP server + reference client** — pick a real workflow (e.g., "Linear ↔ Slack triage assistant") and ship the MCP server, two reference clients, and an eval set. Less benchmark-friendly but very interview-friendly because most people don't ship MCP yet.

All of these get a public headline number that translates to interview signal.

---

## The end (and the beginning)

You've shipped:

- **15 portfolio repos** (the artifacts of Weeks 1–15)
- **One fine-tuned model on Hugging Face** (W8) + a small optional file-localiser (W16)
- **One quantization benchmark** (W9)
- **One vLLM serving stack** (W10)
- **One evaluation harness** (W11)
- **One production-style AI gateway** (W12)
- **One MCP-wrapped tool-using agent** (W13)
- **One LangGraph orchestrator** (W14)
- **One Langfuse-backed agent eval suite** (W15)
- **One capstone with a SWE-bench-Verified-Lite pass-rate on the page** (W16)

For someone walking in from zero only 4 months ago, that is a serious portfolio. You are now an **AI engineer**, by the standard of any reasonable hiring bar.

Next steps:

- **Apply to AI engineer / applied scientist roles** — your portfolio is the application
- **Specialize deeper** — distributed training, RLHF / GRPO, multi-modal, computer-use, browser agents, embedded
- **Open-source** — turn one of your repos into something other people use; submit your SWE-bench results to the leaderboard
- **Write** — blog about what worked and what didn't; AI engineering content from people who actually shipped is rare and valuable
- **Iterate on the capstone** — ship one improvement per week; let it become your product

Whichever path you pick, keep the rule from the master README:

> Every project must produce a working artifact.
> Every model change must have an eval.
> Every eval must have a report.
> Every report must include failure cases.

That's the practice. Everything else is detail.

---

← Previous: [Week 15 — Agent Evaluation + Observability](./WEEK-15.md)
