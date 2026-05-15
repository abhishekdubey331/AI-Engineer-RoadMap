# Week 15 — Agent Evaluation + Observability

> **Month 4 · Agents, Evaluation, Capstone**
> *"If you can't see what your agent did, you can't fix it. If you can't measure if it's getting better, you'll never know if you broke it."*

---

## Why this week

Agents amplify everything. A small bug in a single LLM call becomes a 20-step cascading failure when it's inside an agent. Without observability, debugging is impossible. Without evaluation, every change is a coin flip.

This week you set up the **traceability + measurement** layer that every Week-16 capstone change will rely on. You also harden your agent eval — moving from "did it work?" to "did it work, with what cost, with how many wrong tool calls, with what failure mode?"

---

## Learning objectives

By Sunday night you should be able to:

1. Define and compute the agent-evaluation pillars: **task success**, **trajectory quality**, **tool-call accuracy**, **cost efficiency**
2. Build an agent eval suite with at least 20 hand-curated tasks
3. Use **Langfuse** (or LangSmith, or Arize Phoenix) to capture traces of every agent run
4. Annotate failed traces with categories and surface aggregate failure-mode stats
5. Set up regression alerts (when a metric drops, you know)
6. Use **Inspect AI** or similar to run agent benchmarks reproducibly

---

## Day-by-day plan (≈ 2–3 hrs/day)

### Day 1 — The four pillars of agent eval

**Read (75 min):**
- [Anthropic — *Demystifying evals for AI agents*](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — must read
- [Galileo — *Agent Evaluation Framework: Metrics, Rubrics, Benchmarks*](https://galileo.ai/blog/agent-evaluation-framework-metrics-rubrics-benchmarks)
- [IBM — *What is AI Agent Evaluation?*](https://www.ibm.com/think/topics/ai-agent-evaluation)

**The four pillars:**

1. **Task success** — did the agent achieve the goal? (binary or graded)
2. **Trajectory quality** — was the path efficient and sensible, or did it stumble around?
3. **Tool quality** — were tool calls valid, with correct arguments, in the right order?
4. **Cost efficiency** — tokens, latency, $/successful-task

Outcome metrics scale; trajectory evaluation is for debugging. Use both — outcome metrics in CI, trajectory grading on a curated slice.

---

### Day 2 — Build the agent eval suite

**Hands-on (~2.5 hr):**

Build `agent_eval/` with:

- `tasks/` — 20+ tasks, each with:
  ```json
  {
    "task_id": "T-001",
    "description": "Add a CLI flag --json to print output as JSON",
    "repo_snapshot": "fixtures/T-001/",        // a fresh tarball each run
    "success_test": "pytest tests/test_json_flag.py",
    "max_iters": 15,
    "max_cost_usd": 0.05,
    "category": "cli_change",
    "difficulty": "easy"
  }
  ```
- Mix categories: bug fix, refactor, new feature, doc update, test writing, design-to-code
- Mix difficulties: easy / medium / hard

- `run.py` — runs every task N times against your Week-14 agent, captures full traces
- `score.py` — computes:
  - `task_success_rate` (per task, then aggregate)
  - `avg_iters`, `avg_tokens`, `avg_cost`, `avg_wallclock`
  - `tool_call_validity` (% of tool calls that didn't error)
  - `loop_count` (how often did the loop-detector trip)

---

### Day 3 — Tracing with Langfuse

**Read (45 min):**
- [Langfuse — *Getting started*](https://langfuse.com/docs/get-started)
- [Langfuse — *Tracing concepts*](https://langfuse.com/docs/tracing)
- [Langfuse — *Integrations: LangChain / LangGraph*](https://langfuse.com/docs/integrations/langchain/tracing)
- [Langfuse — *Evaluation*](https://langfuse.com/docs/evaluation/overview)

**Why Langfuse:** it's the strongest OSS option with feature parity between self-host and cloud; transparent volume-based pricing; Apache 2.0 license. Read also the [Langfuse-vs-Phoenix comparison](https://www.zenml.io/blog/langfuse-vs-phoenix) — Phoenix and LangSmith are both fine alternatives; pick one and commit.

**Hands-on (90 min):**
- Self-host Langfuse: `docker compose` (their repo has a compose file)
- Wire your Week-14 LangGraph agent to send traces (LangGraph has a native callback handler)
- Run 5 tasks and inspect the traces in the UI: spans per node, token counts, latency, tool args

---

### Day 4 — Annotate, categorize, debug

Traces alone are noise. Tags + annotations turn them into intelligence.

**Read (45 min):**
- [Langfuse — *Annotation queues*](https://langfuse.com/docs/scores/annotation)
- [Langfuse — *Custom scores*](https://langfuse.com/docs/scores/custom)

**Hands-on (~2 hr):**
- Pick the 20 failed runs from Day 2
- Categorize each manually using a small taxonomy:
  - `hallucinated_tool_call` — agent invented an arg
  - `wrong_tool_choice` — picked a tool that can't solve the subtask
  - `infinite_loop` — same call repeating
  - `early_termination` — gave up before finishing
  - `bad_plan` — plan was wrong from the start
  - `correct_plan_bad_execution`
  - `dataset_bug` — eval task itself was wrong
- Push annotations into Langfuse
- Make a chart: failure modes by frequency. The top 3 are your priorities for the capstone.

---

### Day 5 — Cost & regression alerts

**Read (60 min):**
- [Langfuse — *Cost tracking*](https://langfuse.com/docs/integrations/llm-cost)
- [Langfuse — *Datasets & experiments*](https://langfuse.com/docs/datasets/overview) — used for regression suites
- [DigitalApplied — *Agent Observability: LangSmith, Langfuse, Arize (2026)*](https://www.digitalapplied.com/blog/agent-observability-platforms-langsmith-langfuse-arize-2026)
- [Maxim — *Top 5 LLM Observability Platforms for 2026*](https://www.getmaxim.ai/articles/top-5-llm-observability-platforms-for-2026/)

**Hands-on:**
- Make your eval suite double as a **regression suite** in Langfuse
- After every code change, run it; the dashboard shows delta vs the previous "blessed" run
- Set an alert (or just a Slack webhook from a small script): if success rate drops by >5% or cost rises by >20%, ping

---

### Day 6 — Run a real agent benchmark with Inspect AI

Beyond your custom tasks, run your agent against a real public benchmark to know where it stands.

**Read (45 min):**
- [Inspect AI — *Getting started*](https://inspect.ai-safety-institute.org.uk/)
- [Inspect AI — *Solvers and Agents*](https://inspect.ai-safety-institute.org.uk/agents.html)
- [Inspect AI — *Sandboxing*](https://inspect.ai-safety-institute.org.uk/agents/sandboxing.html)
- [Phil Schmid — *AI Agent Benchmark Compendium*](https://github.com/philschmid/ai-agent-benchmark-compendium) — pick one benchmark suited to your domain

**Hands-on (~2 hr):**
- Run your agent against either:
  - **SWE-bench-Lite** (10 instances — full run is expensive)
  - **TerminalBench** (agentic-shell tasks)
  - **GAIA** (general assistant, a few examples)
- Save the results. Even a low score here is portfolio-worthy because you know how to do it.

---

### Day 7 — Project + retro

Polish + write-up + commit dashboards.

---

## Weekly Project — `agent-eval-and-obs`

The end-to-end evaluation + observability layer for your Week-14 agent.

### Spec

```
agent-eval-and-obs/
  tasks/                       # 20+ hand-curated tasks (JSONL + fixtures)
  benchmarks/
    swe_bench_lite_subset/     # 10 picked instances
  runners/
    run_custom.py
    run_swebench.py
    run_inspect.py
  scoring/
    metrics.py                 # task success, trajectory, tool, cost
    regression.py              # compare current run vs blessed baseline
  langfuse/
    docker-compose.yml         # self-hosted Langfuse stack
    init_sql/
    annotations_taxonomy.md
  dashboards/
    grafana/                   # optional, for system metrics
    langfuse_screenshots/
  reports/
    failure_modes.md
    benchmark_results.md
    regression_history.csv
    final_report.md
  README.md
```

### Requirements

- 20+ custom tasks across ≥4 categories
- Every task runs in a sandboxed env (Docker; from Week 11)
- Full tracing in Langfuse with per-node spans, token counts, $/run
- An `annotations_taxonomy.md` and ≥30 annotated traces
- A bar chart of failure modes by frequency (in `reports/failure_modes.md`)
- A regression baseline + comparison script
- One public benchmark run (SWE-bench-Lite subset, TerminalBench, or GAIA) with a results report
- A `final_report.md` that's frank about your agent's weaknesses

### Stretch

- Add an LLM-judge layer for trajectory quality (using a different model family as the judge)
- A/B run with two LangGraph variants (e.g., different reflection strategies) and report which wins
- Export traces to OpenTelemetry → Jaeger so non-LLM-aware teammates can read them

---

## Curated resources

**Concepts**
- [Anthropic — *Demystifying evals for AI agents*](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [Galileo — *Agent Evaluation Framework*](https://galileo.ai/blog/agent-evaluation-framework-metrics-rubrics-benchmarks)
- [Confident AI — *Definitive AI Agent Evaluation Guide*](https://www.confident-ai.com/blog/definitive-ai-agent-evaluation-guide)
- [Turing College — *Evaluating AI Agents: A Practical Guide*](https://www.turingcollege.com/blog/evaluating-ai-agents-practical-guide)
- [Phil Schmid — *AI Agent Benchmark Compendium*](https://github.com/philschmid/ai-agent-benchmark-compendium) — 50+ benchmarks catalogued
- [AI21 — *How to scale agentic evaluation: lessons from 200,000 SWE-bench runs*](https://www.ai21.com/blog/scaling-agentic-evaluation-swe-bench/)

**Observability platforms (pick one and commit)**
- [Langfuse docs](https://langfuse.com/docs)
- [Langfuse — *vs Phoenix*](https://www.zenml.io/blog/langfuse-vs-phoenix)
- [LangSmith docs](https://docs.smith.langchain.com/)
- [Arize Phoenix](https://docs.arize.com/phoenix)
- [W&B Weave](https://wandb.ai/site/weave)
- [DigitalApplied — *Agent Observability 2026*](https://www.digitalapplied.com/blog/agent-observability-platforms-langsmith-langfuse-arize-2026)
- [Maxim — *Top 5 LLM Observability Platforms for 2026*](https://www.getmaxim.ai/articles/top-5-llm-observability-platforms-for-2026/)
- [Kanerika — *LangSmith vs Arize vs Langfuse vs W&B*](https://medium.com/@kanerika/llmops-observability-langsmith-vs-arize-vs-langfuse-vs-w-b-f1baeabd1bbf)
- [Latitude — *Best LLM Observability Tools for Agents (2026)*](https://latitude.so/blog/best-llm-observability-tools-agents-latitude-vs-langfuse-langsmith)

**Eval frameworks**
- [Inspect AI](https://inspect.ai-safety-institute.org.uk/)
- [DeepEval — *AI Agent Evaluation*](https://deepeval.com/guides/guides-ai-agent-evaluation)
- [promptfoo](https://www.promptfoo.dev/)

**Benchmarks worth knowing**
- [SWE-bench](https://www.swebench.com/) (and SWE-bench-Lite, SWE-bench-Verified)
- [TerminalBench](https://www.tbench.ai/)
- [GAIA (Meta)](https://huggingface.co/papers/2311.12983)
- [WebArena](https://webarena.dev/) — web agents
- [AgentBench](https://github.com/THUDM/AgentBench)

---

## "Done when…" checklist

- [ ] I can list the four pillars of agent eval and what each measures
- [ ] My 20-task eval suite runs end-to-end with sandboxed execution
- [ ] Every agent run produces a Langfuse trace I can inspect
- [ ] I have 30+ annotated traces and a chart of failure modes by frequency
- [ ] I have a regression check that compares current vs blessed run
- [ ] My agent has been benchmarked on at least one public benchmark (results may be low — that's fine)
- [ ] My `final_report.md` is honest about weaknesses
- [ ] I've written a retro

---

## Common pitfalls this week

1. **Aggregate-only metrics.** A 70% success rate hides which 30% always fails. Always break down by task, category, difficulty.
2. **One run per task.** Agents are stochastic; `n>=3` is the minimum.
3. **No annotation discipline.** If "failure" isn't categorized, you can't prioritize.
4. **Trusting LLM judges blindly.** Always validate the judge against human annotations on a sample.
5. **Stale baseline.** If you update the prompt and forget to re-bless the baseline, your regressions become noise. Make re-blessing a one-command step.

---

← Previous: [Week 14 — LangGraph Orchestration](./WEEK-14.md) · → Next: [Week 16 — Capstone: Design-to-Code Agent](./WEEK-16.md)
