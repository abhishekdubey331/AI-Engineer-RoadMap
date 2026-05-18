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
3. Use **Langfuse** (the recommended default; LangSmith / Phoenix only if you have a real reason) to capture traces of every agent run
4. Annotate failed traces with categories and surface aggregate failure-mode stats
5. Set up regression alerts (when a metric drops, you know)
6. Use **Inspect AI** or similar to run agent benchmarks reproducibly

---

## Day-by-day plan (≈ 2–3 hrs/day)

### Day 1 — The four pillars of agent eval

**Read (90 min):**
- [Anthropic — *Demystifying evals for AI agents*](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — **must read**
- [Hamel Husain — *Your AI Product Needs Evals*](https://hamel.dev/blog/posts/evals/) — **must read** (the most-cited practitioner reference)
- [Hamel Husain — *A Field Guide to Rapidly Improving AI Products*](https://hamel.dev/blog/posts/field-guide/) — operating playbook
- [Hamel Husain — *LLM-as-a-Judge*](https://hamel.dev/blog/posts/llm-judge/) — for the trajectory-quality scoring you'll do on Day 7
- [Galileo — *Agent Evaluation Framework*](https://galileo.ai/blog/agent-evaluation-framework-metrics-rubrics-benchmarks) — useful framing; vendor-flavored, read with that in mind

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
- Mix categories: bug fix, refactor, new feature, doc update, test writing, code-review feedback application
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

**Be opinionated: use Langfuse.** OSS, self-hostable, transparent volume-based pricing, Apache 2.0, feature parity between self-host and cloud. Reach for **LangSmith** instead only if you're committed to LangChain's cloud; reach for **Arize Phoenix** only if you live in OpenTelemetry-native infra. The constant "pick whatever you like" advice is paralysis for juniors — pick Langfuse, ship, swap later if you have a real reason. See the [Langfuse-vs-Phoenix comparison](https://www.zenml.io/blog/langfuse-vs-phoenix) for the trade-off.

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
- [OpenTelemetry — *GenAI semantic conventions*](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the emerging cost / latency telemetry standard

**Hands-on:**
- Make your eval suite double as a **regression suite** in Langfuse
- After every code change, run it; the dashboard shows delta vs the previous "blessed" run
- Set an alert (or just a Slack webhook from a small script): if success rate drops by >5% or cost rises by >20%, ping

---

### Day 6 — Run a real agent benchmark with Inspect AI

Beyond your custom tasks, run your agent against a real public benchmark to know where it stands.

**Read (45 min):**
- [Inspect AI — *Getting started*](https://inspect.aisi.org.uk/)
- [Inspect AI — *Solvers and Agents*](https://inspect.aisi.org.uk/agents.html)
- [Inspect AI — *Sandboxing*](https://inspect.aisi.org.uk/agents/sandboxing.html)
- [Phil Schmid — *AI Agent Benchmark Compendium*](https://github.com/philschmid/ai-agent-benchmark-compendium) — pick one benchmark suited to your domain

**Hands-on (~2 hr):**
- Run your agent against **one** of these (pick to match your Week-16 capstone direction):
  - **SWE-bench-Verified (Lite subset, ~10 instances)** — for the coding-agent capstone. Required for the W16 path.
  - **Terminal-Bench 2.0** (agentic-shell tasks)
  - **TAU2-bench** ([sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)) — tool-agent-user evaluation; canonical for customer-service-style agents
  - **GAIA** (general assistant)
- Save the results. Even a low score is portfolio-worthy because you know how to *run* the harness — that's the rare skill.

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
- **One public benchmark run** (SWE-bench-Verified-Lite subset for the W16 path, or Terminal-Bench 2.0 / TAU2-bench / GAIA) with a results report — this becomes the headline number in W16
- **Two-judge calibration:** LLM-judge for trajectory quality must use **two judges from different model families** + calibration against ≥10 human-labeled examples. This single discipline separates serious evals from theater.
- A `final_report.md` that's frank about your agent's weaknesses

### Stretch

- A/B run with two LangGraph variants (e.g., different reflection strategies) and report which wins
- Export traces to OpenTelemetry → Jaeger so non-LLM-aware teammates can read them (use [OTel GenAI conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) from Week 12)

---

## Curated resources

**Concepts (canonical)**
- [Anthropic — *Demystifying evals for AI agents*](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [Hamel Husain — *Your AI Product Needs Evals*](https://hamel.dev/blog/posts/evals/) — required
- [Hamel Husain — *A Field Guide to Rapidly Improving AI Products*](https://hamel.dev/blog/posts/field-guide/)
- [Hamel Husain — *LLM-as-a-Judge*](https://hamel.dev/blog/posts/llm-judge/)
- [Eugene Yan — *LLM Evaluators*](https://eugeneyan.com/writing/llm-evaluators/)
- [Phil Schmid — *AI Agent Benchmark Compendium*](https://github.com/philschmid/ai-agent-benchmark-compendium) — 50+ benchmarks catalogued
- [AI21 — *How to scale agentic evaluation: 200,000 SWE-bench runs*](https://www.ai21.com/blog/scaling-agentic-evaluation-swe-bench/)

**Observability platforms — pick Langfuse unless you have a real reason not to**
- [Langfuse docs](https://langfuse.com/docs) — **default recommendation**
- [Langfuse — *vs Phoenix*](https://www.zenml.io/blog/langfuse-vs-phoenix)
- [LangSmith docs](https://docs.smith.langchain.com/) — best if you're committed to LangChain's cloud
- [Arize Phoenix](https://docs.arize.com/phoenix) — best if you're OpenTelemetry-native
- [W&B Weave](https://wandb.ai/site/weave) — best if you already live in W&B for experiments

**Eval frameworks**
- [Inspect AI](https://inspect.aisi.org.uk/) — modern, sandboxed, agent-aware
- [DeepEval — *AI Agent Evaluation*](https://deepeval.com/guides/guides-ai-agent-evaluation)
- [promptfoo](https://www.promptfoo.dev/) — fast eval for prompts, lightweight

**Benchmarks worth knowing**
- [SWE-bench Verified](https://www.swebench.com/verified.html) — the canonical coding-agent eval
- [Terminal-Bench 2.0](https://www.tbench.ai/)
- [TAU2-bench](https://github.com/sierra-research/tau2-bench) — tool-agent-user
- [Aider Polyglot](https://aider.chat/docs/leaderboards/) — file-editing in 6 languages
- [GAIA (Meta)](https://huggingface.co/papers/2311.12983)
- [WebArena](https://webarena.dev/) + [BrowseComp](https://openai.com/index/browsecomp/) — web agents
- [ARC-AGI 2](https://arcprize.org/) — reasoning frontier

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

← Previous: [Week 14 — Agent Orchestration with LangGraph](./WEEK-14.md) · → Next: [Week 16 — Capstone: Autonomous Coding Agent on SWE-bench-Verified-Lite](./WEEK-16.md)
