# Week 16 — Capstone: Design-to-Code Agent

> **Month 4 · Agents, Evaluation, Capstone**
> *"This is the week the 15 pieces become one product. Everything you've built is on trial. If a piece doesn't earn its place, you cut it."*

---

## Why this week

For 15 weeks you've built standalone pieces. The capstone is where you stitch them into a system that looks like something a real team would ship — and that *you* can talk about for 30 minutes in an interview without running out of substance.

The capstone is the **Design-to-Code Agent**: a system that takes a design specification (component description, design tokens, optional Figma-like JSON) and produces production-quality, tested, design-system-compliant React (or your framework of choice) components.

This is not "the only" capstone. If you have a different specialty (SQL agents, refactoring agents, browser agents, dev-tools), substitute. The architecture template stays the same.

---

## Why this project, specifically

It hits every skill the roadmap built:

| Roadmap piece | How the capstone uses it |
|---|---|
| Week 1: tokenization | Token-budgeting your context window in production |
| Week 2: transformers | (Understanding only — you don't train from scratch in the capstone) |
| Week 3: HF inference | The base inference path for hosted-model fallbacks |
| Week 4: structured outputs / function calling | Component spec → typed JSON, then tool calls |
| Week 5–6: RAG | Retrieve the right design tokens, component examples, accessibility rules |
| Week 7–8: fine-tuning | A small LoRA-tuned model for the narrow "design spec → component" subtask |
| Week 9: quantization | Ship the fine-tuned model as AWQ or GGUF |
| Week 10: vLLM serving | The model server behind the gateway |
| Week 11: code evals | Component tests, accessibility checks, visual regression |
| Week 12: AI gateway | Caching, routing (small local fine-tune for simple cases; frontier model for tricky ones), cost tracking |
| Week 13: tools | `read_design_system`, `lookup_token`, `render_preview`, `write_component`, `run_a11y_check` |
| Week 14: LangGraph | The orchestration; plan → retrieve → generate → preview → fix → test |
| Week 15: eval + obs | Langfuse trace on every run; capstone eval suite with a regression baseline |

---

## Day-by-day plan (~2–3 hrs/day, but bias toward more this week)

### Day 1 — Lock the spec, the architecture, the eval

This is the most important day of the week. **No code today.**

**Write a 3-page architecture doc (`ARCHITECTURE.md`) with:**

1. **Problem statement** in two sentences
2. **Input contract** — what does the user send in?
3. **Output contract** — what comes back? (file paths, test results, preview URL, confidence)
4. **System diagram** (mermaid is fine) — every box & arrow you'll build
5. **Eval criteria** — how you'll measure success and what numbers count as "v1 ships"
6. **Non-goals** — what you'll *not* build (multi-page apps, server-rendering, etc.)
7. **Failure modes you expect** + what the system does about them

**Example input contract:**
```json
{
  "screen": "Pricing page",
  "components": [
    {"type": "navbar", "variant": "default"},
    {"type": "pricing_card", "plan": "Pro"},
    {"type": "button", "variant": "primary", "label": "Start free trial"}
  ],
  "style": {"theme": "modern SaaS", "spacing": "comfortable"},
  "constraints": ["accessible", "responsive", "uses design-system tokens"]
}
```

**Example output contract:**
```json
{
  "files": [{"path": "Pricing.tsx", "diff": "..."}],
  "tests": {"unit": "passed", "a11y": "passed", "visual_regression": "0 diffs"},
  "preview_url": "http://localhost:3000/__preview__/Pricing",
  "trace_id": "thr-...",
  "confidence": 0.84,
  "warnings": []
}
```

---

### Day 2 — Build the knowledge base (the RAG part)

Create a small but realistic **design-system knowledge base**:

- `tokens.json` — colors, spacing, typography, radii (~50 tokens)
- `components/` — 10 reference component MDX files: each with name, description, props schema, code, do/don't, a11y notes
- `examples/` — 5–10 example screens / pages composed from the components
- `a11y_rules.md` — accessibility checklist your generator will be expected to honor

Wire your **Week-6 advanced RAG pipeline** over it. Verify retrieval works for queries like "give me an accessible pricing card" or "how do I use the spacing-md token."

---

### Day 3 — Specialize the model (your fine-tune, plugged in)

Take your Week-8 fine-tuned model — or re-fine-tune now with a fresh 1k examples for the *exact* narrow task: `JSON component spec → component skeleton`.

Quantize it to AWQ or GGUF (Week 9). Serve it behind vLLM (Week 10).

Now the agent can call **two models**:
- Your specialized small model — fast, cheap, narrow
- A frontier model (Claude / GPT-4o) — for the hard / open-ended generation

The Week-12 router picks between them.

---

### Day 4 — The agent graph

Build the LangGraph orchestrator:

```
                       parse_request
                            │
                            ▼
                       retrieve_context  ◄────── (your Week-6 RAG)
                            │
                            ▼
                       plan_components
                            │
                            ├──► generate_component (specialized model)
                            │             │
                            │             ▼
                            │       lint + a11y_check
                            │             │
                            │             ▼
                            │       run_unit_tests
                            │             │
                            │             ▼
                            │       reflect + retry? ──► (loop, max 2)
                            │             │
                            │             ▼
                            └──► assemble_screen
                                        │
                                        ▼
                                  render_preview
                                        │
                                        ▼
                                  hil_approve  ◄───── (Week 14 HIL gate)
                                        │
                                        ▼
                                  write_files + final_report
```

- Use checkpointing (`SqliteSaver`)
- Stream node events to stdout
- Each node sends a span to Langfuse

---

### Day 5 — The gateway + tools + safety

- Wire the Week-12 AI gateway in front of model calls (cache, route, retry, cost track, PII redact)
- Tool layer (Week 13): `read_design_system`, `lookup_token`, `render_preview` (spin up a sandboxed `next dev` or Vite server), `run_unit_test`, `run_a11y_check` (axe-core), `write_component_file`
- All file writes go through the HIL approval node

---

### Day 6 — Eval, benchmark, optimize

Build the capstone eval suite:

- 20+ component-generation tasks, each with:
  - Input spec (JSON)
  - Expected behavior
  - Unit tests (rendering, props, events)
  - Accessibility test (axe + manual)
  - Visual regression baseline (one happy-path screenshot)
  - Success criteria

Run the full system 3× per task. Tabulate per the Week-15 four pillars:

```
configuration                 success  iters  tokens   cost   p95 latency
base frontier only            0.65     5.2    34k      $0.18  18s
small fine-tune only          0.55     7.1    22k      $0.02  9s
router (small + frontier)     0.85     5.7    18k      $0.08  11s   ← winner
router + RAG-on               0.92     5.1    20k      $0.09  12s   ← winner+
router + RAG + reflect-loop   0.94     6.4    27k      $0.12  17s
```

Pick the configuration. Document why.

---

### Day 7 — The README, the demo, the retro

**Capstone README** (`README.md` in the capstone repo) — write it as a recruiter would skim it:

```markdown
# Design-to-Code Agent

## Problem
## Architecture (with diagram)
## Stack (and why each)
## Data pipeline
## Fine-tuning
## Inference optimization
## Agent design
## Evaluation
## Observability
## Results (headline numbers, with the chart)
## Failure modes (be honest)
## Future work
## How to run it locally
## Acknowledgements + resources used
```

**Demo** — record a 3–5 minute screen recording:
- Show one easy task end-to-end (10–20s of model output)
- Show one harder task with a failure → reflect → retry
- Show the Langfuse trace
- Show the eval dashboard

Upload to YouTube unlisted. Link from the README.

**Interview prep** — be ready to explain:
- Why you fine-tuned instead of only prompting
- Why you used RAG
- What quantization changed (numbers)
- How you measured quality
- How your agent avoids infinite loops
- How you handle bad tool calls
- How you'd scale to 1000× more users
- What failed

---

## Deliverable — `design-to-code-agent`

```
design-to-code-agent/
  ARCHITECTURE.md
  README.md
  app/
    api/                  # FastAPI entry
    agent/                # LangGraph orchestrator
    tools/                # the tool layer
    models/               # adapters for specialized + frontier models
    rag/                  # Week 6 pipeline, parameterized
  models/
    finetune/             # adapter weights / training script (link to HF Hub)
  serving/
    docker-compose.yml    # vllm + gateway + langfuse + qdrant
    gateway/              # Week 12 gateway
  knowledge_base/
    tokens.json
    components/
    examples/
    a11y_rules.md
  evals/
    tasks/
    runners/
    reports/
      v1_results.md
      headline_chart.png
  observability/
    langfuse/             # local stack + dashboards
  demo/
    recording.mp4         # (or link)
    screenshots/
  notebooks/              # for visualization, optional
```

---

## Curated resources

This week is mostly about **integration**, not new concepts. Re-read the strongest pieces from earlier weeks:

- [Anthropic — *Building Effective Agents*](https://www.anthropic.com/research/building-effective-agents) (Week 13)
- [Anthropic — *Writing effective tools*](https://www.anthropic.com/engineering/writing-tools-for-agents) (Week 13)
- [Anthropic — *Demystifying evals for AI agents*](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) (Week 15)
- [LangGraph — *Concepts*](https://langchain-ai.github.io/langgraph/concepts/) (Week 14)
- [Sebastian Raschka — *Practical Tips for Finetuning LLMs Using LoRA*](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) (Week 8)
- [vLLM docs](https://docs.vllm.ai/) (Week 10)
- [Anthropic — *Contextual Retrieval*](https://www.anthropic.com/news/contextual-retrieval) (Week 6)

**Inspiration / state-of-the-art reads:**
- [Anthropic — *Claude's research mode* engineering posts (mostly applicable patterns)](https://www.anthropic.com/engineering)
- [Vercel AI SDK — *Generative UI*](https://vercel.com/blog/generative-user-interfaces)
- [Galileo Pixtral / Locofy / v0 by Vercel](https://v0.dev/) — products in this space; study their UX
- [HuggingFace — *Open Source AI Cookbook*](https://huggingface.co/learn/cookbook/index)

---

## "Done when…" checklist

- [ ] `ARCHITECTURE.md` exists and is readable by a senior engineer in 5 minutes
- [ ] The full stack runs from one `docker compose up` (or a `make demo` target)
- [ ] At least one input runs end-to-end without manual intervention
- [ ] The HIL gate works and rejects bad outputs cleanly
- [ ] Eval suite (20+ tasks) produces a `report.md` with headline numbers
- [ ] At least 3 model configurations are compared in the report
- [ ] Langfuse shows a clean trace for every run
- [ ] A 3–5 minute demo exists
- [ ] README has the 11 sections listed above
- [ ] I can pitch this in 60 seconds and answer the 8 interview questions above
- [ ] I've written the longest retro of the roadmap. Be honest — this is the artifact future you (and future readers) will return to.

---

## Common pitfalls this week

1. **Trying to build everything.** Cut ruthlessly. A working v1 with one screen type and 20-task eval beats a half-built "full system."
2. **No HIL.** A demo without an approval gate looks like a toy. With one, it looks like a product.
3. **A README that sounds like a brochure.** Recruiters smell hype. Be specific about numbers — "+15% pass rate vs base model on internal eval, with 4× lower cost via routing."
4. **No failure analysis.** Engineers respect honest weaknesses far more than glossy success claims.
5. **Skipping the demo.** A 3-minute video does more for your portfolio than 3 weeks of README polish.

---

## The end (and the beginning)

You've shipped:
- **12 portfolio repos** (the artifacts of Weeks 1–12)
- **One fine-tuned model on Hugging Face** (Week 8)
- **One quantization benchmark** (Week 9)
- **One evaluation harness** (Week 11)
- **One production-style AI gateway** (Week 12)
- **One agent eval + observability layer** (Week 15)
- **One capstone end-to-end system** (Week 16)

For someone walking in from zero only 4 months ago, that is a serious portfolio. You are now an **AI engineer**, by the standard of any reasonable hiring bar.

The next steps depend on where you want to go:

- **Apply to AI engineer / applied scientist roles** — your portfolio is the application
- **Specialize deeper** — distributed training, RLHF, multi-modal, embedded
- **Open-source** — turn one of your repos into something other people use
- **Write** — blog about what worked and what didn't; AI engineering content from people who actually shipped is rare
- **Iterate on the capstone** — every week, ship one improvement; let it become *your* product

Whichever path you pick, keep the rule from the master README:

> Every project must produce a working artifact.
> Every model change must have an eval.
> Every eval must have a report.
> Every report must include failure cases.

That's the practice. Everything else is detail.

---

← Previous: [Week 15 — Agent Evaluation + Observability](./WEEK-15.md)
