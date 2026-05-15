# Week 13 — Tool Calling + Agent Design Patterns

> **Month 4 · Agents, Evaluation, Capstone**
> *"An agent is not magic. It's a loop where an LLM picks a tool, you run the tool, and you feed the result back. Get the loop right; the magic follows."*

---

## Why this week

Agents are the highest-leverage thing you can build with an LLM today — and the buggiest. Most "agents" you see in demos fall apart on real tasks because:

- The tools are badly designed
- The loop has no termination condition
- There's no error recovery
- There's no observability

This week you fix the foundation. You learn the design patterns (from Anthropic's *Building Effective AI Agents*, which is the most important agent reading of the last two years), build a tool-using code assistant by hand, and learn what makes a tool "agent-friendly."

You do **not** start with LangGraph this week. That's next week. This week is the primitives, so the framework doesn't feel like a black box later.

---

## Learning objectives

By Sunday night you should be able to:

1. Explain the difference between a **workflow** (predefined paths) and an **agent** (LLM-directed paths) per Anthropic's definition
2. Name and use the 5 common agent patterns: augmented LLM, prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer
3. Design good tools: clear names, focused purpose, descriptive errors, token-efficient outputs
4. Build a hand-rolled tool-use loop with retries, validation, and safety boundaries
5. Use the OpenAI Agents SDK for a small agent and know when to reach for it
6. Identify the 3 most common agent failure modes and the defenses against them

---

## Day-by-day plan (≈ 2–3 hrs/day)

### Day 1 — *Building Effective AI Agents* (Anthropic)

This is the single most important reading for the rest of the roadmap. Read it slowly.

**Read (90 min):**
- [Anthropic — *Building Effective Agents*](https://www.anthropic.com/research/building-effective-agents) — **mandatory, top to bottom**
- [Anthropic — *Architecture Patterns and Implementation Frameworks (PDF)*](https://resources.anthropic.com/hubfs/Building%20Effective%20AI%20Agents-%20Architecture%20Patterns%20and%20Implementation%20Frameworks.pdf)

**Patterns to lock in:**

| Pattern | When to use |
|---|---|
| **Augmented LLM** | Single model + tools + retrieval. The default. Start here. |
| **Prompt chaining** | Predefined sequence of LLM calls (e.g., draft → critique → revise) |
| **Routing** | Classifier sends each input to a specialized chain (Week 12 already taught you this!) |
| **Parallelization** | Fan out (e.g., 5 evals at once) or voting (sample-3, pick-best) |
| **Orchestrator-workers** | Lead LLM decomposes a task, dispatches to worker LLMs |
| **Evaluator-optimizer** | One LLM generates, another judges; loop until "good enough" |
| **Agents** | LLM decides each next action in a loop with tool feedback |

---

### Day 2 — Writing good tools

Half of agent quality comes from tool design. Anthropic published a whole engineering post on this, which most people miss.

**Read (75 min):**
- [Anthropic — *Writing effective tools for AI agents — using AI agents*](https://www.anthropic.com/engineering/writing-tools-for-agents) — **mandatory**
- [Anthropic — *Tool use overview*](https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview)
- [Anthropic — *Implement tool use*](https://docs.claude.com/en/docs/agents-and-tools/tool-use/implement-tool-use)
- [OpenAI — *Function calling*](https://platform.openai.com/docs/guides/function-calling) (you read this in Week 4; revisit with agent context)

**Principles to internalize:**
- Tool names are part of the prompt. Make them *unambiguous* (`read_file` > `read`)
- Tool descriptions are the most important text in your agent. Spend time on them.
- Tool *outputs* are the second most important. Don't return a 50KB JSON blob — return a focused, agent-readable summary
- Error messages should tell the agent what to do next, not just what went wrong
- "Combinability" — design tools that compose well

---

### Day 3 — Build the loop by hand

No frameworks today. Just a `while`.

**Hands-on (~2.5 hr):**

```python
def agent_loop(user_query: str, tools: dict, model: str, max_iters: int = 15):
    messages = [{"role": "user", "content": user_query}]
    for _ in range(max_iters):
        response = call_llm(model, messages, tools=list(tools.values()))
        messages.append({"role": "assistant", "content": response.content, "tool_calls": response.tool_calls})

        if not response.tool_calls:
            return response.content   # final answer

        for tc in response.tool_calls:
            try:
                result = tools[tc.name](**tc.arguments)
            except Exception as e:
                result = {"error": str(e)}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    return "Max iterations reached without a final answer."
```

Then build 5 real tools for the agent to use:
- `read_file(path: str) -> str`
- `list_files(directory: str) -> list[str]`
- `grep(pattern: str, path: str) -> list[Match]`
- `write_file(path: str, content: str) -> None`
- `run_tests() -> TestResult`

Run the agent on small, realistic tasks. Watch the trace. Take notes.

---

### Day 4 — Safety boundaries

A tool-using agent can `rm -rf /` if you let it. Today you wire in the guardrails.

**Read (60 min):**
- [Anthropic — *Strengthen guardrails*](https://docs.claude.com/en/docs/test-and-evaluate/strengthen-guardrails/keep-claude-in-character)
- [Anthropic — *Mitigate jailbreaks*](https://docs.claude.com/en/docs/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks)
- [OpenAI Agents SDK — *Guardrails*](https://openai.github.io/openai-agents-python/guardrails/) (preview)
- [OWASP — *LLM06: Excessive Agency*](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/)

**Add to your agent:**
- **Path whitelist** for file tools (no `..`, no absolute paths outside `/workspace/`)
- **`write_file` confirmation** — log the diff; in interactive mode, ask the user "approve?"
- **Max iterations** + max total tokens
- **Loop detector**: if the same tool is called with the same args twice in a row → abort
- **Network egress block** — tools run in a sandboxed environment without DNS unless explicitly granted

---

### Day 5 — OpenAI Agents SDK

You've built it from scratch. Now use the production framework so you know both worlds.

**Read (60 min):**
- [OpenAI Agents SDK — *Welcome / Quickstart*](https://openai.github.io/openai-agents-python/)
- [OpenAI Agents SDK — *Tracing*](https://openai.github.io/openai-agents-python/tracing/) — built-in observability
- [OpenAI — *New tools for building agents*](https://openai.com/index/new-tools-for-building-agents/)

**Note on Swarm:** OpenAI's earlier *Swarm* is now superseded by the Agents SDK; treat Swarm as a reference design. Lexogrine and Morph both have good 2026 write-ups summarizing the transition.

**Hands-on (90 min):**
- Rebuild your Day-3 agent using the Agents SDK
- Use **handoffs** so one agent can delegate to another (e.g., "code generator" hands off to "tester")
- Compare LOC vs your hand-rolled version

---

### Day 6 — Build the weekly project (Part 1)

See **Weekly Project** below.

---

### Day 7 — Project + retro

Polish + write-up.

---

## Weekly Project — `code-helper-agent`

A safe, observable, single-agent code helper that reads a repo, makes a change, runs tests, and reports.

### Spec

```
$ code-helper "Add a CLI flag --json that makes the output JSON"
[plan]   I will:
         1. inspect cli.py
         2. add the --json flag
         3. update the printer
         4. run tests

[tool]   read_file(cli.py)               → 142 lines
[tool]   read_file(printer.py)           → 80 lines
[tool]   write_file(cli.py, ...)         → confirmed (diff shown)
[tool]   write_file(printer.py, ...)     → confirmed
[tool]   run_tests()                     → 12 passed, 0 failed

[done]   Done. PR-ready patch in /workspace/.staged
         Trace: trace_2026-05-15_142233.jsonl
```

### Requirements

- **Tools** (each with a clean schema + good description + thoughtful errors):
  - `read_file`, `list_dir`, `grep`, `write_file`, `run_tests`, `git_diff`
- **Safety boundaries:**
  - Path whitelist; no shell injection; max file size
  - `write_file` shows a diff; either auto-confirm in non-interactive mode (with audit log) or asks the user
- **Loop:** max 15 iterations, max 60k tokens; loop-detector aborts if the same call repeats
- **Trace:** every step → JSONL: `{ts, iter, tool, args, result, latency_ms, tokens}` saved to `traces/`
- **Built in both forms:** `agent_scratch.py` (hand-rolled) and `agent_sdk.py` (OpenAI Agents SDK). README compares.
- **Eval:** 10 hand-written tasks (e.g., "add this CLI flag", "fix this bug", "write a test for this function") with a success criterion. Report success rate, avg iterations, avg cost.

### Stretch

- Add an `evaluator-optimizer` pattern: a second agent grades the patch and the main agent revises if score < 0.7
- Add **plan mode**: before any `write_file`, the agent commits to a plan; the plan is logged and checked at the end
- Add **MCP** (Model Context Protocol) integration so external clients (Claude Desktop, IDEs) can drive your tools

---

## Curated resources

**Anthropic (your primary reading)**
- [*Building Effective AI Agents*](https://www.anthropic.com/research/building-effective-agents) — **must read**
- [*Writing effective tools for AI agents*](https://www.anthropic.com/engineering/writing-tools-for-agents)
- [*Tool use overview*](https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview)
- [*Implement tool use*](https://docs.claude.com/en/docs/agents-and-tools/tool-use/implement-tool-use)
- [Architecture Patterns PDF](https://resources.anthropic.com/hubfs/Building%20Effective%20AI%20Agents-%20Architecture%20Patterns%20and%20Implementation%20Frameworks.pdf)

**OpenAI**
- [Agents SDK docs](https://openai.github.io/openai-agents-python/)
- [Function calling guide](https://platform.openai.com/docs/guides/function-calling)
- [New tools for building agents (2025 release)](https://openai.com/index/new-tools-for-building-agents/)

**Other framework explainers**
- [Morph — *OpenAI Swarm: Lightweight Multi-Agent Orchestration Guide (2026)*](https://www.morphllm.com/openai-swarm)
- [Lexogrine — *OpenAI Swarm Framework in 2026*](https://lexogrine.com/blog/openai-swarm-multi-agent-framework-2026)
- [Maryam Miradi — *Build AI Agents: 40 Key Lessons from Anthropic's Masterclass*](https://www.maryammiradi.com/blog/build-ai-agents-anthropic-lessons)
- [Bharatkumar Kori — *Building Effective AI Agents: A Guide from Anthropic*](https://medium.com/accredian/building-effective-ai-agents-a-guide-from-anthropic-e66b533ff091)

**Lilian Weng (the broad survey)**
- [*LLM Powered Autonomous Agents*](https://lilianweng.github.io/posts/2023-06-23-agent/) — older but still the best survey of the conceptual landscape

**Safety**
- [Anthropic — *Mitigate jailbreaks*](https://docs.claude.com/en/docs/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks)
- [OWASP — *LLM06: Excessive Agency*](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/)
- [Simon Willison — *Prompt injection*](https://simonwillison.net/2023/Apr/14/worst-that-can-happen/)

---

## "Done when…" checklist

- [ ] I can describe each of the 5 agent design patterns and recall when to use which
- [ ] My hand-rolled agent loop has retries, validation, loop-detection, and max-iter caps
- [ ] I have at least 5 well-named tools with deliberate descriptions and good error messages
- [ ] I've rebuilt the same agent with the OpenAI Agents SDK and have an opinion on the trade-off
- [ ] All file-writes go through a confirmation/audit layer
- [ ] Every run produces a JSONL trace I can replay later
- [ ] My 10-task eval has a success rate I'm comfortable putting in a README
- [ ] I've written a retro

---

## Common pitfalls this week

1. **Starting with LangGraph or LangChain.** You'll learn nothing about the loop. Build it raw first.
2. **Tools that dump too much context.** A tool returning a 100k-token file blows the context window. Summarize / truncate intentionally.
3. **No max-iter cap.** Agents will happily infinite-loop. Cap it.
4. **No write-confirmation.** First time the agent goes off the rails and rewrites your project root, you'll wish you had this.
5. **Stochastic comparisons.** Run each eval task 3–5 times; agents are noisy.
6. **Calling it an "agent" when it's a workflow.** Anthropic's distinction is useful. Most "agents" in tutorials are workflows.

---

← Previous: [Week 12 — Caching, Routing, Production API Hardening](./WEEK-12.md) · → Next: [Week 14 — Agent Orchestration with LangGraph](./WEEK-14.md)
