# Week 14 — Agent Orchestration with LangGraph

> **Month 4 · Agents, Evaluation, Capstone**
> *"You wrote the loop by hand. Now you write the graph — and the graph survives 50 turns, restarts, parallel branches, and human-in-the-loop checks."*

---

## Why this week

Last week's hand-rolled loop is fine for a single tool-using agent. It falls apart the moment you need:

- **Branching** (if classifier says X go here, else there)
- **Parallel fan-out / fan-in** (run 4 retrievers, merge)
- **Checkpoint & resume** (the agent died at step 8; resume from step 7)
- **Human-in-the-loop** (pause for approval before a destructive action)
- **Persistent memory** across long conversations or runs

[**LangGraph**](https://langchain-ai.github.io/langgraph/) — the explicit-graph successor to LangChain agents — is the most widely adopted framework for this in 2026 (used in production at Uber, JP Morgan, Klarna, etc.). It's not the only option (LlamaIndex Workflows, OpenAI Agents SDK with handoffs, CrewAI for multi-agent role-play, AutoGen for academic-style multi-agent), but it's the right default.

---

## Learning objectives

By Sunday night you should be able to:

1. Model an agent as a **state machine of nodes and edges** in LangGraph
2. Build the canonical *plan → execute → reflect → respond* coding-agent graph
3. Add **conditional edges** (routing) and **parallel branches** (map-reduce)
4. Use **checkpointers** so the agent can be paused and resumed
5. Insert a **human-in-the-loop** approval node before destructive actions
6. Stream graph state to a UI / CLI so the user sees progress in real time
7. Recognize when *not* to use LangGraph (a single API call doesn't need a graph)

---

## Day-by-day plan (≈ 2–3 hrs/day)

### Day 1 — Concepts: state, nodes, edges

**Read (75 min):**
- [LangGraph — *Conceptual guide*](https://langchain-ai.github.io/langgraph/concepts/) — read the whole top-level page
- [LangGraph — *Why LangGraph?*](https://langchain-ai.github.io/langgraph/concepts/high_level/)
- [LangGraph — *Low-Level Concepts*](https://langchain-ai.github.io/langgraph/concepts/low_level/) — focus on `StateGraph`, nodes, edges, state schemas

**Reflect:** draw your Week-13 agent loop as a graph. Identify each node (LLM call, tool call) and each edge (always-go-here vs conditional).

---

### Day 2 — Quickstart + a real graph

**Read + hands-on (~2.5 hr):**
- [LangGraph — *Quickstart*](https://langchain-ai.github.io/langgraph/tutorials/introduction/) — work through it
- Convert your Week-13 hand-rolled agent into a `StateGraph` with these nodes:
  ```
  classify  →  plan  →  execute (with tools)  →  reflect  →  respond
                ↑__________________________________________|
                            (only loop if reflect says "not done")
  ```
- Stream node events with `app.stream(input, stream_mode="values")` and print them

---

### Day 3 — Routing, parallelism, and structured state

**Read (60 min):**
- [LangGraph — *Conditional edges*](https://langchain-ai.github.io/langgraph/how-tos/branching/)
- [LangGraph — *Subgraphs*](https://langchain-ai.github.io/langgraph/how-tos/subgraphs/) — reusable graph fragments
- [LangGraph — *Map-Reduce branches*](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/) — parallel fan-out
- [LangChain blog — *Plan-and-Execute agents*](https://blog.langchain.dev/planning-agents/) (excellent pattern)

**Hands-on (60 min):**
- Add a `route` node that classifies the request into `simple`, `code_edit`, `multi_file_refactor`
- Send each to a different sub-graph
- Add a parallel fan-out somewhere meaningful (e.g., "search 3 retrievers in parallel, merge results")

---

### Day 4 — Checkpoints, memory, durable execution

The killer feature: your agent can crash, restart, and pick up where it left off.

**Read (60 min):**
- [LangGraph — *Persistence*](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [LangGraph — *Memory*](https://langchain-ai.github.io/langgraph/concepts/memory/)
- [LangGraph — *How-to: add persistence*](https://langchain-ai.github.io/langgraph/how-tos/persistence/)

**Hands-on (60 min):**
- Add a `SqliteSaver` (or `MemorySaver` for dev) so every node's state is checkpointed
- Run an agent. Kill it mid-run. Resume from the same thread_id and watch it pick up.
- Add long-term memory (e.g., what the user prefers in past sessions) using the cross-thread memory API

---

### Day 5 — Human-in-the-loop

The most underrated feature for production agents.

**Read (60 min):**
- [LangGraph — *Human-in-the-loop*](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — concepts
- [LangGraph — *How-to: HIL*](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/) — interrupts + breakpoints
- [LangChain blog — *Interrupt-and-Resume*](https://blog.langchain.dev/agent-protocol-interoperability-for-llm-agents/) (skim)

**Hands-on (60 min):**
- Add a breakpoint before every `write_file` action
- When breakpoint hits: pause graph, present diff to the user, on approval continue, on rejection feed rejection reason into state and continue
- This is the foundation for an "Auto / Manual" toggle in a real product

---

### Day 6 — Build the weekly project (Part 1)

See **Weekly Project** below.

---

### Day 7 — Project + retro

Polish + write-up.

---

## Weekly Project — `coding-agent-graph`

A LangGraph multi-step coding agent that:
- Classifies the request
- Retrieves relevant context (RAG from Week 6 — yes, integrate it!)
- Plans
- Executes tools (read/write/test, from Week 13)
- Reflects on test failures and retries
- Pauses for human approval before file writes
- Streams live state to the CLI

### Spec

```
$ coding-agent "Refactor src/utils.py to be type-checked end-to-end"

[classify]       multi_file_refactor
[retrieve]       loaded 6 chunks from project docs
[plan]
   1. Inspect src/utils.py
   2. Run mypy to find current errors
   3. Add type annotations
   4. Run mypy + tests
   5. If failures, reflect and retry (max 2)
[execute] read_file src/utils.py        → 142 lines
[execute] run_tool   mypy src/utils.py  → 17 errors
[execute] llm_edit   src/utils.py       → diff (await approval)
[hil]     Approve diff? [y/N]: y
[execute] write_file src/utils.py       → ok
[execute] run_tool   mypy src/utils.py  → 0 errors
[execute] run_tool   pytest             → 23 passed
[reflect] Done.
[done]    trace_id: thr-abc123
```

### Requirements

- LangGraph `StateGraph` with at least 6 named nodes
- At least 1 conditional edge (router) and 1 parallel branch
- `SqliteSaver` checkpointer; runs are resumable
- HIL breakpoint before any `write_file`
- Streamed node events to stdout (and to a JSONL trace file)
- RAG retrieval node uses your Week-6 advanced-RAG pipeline against a local docs corpus
- Eval against 10 hand-written tasks; report success rate, avg nodes touched, avg cost
- README compares LangGraph vs your Week-13 hand-rolled agent on the same tasks

### Stretch

- Add an evaluator-optimizer loop: a `judge` node grades the patch; if score < threshold, loop back to `plan`
- Add a [LangGraph Studio](https://langchain-ai.github.io/langgraph/cloud/) launch config and screenshot of the visualizer
- Add MCP server exposure so your agent can be driven from Claude Desktop / Cursor / VS Code

---

## Curated resources

**Official docs (your primary reference)**
- [LangGraph — Home](https://langchain-ai.github.io/langgraph/)
- [LangGraph — *Quickstart*](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [LangGraph — *Conceptual guide*](https://langchain-ai.github.io/langgraph/concepts/)
- [LangGraph — *Low-Level Concepts*](https://langchain-ai.github.io/langgraph/concepts/low_level/)
- [LangGraph — *Persistence / Checkpointers*](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [LangGraph — *Human-in-the-loop*](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [LangGraph — *Memory*](https://langchain-ai.github.io/langgraph/concepts/memory/)
- [LangGraph — *Map-Reduce*](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)

**Tutorials (2026)**
- [PyCharm Blog — *LangChain Python Tutorial: A Complete Guide for 2026*](https://blog.jetbrains.com/pycharm/2026/02/langchain-tutorial-2026/)
- [DEV — *LangGraph 2.0: The Definitive Guide to Production-Grade AI Agents (2026)*](https://dev.to/richard_dillon_b9c238186e/langgraph-20-the-definitive-guide-to-building-production-grade-ai-agents-in-2026-4j2b)
- [DEV — *Building Production AI Agents with LangGraph: Beyond the Toy Examples (2026)*](https://dev.to/young_gao/building-production-ai-agents-with-langgraph-beyond-the-toy-examples-2idm)
- [tech-insider.org — *Build an AI Agent with LangGraph Python in 14 Steps (2026)*](https://tech-insider.org/langgraph-tutorial-ai-agent-python-2026/)
- [Alphabold — *LangGraph Agents in Production: Architecture, Costs & Real-World Outcomes*](https://www.alphabold.com/langgraph-agents-in-production/)

**Pattern essays**
- [LangChain blog — *Plan-and-Execute agents*](https://blog.langchain.dev/planning-agents/)
- [Anthropic — *Building Effective Agents* (revisit)](https://www.anthropic.com/research/building-effective-agents)

**Other frameworks (know they exist)**
- [LlamaIndex — *Workflows*](https://docs.llamaindex.ai/en/stable/module_guides/workflow/) — light, event-driven
- [CrewAI](https://docs.crewai.com/) — multi-agent role-play
- [AutoGen](https://microsoft.github.io/autogen/) — Microsoft, conversational multi-agent
- [OpenAI Agents SDK — *Handoffs*](https://openai.github.io/openai-agents-python/handoffs/) — if you prefer the OAI ecosystem
- [GuruSup — *Best Multi-Agent Frameworks in 2026: LangGraph, CrewAI...*](https://gurusup.com/blog/best-multi-agent-frameworks-2026)

---

## "Done when…" checklist

- [ ] My agent is a LangGraph `StateGraph` I can render as an image and explain node by node
- [ ] At least one conditional edge + one parallel branch are in the graph
- [ ] Checkpointer is wired in; I have demonstrated resume-after-crash
- [ ] HIL works: I can approve / reject a file write
- [ ] The graph streams events to stdout / JSONL
- [ ] My RAG (Week 6) is integrated as a retrieval node
- [ ] 10-task eval is on file with a success rate and a failure mode list
- [ ] I have an honest README section: *"When LangGraph is overkill"*
- [ ] I've written a retro

---

## Common pitfalls this week

1. **Over-engineering the state.** Keep `State` minimal — what *must* persist between nodes. Don't dump everything into it.
2. **Using LangChain abstractions you don't need.** LangGraph stands alone; you don't need LangChain LCEL for everything.
3. **Forgetting that streaming != progress bar.** Stream nodes; emit human-friendly status from each node.
4. **HIL that always says yes.** A "press enter to approve" without a diff is pointless. Show the diff. Force a real decision.
5. **No graph visualization.** `app.get_graph().draw_mermaid()` is free and clarifies everything. Save the image into your repo.

---

← Previous: [Week 13 — Tool Calling + Agent Design Patterns](./WEEK-13.md) · → Next: [Week 15 — Agent Evaluation + Observability](./WEEK-15.md)
