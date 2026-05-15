# 6-Month Roadmap: From Zero to AI Engineer for Code Models, Agents, and LLM Systems

This roadmap is designed for the role described:

- Own training-to-inference pipelines for large code models
- Optimize inference with quantization, distillation, and caching
- Fine-tune domain-specific LLMs for code generation, refactoring, debugging, and reasoning
- Design AI pipelines: prompting, retrieval, agents, evaluation
- Build agentic systems and multi-step tool calling
- Establish standards for benchmarking, evaluation, and observability
- Ship AI features with product, design, and engineering teams

## Important Reality Check

From absolute zero, 6 months will not make you a senior AI infrastructure engineer.

But 6 months can make you capable of building real systems, understanding the field, and creating a portfolio strong enough to pursue junior/mid AI engineering roles or transition toward this kind of role.

The goal is not to "learn AI generally."
The goal is to build production-style AI systems.

---

## Target Outcome After 6 Months

By the end of this roadmap, you should have built:

1. A fine-tuned code model using LoRA or QLoRA
2. A design-to-code or spec-to-code prototype
3. A RAG system over docs/codebases
4. A multi-step coding agent with tool use
5. A benchmark/evaluation harness for code generation
6. A production-style inference API with streaming, caching, quantization, logging, and observability
7. A capstone project that looks like a serious AI engineering system, not a toy notebook

---

## Recommended Stack

### Core
- Python
- TypeScript basics
- Git
- Linux shell
- Docker
- FastAPI
- Pytest

### Machine Learning
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Accelerate
- PEFT
- LoRA
- QLoRA
- TRL

### Inference
- vLLM
- llama.cpp
- Ollama
- Hugging Face Text Generation Inference

### Retrieval and RAG
- Embeddings
- Vector databases
- Chunking
- Reranking
- Context packing

### Agents
- LangGraph
- OpenAI Agents SDK
- Anthropic tool use concepts

### Evaluation
- HumanEval-style tests
- SWE-bench-style patch tests
- Ragas
- LLM-as-judge
- Unit-test-based evaluation

### Observability
- Structured logs
- Traces
- Prompt/version tracking
- Latency metrics
- Token/cost metrics

---

# Month 1: Software Engineering and ML Foundations

## Goal

Become comfortable with Python, APIs, data pipelines, and basic machine learning.

You need enough engineering foundation so that LLM systems do not feel like magic.

---

## Week 1: Python, Git, Linux, and Engineering Hygiene

### Learn

- Python functions
- Classes
- Type hints
- Exceptions
- Context managers
- File handling
- Git branches
- Commits
- Pull requests
- Basic rebasing
- Linux shell
- Environment variables
- Virtual environments
- Pytest

### Build: Project 1 — CLI Code Analyzer

Create a Python CLI that:

- Accepts a folder path
- Scans `.py` or `.ts` files
- Extracts functions and classes
- Counts lines of code
- Finds TODO comments
- Outputs JSON

### Deliverable

```text
code-analyzer/
  src/
  tests/
  README.md
  pyproject.toml
```

### Why This Matters

You are training the basic muscle required for AI coding systems:

- Reading code
- Processing files
- Producing structured output
- Testing your tools

---

## Week 2: Backend APIs and Streaming

### Learn

- FastAPI
- REST endpoints
- JSON schemas
- Server-Sent Events
- WebSockets
- Async Python basics
- Docker basics

### Build: Project 2 — Streaming Code Assistant API

Create an API:

```text
POST /analyze
POST /generate
GET /stream
```

For now, fake the model response. The important part is to stream tokens one by one.

### Deliverable

- Dockerized FastAPI app
- Streaming endpoint
- Basic structured logging
- Basic tests

### Why This Matters

The role explicitly mentions real-time, streaming AI experiences. You need to understand how streamed model output reaches a product UI.

---

## Week 3: Data Acquisition and Cleaning

### Learn

- Pandas
- JSONL
- Hugging Face Datasets
- Deduplication
- Train/validation/test splits
- Data leakage
- PII removal
- Secret detection
- License awareness for code datasets

### Build: Project 3 — Code Dataset Builder

Create a dataset from your own repositories or permissive sample repositories.

Input:

```text
repo/
```

Output:

```jsonl
{"instruction": "Refactor this function", "input": "...", "output": "..."}
{"instruction": "Explain this code", "input": "...", "output": "..."}
{"instruction": "Write tests for this function", "input": "...", "output": "..."}
```

### Deliverable

```text
data/raw/
data/processed/
scripts/build_dataset.py
dataset_card.md
```

### Why This Matters

The job requires:

- Data acquisition
- Data cleaning
- Data pipelines

Most AI projects fail because of bad data, not bad models.

---

## Week 4: ML Basics

### Learn

- Tensors
- Loss functions
- Gradient descent
- Train/validation/test split
- Overfitting
- Embeddings
- Classification vs generation
- GPU basics

### Build: Project 4 — Tiny Neural Network from Scratch

Train a small text classifier.

Possible tasks:

- Classify code snippets by programming language
- Classify GitHub issues as bug/refactor/docs/test
- Classify prompts by task type

Use PyTorch. Do not use an LLM yet.

### Deliverable

- Training script or notebook
- Loss curve
- Accuracy report
- README explaining the model

### Why This Matters

Do not skip this. People who skip ML basics become prompt hackers, not AI engineers.

---

# Month 2: NLP, Transformers, and LLM Fundamentals

## Goal

Understand how LLMs work well enough to fine-tune, evaluate, serve, and debug them.

---

## Week 5: NLP Basics

### Learn

- Tokenization
- Subword tokens
- Embeddings
- Attention
- Sequence-to-sequence models
- Decoder-only models
- Perplexity
- Context windows

### Build: Project 5 — Tokenizer Explorer

Create a small app or notebook that:

- Compares tokenization across code snippets
- Shows token counts
- Estimates cost and latency impact
- Demonstrates why whitespace and code formatting matter

### Deliverable

- Tokenization report for Python, JavaScript, HTML, CSS, and JSON snippets
- Small script that prints token counts for input files

---

## Week 6: Transformers and Hugging Face

### Learn

- transformers
- AutoTokenizer
- AutoModelForCausalLM
- Text generation parameters
- Temperature
- Top-p
- Max tokens
- Stop sequences
- Batching
- GPU memory basics

### Build: Project 6 — Local Code Completion

Run a small open-weight model locally. Start tiny if your hardware is limited.

Create functions:

```python
def complete_code(prompt: str) -> str:
    ...

def explain_code(code: str) -> str:
    ...

def generate_tests(code: str) -> str:
    ...
```

### Deliverable

- Python script
- Small benchmark set
- Latency measurement
- README with examples

---

## Week 7: Prompting and Structured Outputs

### Learn

- System/user/developer message separation
- Few-shot prompting
- JSON schema outputs
- Tool/function calling concepts
- Prompt injection basics
- Context engineering
- Output validation
- Retry logic

### Build: Project 7 — Structured Code Review Bot

Input: code diff

Output:

```json
{
  "summary": "...",
  "bugs": [],
  "security_risks": [],
  "performance_issues": [],
  "suggested_patch": "..."
}
```

### Deliverable

- Schema validation
- Failed-output retry logic
- 20 test examples
- Failure report

### Why This Matters

Production AI systems cannot just return random prose. They need reliable, parseable, structured outputs.

---

## Week 8: RAG Basics

### Learn

- Embeddings
- Chunking
- Vector search
- Reranking
- Context packing
- Citation and grounding
- Retrieval evaluation

### Build: Project 8 — Docs-Aware Code Assistant

Index documentation for:

- A small framework
- Your own fake design system
- A component library
- A small internal API

The assistant should answer:

- "Which component should I use?"
- "Generate a button using our design tokens."
- "Explain this API from the docs."

### Deliverable

- Ingestion script
- Vector search
- Answer generation
- Citations to source chunks
- Retrieval quality notes

### Why This Matters

Companies do not just ask a model to "write code." They retrieve:

- Docs
- Component libraries
- Examples
- Tickets
- Repository context
- Design system rules

---

# Month 3: Fine-Tuning Code Models and Building SLMs

## Goal

Train or fine-tune a small model for a narrow code/design task.

---

## Week 9: Fine-Tuning Concepts

### Learn

- Pretraining vs supervised fine-tuning
- Instruction tuning
- LoRA
- QLoRA
- Full fine-tuning
- Adapter tuning
- Trainable parameters
- Catastrophic forgetting
- Dataset quality
- Learning rate
- Batch size
- Gradient accumulation

### Build: Project 9 — Instruction Dataset v1

Create 1,000–5,000 examples for one narrow task.

Good choices:

- Design tokens to React component
- Figma-like JSON to Tailwind component
- Code comment to unit test
- Bug report to patch suggestion
- Old code to refactored code
- Component props to implementation

Example format:

```json
{
  "instruction": "Generate a React component using the design system.",
  "input": "...",
  "output": "...",
  "metadata": {
    "task": "design_to_code",
    "language": "typescript",
    "source": "synthetic/manual/repo"
  }
}
```

### Deliverable

- Dataset builder
- Dataset quality checklist
- Train/validation/test split
- Dataset card

---

## Week 10: First LoRA Fine-Tune

### Learn

- Hugging Face Trainer
- TRL SFTTrainer
- PEFT config
- LoRA rank
- LoRA alpha
- LoRA dropout
- Checkpointing
- Validation loss
- Model cards

### Build: Project 10 — Fine-Tune a Small Code Model

Use a small model your hardware supports. Fine-tune it on your dataset.

Track:

- Training loss
- Validation loss
- Examples before fine-tuning
- Examples after fine-tuning
- Failure cases

### Deliverable

- Reproducible training script
- Model card
- Before/after evaluation
- README explaining what improved and what did not

---

## Week 11: QLoRA and Memory-Efficient Training

### Learn

- 4-bit loading
- BitsAndBytes
- Adapter merging
- Inference with adapter
- Quantized training issues
- VRAM limits

### Build: Project 11 — LoRA vs QLoRA Comparison

Compare:

- Base model
- LoRA fine-tuned model
- QLoRA fine-tuned model

Measure:

- Output quality
- Latency
- GPU memory
- Disk size
- Failure modes

### Deliverable

```text
reports/lora_vs_qlora.md
```

### Why This Matters

The job mentions model optimization. You need to understand the tradeoff between quality, memory, speed, and cost.

---

## Week 12: Code Model Evaluation

### Learn

- pass@k
- Unit-test-based evaluation
- Why exact match is weak for code
- Compile/run-test evaluation
- HumanEval-style evaluation
- SWE-bench-style patch evaluation
- Sandboxed execution

### Build: Project 12 — Code Eval Harness

Create an evaluation system.

Input:

```json
{
  "prompt": "Write a function that...",
  "tests": "pytest tests..."
}
```

Output:

```json
{
  "passed": true,
  "latency_ms": 1240,
  "tokens_in": 500,
  "tokens_out": 200,
  "error": null
}
```

### Deliverable

- 50–100 test tasks
- Sandboxed execution
- Score report
- Failure analysis

### Why This Matters

This is one of the most important portfolio pieces. A real AI engineer does not judge model quality by vibes.

---

# Month 4: Inference Optimization and Production Serving

## Goal

Understand how to serve models cheaply, quickly, and reliably.

---

## Week 13: Inference Fundamentals

### Learn

- Prefill vs decode
- KV cache
- Batching
- Throughput vs latency
- Streaming
- Quantization
- GPU memory
- Context length tradeoffs
- Model serving APIs

### Build: Project 13 — Local Inference Server

Serve your fine-tuned model behind an API.

Features:

```text
POST /generate
GET /stream
GET /health
```

Also include:

- Request IDs
- Structured logs
- Timeout handling
- Max-token limits
- Error handling

### Deliverable

- FastAPI + vLLM or equivalent
- Docker Compose
- Latency report
- README

---

## Week 14: Quantization

### Learn

- FP16
- BF16
- INT8
- INT4
- GPTQ
- AWQ
- GGUF
- Quality vs speed vs memory
- When quantization hurts code generation

### Build: Project 14 — Quantization Benchmark

Run your model in at least two modes:

- Unquantized or half precision
- Quantized

Measure:

- Memory
- First-token latency
- Tokens per second
- Evaluation score
- Failure examples

### Deliverable

```text
reports/quantization_benchmark.md
```

---

## Week 15: Caching and Routing

### Learn

- Prompt cache
- Semantic cache
- Prefix cache
- Response cache
- Model routing
- Fallback models
- Cheap model vs expensive model routing
- Cache invalidation

### Build: Project 15 — Model Router

Create a router:

- Easy tasks use small local model
- Hard tasks use larger hosted model or simulated larger model
- Repeated context uses cached response
- Long-doc tasks use RAG first

Example:

```python
def route_request(task) -> ModelPlan:
    ...
```

Track:

- Cost estimate
- Latency
- Quality
- Cache hit rate

### Deliverable

- Routing logic
- Benchmark results
- Failure cases
- README explaining tradeoffs

---

## Week 16: Production API Hardening

### Learn

- Rate limits
- Retries
- Circuit breakers
- Request queues
- Concurrency
- Authentication
- PII filtering
- Secret filtering
- Logging without leaking sensitive prompts

### Build: Project 16 — Production-Style AI Gateway

Features:

- Request validation
- Prompt versioning
- Model versioning
- Cost tracking
- Latency metrics
- Error taxonomy
- Streaming responses
- Configurable model backends

### Deliverable

- API gateway repo
- Metrics dashboard or log viewer
- README with architecture diagram

### Why This Matters

This maps directly to "own systems end-to-end."

---

# Month 5: Agents, Tools, Retrieval, and Observability

## Goal

Build a multi-step AI coding agent that uses tools safely and is measurable.

---

## Week 17: Tool Calling

### Learn

- Tool schemas
- Function calling
- Argument validation
- Tool result compression
- Retry logic
- Unsafe tool boundaries
- Filesystem permissions
- Sandboxing

### Build: Project 17 — Tool-Using Code Assistant

Create tools:

```python
read_file(path)
search_code(query)
write_file(path, content)
run_tests()
lint()
git_diff()
```

The agent should:

1. Inspect files
2. Propose a change
3. Edit code
4. Run tests
5. Summarize result

### Deliverable

- Tool registry
- Mocked tools
- Real tools
- Trace logs
- Safety limits

---

## Week 18: Agent Orchestration

### Learn

- State machines
- Nodes
- Edges
- Checkpoints
- Human-in-the-loop
- Durable execution
- Streaming agent events
- Loop prevention

### Build: Project 18 — Multi-Step Coding Agent

Graph:

```text
User Request
  -> Classify request
  -> Retrieve context
  -> Plan
  -> Edit
  -> Test
  -> Reflect
  -> Final answer
```

### Deliverable

- Visible trace for every step
- Retry on failed tests
- Stop condition
- Human approval before file write
- README with diagrams

---

## Week 19: Agent Evaluation

### Learn

- Task success rate
- Tool-call accuracy
- Test pass rate
- Regression tests
- Hallucinated tool calls
- Invalid arguments
- Loop detection
- Cost per successful task

### Build: Project 19 — Agent Eval Suite

Create 30 tasks:

- Add a utility function
- Fix a failing test
- Refactor a component
- Update documentation
- Find a bug and explain it
- Generate component from design spec

Each task has:

```json
{
  "task": "...",
  "repo_snapshot": "...",
  "success_criteria": "...",
  "tests": "..."
}
```

Score:

- Solved/not solved
- Number of tool calls
- Tokens used
- Retries
- Time to solve
- Error type

### Deliverable

```text
evals/agent_eval_report.md
```

---

## Week 20: Observability

### Learn

- Traces
- Spans
- Prompt/version logging
- Latency buckets
- Token usage
- Tool failure monitoring
- Evaluation dashboards
- User feedback loops

### Build: Project 20 — AI Observability Layer

Log every request:

```json
{
  "request_id": "...",
  "prompt_version": "...",
  "model": "...",
  "tools_called": [],
  "latency_ms": 0,
  "tokens_in": 0,
  "tokens_out": 0,
  "eval_score": null,
  "error_type": null
}
```

### Deliverable

- Trace viewer or simple dashboard
- CSV, SQLite, or Postgres logs
- Failure analysis report
- README explaining observability strategy

### Why This Matters

This is how you stop building demos and start building systems.

---

# Month 6: Capstone — Design-to-Code Coding Agent

## Goal

Build one impressive end-to-end system matching the job description.

---

## Capstone Project

### Project Name

```text
Design-to-Code Agent for a Custom Design System
```

### Input Example

```json
{
  "screen": "Pricing page",
  "components": [
    {"type": "navbar", "variant": "default"},
    {"type": "pricing_card", "plan": "Pro"},
    {"type": "button", "variant": "primary"}
  ],
  "style": {
    "theme": "modern SaaS",
    "spacing": "comfortable"
  }
}
```

### Expected Output

The system should generate:

- React or Next.js component
- Tailwind or CSS module styling
- Design-token-compliant code
- Accessible markup
- Tests
- Revision based on user feedback

---

## Week 21: Capstone Architecture

Design the system:

```text
Frontend or CLI
  -> AI Gateway
  -> Router
  -> RAG over design docs/components
  -> Fine-tuned small model
  -> Agent orchestrator
  -> Tool layer
  -> Eval harness
  -> Observability
```

### Deliverable

- Architecture diagram
- Repo structure
- API design
- Eval plan

---

## Week 22: Build the Design/Code Knowledge Base

Create:

- Fake design system docs
- Component examples
- Design tokens
- Code templates
- Accessibility rules
- Bad examples
- Good examples

Build ingestion:

- Parse markdown/docs
- Chunk documents
- Embed chunks
- Retrieve relevant chunks
- Rerank if possible

### Deliverable

- Searchable design system RAG
- Retrieval quality report

---

## Week 23: Integrate Fine-Tuned Model

Use your Month 3 fine-tuned model.

The model should handle one narrow task:

- JSON design spec to component skeleton
- Component props to React code
- Design token mapping
- Raw HTML to design-system component

### Deliverable

- Base model vs fine-tuned model comparison
- Evaluation score
- Example outputs
- Failure cases

---

## Week 24: Build the Full Agent

Agent flow:

1. Parse user request
2. Retrieve relevant components/docs
3. Generate implementation plan
4. Generate code
5. Write files
6. Run tests/lint
7. Fix failures
8. Stream final answer
9. Log everything

### Deliverable

- Working demo
- Trace view
- Test pass/fail report
- README explaining the full system

---

## Week 25: Benchmark and Optimize

Measure:

- Generation quality
- Test pass rate
- Latency
- Tokens used
- Cost
- Cache hit rate
- Retrieval quality
- Agent tool failure rate

Create a report:

```text
benchmark_report.md
```

Include:

```text
Base model score:
Fine-tuned model score:
Agent score:
RAG-on score:
RAG-off score:
Quantized score:
Latency p50:
Latency p95:
Cost per successful task:
Top 10 failure modes:
```

---

## Week 26: Polish for Interviews

Create a serious README:

```markdown
# Design-to-Code Agent

## Problem
## Architecture
## Data Pipeline
## Fine-Tuning
## Inference Optimization
## Agent Design
## Evaluation
## Observability
## Results
## Failure Modes
## Future Work
```

Record a 3–5 minute demo.

Prepare to explain:

- Why you fine-tuned instead of only prompting
- Why you used RAG
- What quantization changed
- How you measured quality
- How your agent avoids infinite loops
- How you handle bad tool calls
- How you would scale it
- What failed

The failure analysis is important. Strong engineers do not pretend everything worked.

---

# Weekly Time Commitment

## Minimum

```text
10 hours/week
```

This is enough for a shallow-to-medium version.

## Strong

```text
20 hours/week
```

This is enough to build a serious portfolio.

## Aggressive

```text
30+ hours/week
```

This can make you genuinely competitive if you already have some software engineering background.

---

# Recommended Weekly Rhythm

```text
40% building
25% reading/docs
20% debugging/evaluation
10% writing notes
 5% sharing/publishing
```

Do not spend 80% of your time watching courses. That feels productive, but it usually avoids the real work.

---

# Core Resources

## Foundations

- Python docs
- PyTorch tutorials
- FastAPI docs
- Docker docs
- Hugging Face course

## LLM Fine-Tuning

- Hugging Face Transformers training docs
- Hugging Face PEFT docs
- PEFT LoRA docs
- PEFT quantization docs
- TRL docs

## Inference

- vLLM docs
- llama.cpp
- Ollama
- Hugging Face Text Generation Inference

## Agents

- LangGraph docs
- OpenAI Agents docs
- Anthropic tool-use/context-engineering docs

## Evaluation

- HumanEval
- SWE-bench
- Ragas
- OpenAI evals/agent evals docs

---

# What to Ignore for Now

Do not start with:

- Training a model from scratch
- CUDA kernel programming
- Distributed training
- RLHF
- Complex multi-agent frameworks
- 50 different vector databases
- Reading every transformer paper
- Obsessing over the newest model leaderboard

Those are useful later.

For this job profile, the highest ROI is:

```text
software engineering + data pipelines + fine-tuning + serving + agents + evals
```

---

# Skill Map

By the end, you should be able to say the following.

### Software Engineering
I can build and ship a backend service around an LLM, with streaming, tests, Docker, logs, and API boundaries.

### Data
I can collect, clean, format, split, and validate code/design datasets for fine-tuning and evaluation.

### LLMs
I understand tokenization, context windows, decoding, fine-tuning, LoRA, QLoRA, quantization, and inference tradeoffs.

### Code Models
I can fine-tune and evaluate a small code model for a narrow task.

### Agents
I can build an agent that calls tools, edits files, runs tests, recovers from failure, and produces traceable outputs.

### Evaluation
I do not judge models by vibes. I use task datasets, unit tests, pass rates, latency, cost, and failure analysis.

### Production
I can reason about cost, latency, caching, model routing, observability, and fallback behavior.

---

# Final Portfolio Layout

Create one GitHub organization or pinned repo set:

```text
01-code-analyzer
02-streaming-ai-api
03-code-dataset-builder
04-docs-rag-assistant
05-lora-code-finetune
06-code-eval-harness
07-vllm-inference-server
08-agentic-code-editor
09-design-to-code-capstone
```

Or combine the best pieces into one polished capstone:

```text
design-to-code-agent/
  app/
  data/
  training/
  inference/
  agent/
  evals/
  observability/
  docs/
  reports/
```

The second option is stronger for interviews.

---

# Final Rule

Every month must produce a working system.
Every system must have tests.
Every model change must have an eval.
Every eval must produce a report.
Every report must include failure cases.

That mindset is what separates someone who can call Claude from someone who can build a production AI product.
