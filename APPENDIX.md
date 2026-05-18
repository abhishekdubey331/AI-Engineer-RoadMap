# Appendix — Beyond the 16 weeks

> Two things the 16-week path doesn't cover in depth, plus the reading method that ties the citations together.

The 4-month roadmap is scoped to **LLM systems engineering**: tokenization → transformers → fine-tuning → RAG → serving → agents → eval. Foundations that load-bear under that path (basic math, PyTorch fundamentals) live in [`PREREQUISITES.md`](./PREREQUISITES.md) — read those *before* Week 1.

This appendix covers the two adjacent areas that the weekly content deliberately stays out of:

- **MLOps** — the classical ML-systems discipline (data versioning, experiment tracking, model registries, drift, deployment patterns). The roadmap already does a lot of what's now called *LLMOps* — serving in W10, the gateway in W12, evals in W11 and W15. Full MLOps is another month of material; this section curates that path.
- **How to read AI papers** — almost every week cites papers. The *method* for reading them is what turns those citations into a usable mental model. The roadmap implies this; the appendix makes it explicit.

A third short section at the end is the north-star sentence to hold yourself to after the 16 weeks.

---

## A) MLOps extensions

The roadmap covers what could fairly be called **LLMOps**: serving (W10), gateway and caching (W12), code evaluation (W11), agent evaluation + observability (W15), Hugging Face Hub publishing (W7/W8). What it does *not* cover, on purpose:

| Topic | Why it's out of scope | When you'll need it |
|---|---|---|
| **Dataset versioning** (DVC, lakeFS) | Adds weight; HF Hub + git is enough for the roadmap's project scale | The moment you have >1 person committing to a dataset, or you're regenerating eval sets and comparing across versions |
| **Experiment tracking** at scale (W&B, MLflow, Comet) | Briefly touched in W11 stretch goals (`Export to W&B or MLflow`) | You're training >5 fine-tunes a week and your `outputs/` folder becomes unreadable |
| **Model registry & promotion** (W&B Models, MLflow Model Registry) | HF Hub doubles as registry for the roadmap's purposes | Multi-environment promotion (dev → staging → prod) with approval gates |
| **Pipeline orchestration** (Airflow, Prefect, Dagster, Kubeflow Pipelines, ZenML) | Out of scope; each is a separate week | You're running scheduled batch jobs (nightly evals, ingestion, retraining) |
| **CI for evals / model gates** | Lightly covered (W15 regression suite) | You want a PR to fail if pass-rate drops by ≥X% |
| **Production monitoring & drift** | W12 + W15 cover the observability piece; *drift over time* is not in scope | Your model has been in production for 2+ months and quality is decaying |
| **Deployment patterns** (blue/green, canary, shadow) | Out of scope | You can't take downtime on a model swap |
| **Cost & FinOps for LLMs** | Touched in W12 cost tracking; broader FinOps is its own discipline | Your monthly LLM bill crosses $5k and someone asks why |

### Recommended path

Pick **one** of these as a "Month 5" or as a job-relevant extension after the roadmap:

- 🥇 **[Made With ML — *MLOps course*](https://madewithml.com/courses/mlops/)** (Goku Mohandas, free, code-first). The most-cited free MLOps curriculum. Covers data versioning, experiment tracking, orchestration, CI/CD, monitoring. Has a [Versioning Code, Data and Models](https://madewithml.com/courses/mlops/versioning/) module that pairs directly with our W7 dataset prep.
- 🥈 **[DataTalksClub MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)** (free, 6 modules + a portfolio project). Production-flavored: Docker, MLflow, Prefect/Kestra, monitoring with Grafana/Evidently. Newer cohorts run yearly; the repo is the canonical material.
- 📖 **[Chip Huyen — *Designing Machine Learning Systems*](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)** (book; companion site at [huyenchip.com/mlops/](https://huyenchip.com/mlops/)). The reference text for ML-systems thinking. Read alongside any of the courses above.
- 🧰 **[Chip Huyen — *AI Engineering*](https://www.oreilly.com/library/view/ai-engineering/9781098166298/)** (book, 2025). The newer follow-up specifically on the LLM era — pairs very well with this roadmap.

### Concrete pickups (do these even without a full course)

If you want the minimal-cost MLOps additions to plug into the roadmap's existing projects:

1. **Add DVC to your Week-7 dataset project.** Track `data/processed/{train,val,test}.jsonl` with [DVC](https://doc.dvc.org/use-cases/versioning-data-and-models/tutorial) so future commits can diff dataset versions. ~20 lines of config.
2. **Track Week-8 fine-tune runs with MLflow or W&B.** Two extra `mlflow.log_metric()` calls per training loop. You'll thank yourself when you've run 12 LoRA variants.
3. **Promote your Week-8 adapter via a tagged HF Hub release** instead of overwriting the same repo. Cheap, immediate model-registry behavior.
4. **Add a CI eval gate** to your Week-11 harness: a GitHub Action that runs the eval on a PR and blocks merge if pass-rate drops by ≥5%. This is the single highest-signal MLOps addition for a portfolio.
5. **Wire one drift signal** into your Week-12 gateway: log per-day pass-rate against a held-out canary task set; alert when it slips.

---

## B) How to read AI papers (and why you should)

Every week of this roadmap cites at least one paper. By Week 8 you're expected to skim QLoRA; by Week 14 you've encountered the vLLM, ReAct (implicitly), and SWE-bench papers. **A reading method matters more than a reading list.**

The canonical method is **Keshav's three-pass approach**:

📄 [S. Keshav — *How to Read a Paper* (PDF, free)](https://www.cs.tufts.edu/~nr/cs257/archive/keshav/paper-reading.pdf) — 2 pages, the gold standard.

### The three passes (condensed)

**🥇 First pass — 5 to 10 minutes — "Should I read this?"**
- Read the **title**, **abstract**, **intro**
- Read the **headings** of every section; ignore the body
- Read the **conclusion**
- Glance at **references**, mentally tick the ones you've already read

*After pass 1 you should be able to answer:* (a) what type of paper is this? (b) what's the context? (c) what is the basic contribution? (d) is it well written? *If the answer to "do I care?" is no, stop here.*

**🥈 Second pass — ~1 hour — "Grasp the content"**
- Read with attention; **skip math/proofs** on this pass
- **Read every figure and table carefully** — figures often carry the paper's main idea
- Mark relevant unread references for follow-up
- Write a short summary in your own words

*After pass 2 you can summarize the paper to a colleague without notes. Most engineering papers only need a pass-2 read.*

**🥉 Third pass — 4–5 hours (or ~1 hour if you're experienced) — "Re-implement in your head"**
- Read every section, including the math
- Identify and challenge **every assumption**
- Reproduce the paper's claims mentally: would your version of the experiment work?

*After pass 3 you should be able to reconstruct the paper's structure from memory and identify its weaknesses.*

### LLM-augmented reading (Karpathy's 2025 method)

Andrej Karpathy popularized using LLMs as a reading partner: paste the paper, ask for a section-by-section explanation, ask "what assumption is the weakest?", ask for code that reproduces a small claim. His tool [reader3](https://github.com/karpathy/reader3) is purpose-built for it. The right shape is: do pass 1 yourself, then use the LLM as a sparring partner on pass 2/3.

> ⚠️ **Calibration check:** if you've finished a "reading" and you can't summarize the paper to a friend at a whiteboard, the LLM did the reading, not you. Cycle back.

### The papers worth reading (mapped to roadmap weeks)

If you do nothing else, do at least one **full pass-2 read** of each paper below — they're the load-bearing references of modern LLM engineering.

| Paper | Why | Week |
|---|---|---|
| [Vaswani et al. — *Attention Is All You Need*](https://arxiv.org/abs/1706.03762) | The transformer | W2 |
| [Radford et al. — *GPT-2*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Decoder-only LM at scale | W2 |
| [Sennrich et al. — *BPE for NMT*](https://arxiv.org/abs/1508.07909) | The tokenizer algorithm | W1 |
| [Hu et al. — *LoRA*](https://arxiv.org/abs/2106.09685) | Low-rank adaptation | W8 |
| [Dettmers et al. — *QLoRA*](https://arxiv.org/abs/2305.14314) | 4-bit fine-tuning | W9 |
| [Frantar et al. — *GPTQ*](https://arxiv.org/abs/2210.17323) | Post-training quantization | W9 |
| [Lin et al. — *AWQ*](https://arxiv.org/abs/2306.00978) | Activation-aware quantization | W9 |
| [Dao — *FlashAttention-3*](https://tridao.me/blog/2024/flash3/) (blog + paper) | The kernel that makes attention fast | W2/W3 |
| [Kwon et al. — *vLLM / PagedAttention*](https://arxiv.org/abs/2309.06180) | Modern KV-cache management | W10 |
| [Leviathan et al. — *Speculative Decoding*](https://arxiv.org/abs/2211.17192) | Faster inference | W10 |
| [Lewis et al. — *RAG*](https://arxiv.org/abs/2005.11401) | The original RAG paper | W5 |
| [Gao et al. — *HyDE*](https://arxiv.org/abs/2212.10496) | Query rewriting for retrieval | W6 |
| [Anthropic — *Contextual Retrieval* (blog)](https://www.anthropic.com/news/contextual-retrieval) | Modern RAG upgrade | W6 |
| [Yao et al. — *ReAct*](https://arxiv.org/abs/2210.03629) | Reasoning + acting; the agent loop | W13 |
| [Schick et al. — *Toolformer*](https://arxiv.org/abs/2302.04761) | Self-taught tool use (historical foundation) | W13 |
| [Chen et al. — *HumanEval*](https://arxiv.org/abs/2107.03374) | The pass@k benchmark | W11 |
| [Jimenez et al. — *SWE-bench*](https://arxiv.org/abs/2310.06770) | Repo-level coding eval | W11/W16 |
| [Yang et al. — *SWE-agent*](https://arxiv.org/abs/2405.15793) | Reference architecture for coding agents | W16 |

### Building the habit

- **One paper a week**, full pass-2, paired with the week's project. Not more — depth > breadth.
- Keep a `papers.md` in a notes repo. After each paper: 3–5 bullet "this changed how I'd build X" notes. After a year you have a meaningful artifact.
- For the deep-dives (W2 transformer, W8 LoRA, W10 vLLM, W16 SWE-agent), do a pass-3 read and try to reimplement one tiny claim.

### Further reading on reading

- [Three-pass overview at Researcher Connect (HKU)](https://blog-sc.hku.hk/reading-papers-efficiently-with-the-three-pass-approach/)
- [Towards Data Science — *How to Read Machine Learning Papers Easily*](https://towardsdatascience.com/how-to-read-machine-learning-papers-easily-2555deb78d80/)
- [Andrej Karpathy's *LLM Paper Reading List* (curated by community)](https://towardsai.net/p/data-science/andrej-karpathy-llm-paper-reading-list-for-llm-mastery) — the broader canon if you want to keep going beyond the roadmap's required papers

---

## C) North-star reminder

> *"I can build an LLM system from tokenizer-level understanding to production deployment and evaluation."*

If after this roadmap + a couple of the extensions above (one MLOps course + a steady paper-reading habit) you can stand behind that sentence without flinching, you are no longer a "shallow AI engineer" — you're the kind the field is genuinely short on.

Keep going.
