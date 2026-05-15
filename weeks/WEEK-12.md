# Week 12 — Caching, Routing, Production API Hardening

> **Month 3 · Production LLM Systems**
> *"Cost, latency, and reliability are the three things that get an AI feature killed. This week is the toolbox for keeping it alive."*

---

## Why this week

You've got a fast inference server (Week 10) and a way to measure quality (Week 11). Now you wrap them in the gateway every real product needs: **caching**, **routing**, **rate limits**, **retries**, **fallbacks**, **PII filtering**, **logging**, **cost tracking**.

This is the boring stuff that decides whether your feature stays in production after the first week. It's also the work most heavily emphasized in senior AI-engineering job descriptions.

---

## Learning objectives

By Sunday night you should be able to:

1. Explain the three layers of LLM caching: **prompt/prefix cache**, **semantic cache**, **response cache**. When to use which.
2. Implement semantic caching with `GPTCache` (or a Redis-vector setup) and measure hit-rate
3. Implement a **model router**: easy tasks → cheap small model; hard tasks → expensive large model; with a fallback
4. Add the production cross-cutting concerns: structured logging, rate limits, retries with exponential backoff, circuit breakers, request validation, PII redaction
5. Track per-request cost + latency + tokens, and surface that in a dashboard
6. Compare your hand-rolled gateway against one OSS gateway (LiteLLM Proxy, Portkey, etc.)

---

## Day-by-day plan (≈ 2–3 hrs/day)

### Day 1 — The three caches

**Read (75 min):**
- [Akshay Ghalme — *How LLM Caching Actually Works — Prompt Cache, Semantic Cache & the CDN Patterns Nobody Documents (2026)*](https://akshayghalme.com/blogs/how-llm-caching-actually-works/) — the best overview
- [Spheron — *Semantic Caching for LLM Inference: GPTCache, Redis Vector Cache, and Prompt Cache Setup (2026)*](https://www.spheron.network/blog/semantic-cache-llm-inference-gpu-cloud/)
- [CallSphere — *LLM Caching Strategies for Cost Optimization: Prompt, Semantic, and KV Caching*](https://callsphere.ai/blog/llm-caching-strategies-cost-optimization-2026)

**The taxonomy:**

| Cache | Granularity | Match rule | Best for | Hit savings |
|---|---|---|---|---|
| **KV / Prefix cache** (server-side) | Tokens | Identical prefix | Shared system prompts, repeated context | 50–90% latency |
| **Semantic cache** (gateway-side) | Whole request | Embedding similarity > threshold | Common-question Q&A | ~100% (skip the LLM) |
| **Response cache** (gateway-side) | Whole request | Exact hash match | Idempotent calls, deterministic prompts | ~100% |

Also: **provider-side prompt caching** (Anthropic's `cache_control`, OpenAI's prompt caching) — opt-in, much cheaper for repeated prefixes. Use it.

---

### Day 2 — Build a semantic cache

**Read (45 min):**
- [GPTCache — *Quick start*](https://gptcache.readthedocs.io/en/latest/usage.html)
- [zilliztech/GPTCache (README)](https://github.com/zilliztech/GPTCache)
- [Bhavishya Pandit — *GPTCache: A practical guide*](https://bhavishyapandit9.substack.com/p/gptcache-a-practical-guide)

**Hands-on (90 min):**
- Set up GPTCache wrapping your OpenAI/vLLM client:
  ```python
  from gptcache import cache
  from gptcache.adapter import openai
  cache.init()  # default: in-memory + a small embedding model
  ```
- Or hand-roll one: SQLite + a sentence-transformer + cosine threshold of ~0.95
- Build a small benchmark: 200 queries, where ~30% are paraphrases of earlier queries
- Measure: hit rate, p50 latency on hit vs miss

---

### Day 3 — Provider-side prompt caching

This is the cheapest, easiest cache win in production. It's also one most teams forget.

**Read (60 min):**
- [Anthropic — *Prompt caching*](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)
- [OpenAI — *Prompt caching*](https://platform.openai.com/docs/guides/prompt-caching)
- [Introl — *Prompt Caching Infrastructure*](https://introl.com/blog/prompt-caching-infrastructure-llm-cost-latency-reduction-guide-2025)

**Hands-on (45 min):**
- Take a long system prompt (5k+ tokens) you might reuse — a system prompt with style rules, a docs corpus, a tool list
- Send 10 requests with and without `cache_control` on the system message (Claude) or with the prompt-caching prefix (OpenAI)
- Measure: per-call cost difference, latency difference

---

### Day 4 — Model routing & fallbacks

**Read (60 min):**
- [Martian — *Model Router patterns*](https://withmartian.com/) (browse their docs; their entire product is the router pattern)
- [RouteLLM project](https://github.com/lm-sys/RouteLLM)
- [LangChain — *Routing*](https://python.langchain.com/docs/how_to/routing/)

**Patterns to implement:**

1. **Classifier-first router**
   ```python
   def route(query: str) -> ModelPlan:
       category = classifier(query)  # tiny on-device or a cheap LLM
       if category == "simple_qa":     return small_local_model
       if category == "code_gen":      return mid_tier
       if category == "complex_reason": return frontier
       return small_local_model
   ```

2. **Cascade**
   - Try small model; if confidence (or a verifier) low → escalate to bigger

3. **Fallback**
   - Primary model → on error → secondary provider (different API, different region)
   - On timeout → fast smaller model as a graceful degradation

**Hands-on (90 min):** implement option 1 with at least 3 models (e.g., vLLM local + GPT-4o-mini + Claude-Sonnet) and a cascade fallback.

---

### Day 5 — Cross-cutting concerns

This is "make it not crash" day.

**Read (75 min):**
- [LiteLLM — *Proxy Server*](https://docs.litellm.ai/docs/simple_proxy) — read the features list; this is everything you should ship
- [Tenacity — *Retrying*](https://tenacity.readthedocs.io/) — the standard Python retry library
- [Anthropic — *Mitigating jailbreaks*](https://docs.claude.com/en/docs/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks)
- [Microsoft Presidio (PII redaction)](https://microsoft.github.io/presidio/) — solid open-source PII scrubber
- [OWASP — *LLM API Security*](https://genai.owasp.org/llmrisk/) — the broader threat model

**Hands-on (90 min): add to your gateway**
- **Rate limits** per API key (a token bucket; `slowapi` is fine)
- **Timeouts** on every upstream call
- **Retries with exponential backoff** + jitter, max 3
- **Circuit breaker** so one failing provider doesn't take everyone down
- **PII redaction** with Presidio before sending to third-party providers (and logging the redactions)
- **Structured request logs** (JSON) with `request_id`, `prompt_version`, `model`, `tokens_in/out`, `latency_ms`, `cost_usd`, `cache_hit`, `error_type`

---

### Day 6 — Cost & latency dashboards

**Read (30 min):**
- [Helicone — *Why we built an LLM observability tool*](https://www.helicone.ai/blog) (browse a few posts)
- [Langfuse — *Pricing & cost tracking*](https://langfuse.com/docs/integrations/llm-cost)
- [OpenAI — *Usage* (UI screenshots, copy the rough shape)](https://platform.openai.com/usage)

**Hands-on (~2 hr):**
- Emit Prometheus metrics from your gateway: `requests_total`, `latency_ms_histogram`, `tokens_in_total`, `tokens_out_total`, `cache_hit_total`, `cost_usd_total`
- Provision a Grafana dashboard with:
  - p50 / p95 / p99 latency per route
  - Cost per hour per model
  - Cache hit rate
  - Error rate by class

---

### Day 7 — Project + retro

Ship the weekly project (next section).

---

## Weekly Project — `ai-gateway`

A FastAPI-based AI gateway that wraps your vLLM server + at least one hosted provider, with all the cross-cutting concerns.

### Spec

Routes:
```
POST /v1/chat/completions     # OpenAI-compatible
GET  /v1/health
GET  /metrics                 # Prometheus
GET  /admin/cache/stats
POST /admin/cache/clear
```

### Required behaviors

- **Routing:** based on a request header (`X-Task-Type: simple_qa | code_gen | complex_reason`) or auto-classify
- **Semantic cache:** GPTCache or hand-rolled; configurable threshold; cache hit returns in < 50ms
- **Provider-side prompt cache:** opt-in via a header / config
- **Retries + circuit breaker:** misbehaving provider gets skipped for N seconds
- **PII redaction:** Presidio in the request path; redacted events go to an audit log
- **Rate limits** per API key (token bucket; configurable)
- **Structured logs**: JSON to stdout *and* SQLite (or Postgres) for queryability
- **Cost tracking:** per request, in USD, using current prices hardcoded with a comment
- **Prometheus** `/metrics` + Grafana dashboard JSON in the repo

### Deliverable

```
ai-gateway/
  app/
    main.py
    routing/
    cache/
    middleware/
    providers/         # adapters: vllm, openai, anthropic
  docker-compose.yml   # gateway + redis + prometheus + grafana
  grafana/
    dashboard.json
  tests/
  load_test/
  reports/
    cache_hit_rate.md
    cost_breakdown.md
    final_report.md
  README.md
```

### Stretch

- Wire up [**LiteLLM**](https://docs.litellm.ai/) as an alternative implementation; compare LOC + features
- Add **prompt versioning**: every prompt is registered with an id; calls log `prompt_version`
- Add **A/B routing**: 10% of traffic goes to model B; report quality and cost delta
- Add **shadow mode**: replay 10% of real traffic against a second model offline for comparison

---

## Curated resources

**Caching**
- [Akshay Ghalme — *How LLM Caching Actually Works (2026)*](https://akshayghalme.com/blogs/how-llm-caching-actually-works/)
- [Spheron — *Semantic Caching, GPTCache, Redis Vector Cache*](https://www.spheron.network/blog/semantic-cache-llm-inference-gpu-cloud/)
- [CallSphere — *LLM Caching Strategies for Cost Optimization*](https://callsphere.ai/blog/llm-caching-strategies-cost-optimization-2026)
- [GPTCache docs](https://gptcache.readthedocs.io/en/latest/)
- [zilliztech/GPTCache (GitHub)](https://github.com/zilliztech/GPTCache)
- [Anthropic — *Prompt caching*](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)
- [OpenAI — *Prompt caching*](https://platform.openai.com/docs/guides/prompt-caching)
- [Introl — *Prompt Caching Infrastructure*](https://introl.com/blog/prompt-caching-infrastructure-llm-cost-latency-reduction-guide-2025)

**Routing**
- [RouteLLM (LMSYS)](https://github.com/lm-sys/RouteLLM)
- [LangChain — *Routing*](https://python.langchain.com/docs/how_to/routing/)
- [DEV — *Top LLM Gateways That Support Semantic Caching in 2026*](https://dev.to/debmckinney/top-llm-gateways-that-support-semantic-caching-in-2026-3dho)
- [Martian — *Model Router patterns*](https://withmartian.com/)

**Production gateways (read, then maybe use)**
- [LiteLLM Proxy](https://docs.litellm.ai/docs/simple_proxy)
- [Portkey](https://portkey.ai/docs/welcome/introduction)
- [Kong AI Gateway](https://docs.konghq.com/hub/kong-inc/ai-proxy/)
- [Helicone](https://docs.helicone.ai/)

**Hardening**
- [Tenacity](https://tenacity.readthedocs.io/)
- [SlowAPI (rate limits for FastAPI)](https://slowapi.readthedocs.io/)
- [Microsoft Presidio (PII)](https://microsoft.github.io/presidio/)
- [OWASP — *LLM API Security*](https://genai.owasp.org/llmrisk/)
- [Anthropic — *Mitigating jailbreaks*](https://docs.claude.com/en/docs/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks)

---

## "Done when…" checklist

- [ ] I can explain the three caching layers and when each pays for itself
- [ ] My gateway exposes an OpenAI-compatible endpoint and is in Docker
- [ ] Semantic cache hit rate is measured on a real traffic mix
- [ ] Provider-side prompt caching is enabled and I have cost-savings numbers
- [ ] Routing decisions are visible in the logs (which model handled which request)
- [ ] Rate limit + retries + circuit breaker all have tests
- [ ] PII redaction is in the request path with an audit log
- [ ] Grafana dashboard shows latency, cost, hit rate
- [ ] I've written a retro

---

## Common pitfalls this week

1. **Semantic cache threshold too low** → wrong responses get served. Too high → useless. Tune empirically per workload.
2. **Caching personalized responses.** A cache key needs every input that materially changes the answer (user id, role, locale).
3. **Hand-rolling everything.** It's good to do it once. In real life, you'll probably run LiteLLM or Portkey. Know both worlds.
4. **Logging the prompt with PII.** Redact *before* logging. Once it's in your log store it's everywhere.
5. **Retry storms.** A provider blips; your gateway hammers it; the blip becomes an outage. Backoff + circuit breakers exist for this.
6. **Forgetting timeouts.** A hung request without a timeout pins a worker forever. Always set one.

---

## End-of-Month-3 checkpoint

You now have **12 GitHub repos** spanning data → training → inference → eval → production gateway. By any reasonable bar, this is a strong portfolio. Month 4 turns this into a coherent system.

---

← Previous: [Week 11 — Code Evaluation](./WEEK-11.md) · → Next: [Week 13 — Tool Calling + Agent Design Patterns](./WEEK-13.md)
