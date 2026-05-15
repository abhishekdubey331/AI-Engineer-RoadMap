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
- [Helicone — *Effective LLM Caching*](https://www.helicone.ai/blog/effective-llm-caching) — the most thorough vendor-engineering overview
- [AWS Database Blog — *Optimize LLM response costs and latency with effective caching*](https://aws.amazon.com/blogs/database/optimize-llm-response-costs-and-latency-with-effective-caching/) — vendor-neutral architectural patterns
- [Redis — *Vector database*](https://redis.io/docs/latest/develop/get-started/vector-database/) — the cache backend you'll likely use in production

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
- [GPTCache — *Quick start*](https://gptcache.readthedocs.io/en/latest/usage.html) — **note:** GPTCache has had near-zero recent commits and is effectively maintenance-mode. Still teaches the right pattern; for production prefer LangChain's `RedisSemanticCache`, [LiteLLM's built-in cache](https://docs.litellm.ai/docs/caching), or Portkey.
- [zilliztech/GPTCache (README)](https://github.com/zilliztech/GPTCache)
- [GPT Semantic Cache paper](https://arxiv.org/abs/2411.05276) — the design rationale

**Hands-on (90 min):**
- Hand-roll a semantic cache: SQLite (or Redis) + a sentence-transformer + cosine threshold ~0.95 — this is what production teams actually deploy
- Or use LiteLLM Proxy's built-in cache (one config line)
- Build a small benchmark: 200 queries, where ~30% are paraphrases of earlier queries
- Measure: hit rate, p50 latency on hit vs miss
- **Try an adversarial paraphrase set** — engineer 10 queries that *look* paraphrased but should return different answers ("how do I delete a user?" vs "how do I delete *the* user?"). See what your threshold does. This failure mode is real.

---

### Day 3 — Provider-side prompt caching

This is the cheapest, easiest cache win in production. It's also one most teams forget.

**Read (60 min):**
- [Anthropic — *Prompt caching*](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) — note the 2025 addition of a **1-hour cache TTL tier** alongside the original 5-minute one; the cost math has shifted
- [OpenAI — *Prompt caching*](https://platform.openai.com/docs/guides/prompt-caching)
- [Anthropic cookbook — *Prompt caching examples*](https://github.com/anthropics/anthropic-cookbook/tree/main/misc) — concrete patterns

**Hands-on (45 min):**
- Take a long system prompt (5k+ tokens) you might reuse — a system prompt with style rules, a docs corpus, a tool list
- Send 10 requests with and without `cache_control` on the system message (Claude) or with the prompt-caching prefix (OpenAI)
- Measure: per-call cost difference, latency difference

---

### Day 4 — Model routing & fallbacks

**Read (60 min):**
- [LiteLLM — *Routing*](https://docs.litellm.ai/docs/routing) — the de facto OSS routing implementation in 2026; this is what most teams actually ship
- [Not-Diamond — *awesome-ai-model-routing*](https://github.com/Not-Diamond/awesome-ai-model-routing) — curated, community-maintained list
- [NotDiamond](https://www.notdiamond.ai/) — hosted routing-as-a-product
- [Martian Router](https://route.withmartian.com/) — another hosted router
- [RouteLLM (LMSYS)](https://github.com/lm-sys/RouteLLM) — reference research project (note: last update Aug 2024)
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

**Read (90 min):**
- [LiteLLM — *Proxy Server*](https://docs.litellm.ai/docs/simple_proxy) — **the** OSS gateway most production teams run in 2026. Read the features list as the reference implementation your hand-rolled gateway competes with.
- [LiteLLM — *Caching*](https://docs.litellm.ai/docs/caching), [*Routing*](https://docs.litellm.ai/docs/routing), [*Fallbacks*](https://docs.litellm.ai/docs/proxy/reliability)
- [Tenacity — *Retrying*](https://tenacity.readthedocs.io/) — the standard Python retry library
- [Anthropic — *Mitigating jailbreaks*](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks)
- [Microsoft Presidio (PII redaction)](https://microsoft.github.io/presidio/) — solid open-source PII scrubber
- [OWASP — *GenAI Top 10 (2025)*](https://genai.owasp.org/llm-top-10/) — the current canonical threat model
- [Meta Llama Guard / Purple-Llama](https://github.com/meta-llama/PurpleLlama) — input/output safety classifiers
- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) — programmable guardrails framework

**Hands-on (90 min): add to your gateway**
- **Rate limits** per API key (a token bucket; `slowapi` is fine)
- **Timeouts** on every upstream call
- **Retries with exponential backoff** + jitter, max 3
- **Circuit breaker** so one failing provider doesn't take everyone down
- **PII redaction** with Presidio before sending to third-party providers (and logging the redactions)
- **Structured request logs** (JSON) with `request_id`, `prompt_version`, `model`, `tokens_in/out`, `latency_ms`, `cost_usd`, `cache_hit`, `error_type`

---

### Day 6 — Cost & latency dashboards

**Read (45 min):**
- [Langfuse — *Pricing & cost tracking*](https://langfuse.com/docs/integrations/llm-cost) — what you'll wire up in Week 15
- [OpenTelemetry — *GenAI semantic conventions*](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the emerging standard for LLM telemetry
- [Traceloop OpenLLMetry](https://github.com/traceloop/openllmetry) — drop-in OTel instrumentation for LLM SDKs
- [Helicone — *Effective LLM Caching*](https://www.helicone.ai/blog/effective-llm-caching) — the operating-numbers blog

**Hands-on (~2 hr):**
- Emit Prometheus metrics from your gateway: `requests_total`, `latency_ms_histogram`, `tokens_in_total`, `tokens_out_total`, `cache_hit_total`, `cost_usd_total`
- (Stretch) Also emit OpenTelemetry GenAI spans so the same telemetry feeds Langfuse / Phoenix / Honeycomb
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
- **Semantic cache:** hand-rolled (Redis + sentence-transformer); configurable threshold; cache hit returns in < 50ms
- **Provider-side prompt cache:** opt-in via a header / config (Anthropic 1-hour tier or OpenAI prompt caching)
- **Retries + circuit breaker:** misbehaving provider gets skipped for N seconds
- **PII redaction:** Presidio in the request path; redacted events go to an audit log
- **Rate limits** per API key (token bucket; configurable)
- **Structured logs**: JSON to stdout *and* SQLite (or Postgres) for queryability
- **Cost tracking:** per request, in USD, using current prices hardcoded with a comment
- **Prometheus** `/metrics` + Grafana dashboard JSON in the repo
- **LiteLLM comparison:** alongside `app/`, include a `litellm_proxy/config.yaml` that achieves the same routing+caching behaviour. Write a 1-page `vs-litellm.md` in `reports/` honestly comparing your hand-rolled gateway to the LiteLLM config (LOC, features, where each wins). Shows you know both worlds — most teams will end up running LiteLLM.

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
- [Helicone — *Effective LLM Caching*](https://www.helicone.ai/blog/effective-llm-caching) — vendor engineering blog with real numbers
- [AWS Database Blog — *Optimize LLM response costs and latency with effective caching*](https://aws.amazon.com/blogs/database/optimize-llm-response-costs-and-latency-with-effective-caching/)
- [Redis — *Vector database*](https://redis.io/docs/latest/develop/get-started/vector-database/)
- [GPTCache docs](https://gptcache.readthedocs.io/en/latest/) — for the algorithm; the library itself is in maintenance mode
- [GPT Semantic Cache paper](https://arxiv.org/abs/2411.05276)
- [Anthropic — *Prompt caching*](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) (incl. 1-hour TTL tier)
- [OpenAI — *Prompt caching*](https://platform.openai.com/docs/guides/prompt-caching)

**Routing**
- [LiteLLM — *Routing*](https://docs.litellm.ai/docs/routing) — the OSS de facto
- [Not-Diamond — *awesome-ai-model-routing*](https://github.com/Not-Diamond/awesome-ai-model-routing) — curated list
- [NotDiamond](https://www.notdiamond.ai/), [Martian Router](https://route.withmartian.com/) — hosted products
- [RouteLLM (LMSYS)](https://github.com/lm-sys/RouteLLM) — reference research project
- [LangChain — *Routing*](https://python.langchain.com/docs/how_to/routing/)

**Production gateways**
- [LiteLLM Proxy](https://docs.litellm.ai/docs/simple_proxy) — **the** OSS gateway in 2026
- [Portkey](https://portkey.ai/docs/welcome/introduction)
- [Kong AI Gateway](https://docs.konghq.com/hub/kong-inc/ai-proxy/)
- [Cloudflare AI Gateway](https://developers.cloudflare.com/ai-gateway/)
- [Helicone](https://docs.helicone.ai/)

**Observability conventions (preview for Week 15)**
- [OpenTelemetry — *GenAI semantic conventions*](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Traceloop OpenLLMetry](https://github.com/traceloop/openllmetry)

**Hardening / security**
- [Tenacity](https://tenacity.readthedocs.io/)
- [SlowAPI (rate limits for FastAPI)](https://slowapi.readthedocs.io/)
- [Microsoft Presidio (PII)](https://microsoft.github.io/presidio/)
- [OWASP — *GenAI Top 10 (2025)*](https://genai.owasp.org/llm-top-10/)
- [Meta Llama Guard / Purple-Llama](https://github.com/meta-llama/PurpleLlama)
- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
- [Anthropic — *Mitigating jailbreaks*](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks)

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
