# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh vien:** Vu Quang Huy  
**Ngay:** 2026-08-26

## Guard Stack Architecture

```
User Input
    |
    v
[PII Scan]
    | block if VN_CCCD / VN_PHONE / EMAIL_ADDRESS is detected
    | action: reject + log sanitized entity type
    v
[Input Rail]
    | block if off-topic / jailbreak / prompt injection / sensitive-data request
    | action: return refusal with blocked_reason
    v
[RAG Pipeline - Day 18]
    | M1 chunking -> M2 hybrid search -> M3 rerank -> answer synthesis
    v
[Output Rail]
    | redact PII and block unsafe generated content
    | action: redact or replace with safe response
    v
User Response
```

## Guard Stack Pipeline

| Layer | Tool | Latency P95 | Failure Action |
|---|---|---:|---|
| PII Detection | Local Presidio-compatible regex recognizers | 0.01ms | Reject + log entity type |
| Topic/Jailbreak | Local NeMo-compatible input rail fallback | 0.01ms | Block + reason |
| RAG Pipeline | Day 18 BM25/dense fallback + reranker fallback | Offline measured in setup | Fallback to retrieved context |
| Output Check | Local output rail | Included in guard total | Redact PII or block response |

## Latency Budget

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---:|---:|---:|---:|
| PII Detection | 0.01 | 0.01 | 0.01 | <10ms |
| Input Rail | 0.01 | 0.01 | 0.01 | <300ms |
| Total Guard | 0.01 | 0.02 | 0.02 | <500ms |

**Budget OK?** Yes  
**Comment:** The measured guard stack is local/offline, so latency is much lower than a hosted NeMo or LLM rail. If replacing the local rail with an API-backed guard, the likely bottleneck becomes the input/output rail network call.

## CI/CD Gates

- [x] `python src/phase_a_ragas.py` must produce `reports/ragas_50q.json` with 50 questions.
- [x] RAGAS-style adversarial avg_score must be lower than factual avg_score to confirm the stress set is meaningful.
- [x] `python src/phase_b_judge.py` must produce `reports/judge_results.json` and Cohen kappa must stay above 0.6 on the 10 human labels.
- [x] `python src/phase_c_guard.py` must produce `reports/guard_results.json`.
- [x] Adversarial suite pass rate must be at least 15/20; current result is 20/20.
- [x] Guard P95 latency must be below 500ms; current result is 0.02ms.
- [x] `pytest tests/ -q` must pass before push.

## Monitoring

| Metric | Current Lab Value | Alert Threshold | Action |
|---|---:|---:|---|
| RAGAS avg_score factual | 0.8510 | <0.75 | Review retrieval and answer prompt |
| RAGAS avg_score adversarial | 0.5804 | Unexpectedly high or falling trend | Review version-conflict set |
| Worst RAGAS metric | context_recall / answer_relevancy | <0.60 | Improve parent retrieval and query-focused synthesis |
| Dominant weak distribution by avg | adversarial | Any spike in production failures | Add metadata filters and new adversarial tests |
| Cohen kappa | 1.0000 | <0.60 | Recalibrate judge rubric with human labels |
| Adversarial pass rate | 20/20 | <18/20 production gate | Add/adjust input rail patterns |
| Guard P95 latency | 0.02ms | >500ms | Profile rail layer and cache static checks |

## Ket qua thuc te tu Lab

| Item | Result |
|---|---:|
| RAGAS avg_score factual | 0.8510 |
| RAGAS avg_score multi_hop | 0.6953 |
| RAGAS avg_score adversarial | 0.5804 |
| Worst metric | context_recall / answer_relevancy |
| Dominant failure distribution | adversarial by avg_score |
| Cohen kappa | 1.0000 |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 0.02ms |

## Nhan xet & Cai tien

The eval stack runs fully offline and can be used as a CI smoke gate without OpenAI API access. The RAG pipeline behaves as expected under stress: adversarial questions score lower than factual questions, especially for version conflicts and negation traps. The most valuable next improvement is metadata-aware retrieval that prefers active policy versions and excludes expired ones. For production, the local rule rails should be kept as a cheap first layer, then combined with an API-backed NeMo/LLM rail for broader unseen attacks.
