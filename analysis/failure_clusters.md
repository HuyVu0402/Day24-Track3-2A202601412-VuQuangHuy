# Failure Cluster Analysis - Phase A

**Sinh vien:** Vu Quang Huy  
**Ngay:** 2026-08-26  
**Eval mode:** Day18 lexical fallback evaluator, generated from `answers_50q.json`

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.8187 | 0.8284 | 0.7908 |
| answer_relevancy | 0.7962 | 0.5436 | 0.5019 |
| context_precision | 0.9000 | 0.8333 | 0.5667 |
| context_recall | 0.8890 | 0.5758 | 0.4624 |
| **avg_score** | **0.8510** | **0.6953** | **0.5804** |

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---:|---|---|---:|---|
| 1 | adversarial | Bao lau phai doi mat khau mot lan? | 0.3459 | context_precision |
| 2 | adversarial | Manager co the dung VPN ca nhan khi WFH khong? | 0.4069 | answer_relevancy |
| 3 | multi_hop | Laptop 30 trieu cho nhan vien moi can ai phe duyet va can gi tu CNTT? | 0.4233 | answer_relevancy |
| 4 | adversarial | Nhan vien duoc nghi bao nhieu ngay phep nam? | 0.4613 | context_recall |
| 5 | multi_hop | Thu viec thang 3 phat hien vi pham bao mat nen va khong nen lam gi? | 0.4694 | answer_relevancy |
| 6 | multi_hop | Tam ung 4 trieu va 7 trieu khac nhau the nao? | 0.5116 | context_recall |
| 7 | adversarial | Nhan vien thu viec co duoc huong PVI khong? | 0.5337 | context_recall |
| 8 | adversarial | Co can kich hoat MFA khong? | 0.5448 | context_precision |
| 9 | multi_hop | So sanh mat khau policy v1.0 va v2.0 | 0.5614 | context_recall |
| 10 | adversarial | Mat khau toi thieu bao nhieu ky tu? | 0.5947 | context_recall |

## 3. Failure Cluster Matrix

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 7 | 1 | 0 | 8 |
| answer_relevancy | 9 | 7 | 1 | 17 |
| context_precision | 3 | 2 | 3 | 8 |
| context_recall | 1 | 10 | 6 | 17 |

## 4. Dominant Failure Analysis

**Dominant distribution:** factual by raw weakest-metric count, but adversarial is the weakest by average score.  
**Dominant metric:** context_recall and answer_relevancy are tied with 17 weakest cases each.

Factual questions have many weakest-metric labels because there are 20 factual examples and the lexical fallback often retrieves related but not perfectly focused context. The more important production signal is distribution average: adversarial is clearly lowest at 0.5804, followed by multi-hop at 0.6953. Version-conflict and negation-trap questions need stronger metadata filtering for `status`, `version`, and effective date.

## 5. Suggested Fixes

| Metric yeu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | Answer is not tightly grounded in retrieved chunks | Require citations and constrain generation to retrieved evidence only |
| context_recall | Required document/version is missing from top-k | Return parent chunks, raise BM25 coverage, and filter by effective policy version |
| context_precision | Retrieved chunks mix old and current policies | Add metadata filters for `status`, `version`, and `effective_date`; keep reranker enabled |
| answer_relevancy | Answer copies context but misses the exact question intent | Add query-intent rewrite and answer templates for calculation/comparison questions |

## 6. Nhan xet ve Adversarial Distribution

Adversarial avg_score is 0.5804, lower than factual 0.8510 and multi-hop 0.6953, which is the expected stress-test pattern. Six of the bottom ten are adversarial, mostly involving password/VPN/current-policy conflicts. The main weakness is version selection: the retriever can surface both old and current policy chunks, so the answer layer must explicitly prefer active versions such as v2024 leave policy and v2.0 password policy.
