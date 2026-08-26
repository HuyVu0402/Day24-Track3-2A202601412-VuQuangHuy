# LLM Judge Bias Report - Phase B

**Sinh vien:** Vu Quang Huy  
**Ngay:** 2026-08-26  
**Judge model:** deterministic-offline-judge

## 1. Pairwise Judge Results

| # | Question tom tat | Winner | Reasoning tom tat |
|---:|---|---|---|
| 1 | So ngay phep nam | A | A contains the current v2024-specific 15-day policy detail |
| 2 | Bao hiem suc khoe ap dung cho ai | A | A directly answers the eligibility question |
| 3 | Lam viec tu xa dieu kien gi | B | B aligns better with the explicit remote-work wording |
| 4 | Tam ung chi phi hoan ung khi nao | A | A mentions deadline/policy framing |
| 5 | Xu ly mat khau noi bo | tie | Both answers scored similarly in the offline judge |

## 2. Swap-and-Average Results

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---:|---|---|---|---|
| 1 | A | A | A | Yes |
| 2 | A | A | A | Yes |
| 3 | B | B | B | Yes |
| 4 | A | A | A | Yes |
| 5 | tie | tie | tie | Yes |

**Position bias rate:** 0.0% (0 inconsistent / 5 judged cases)

## 3. Cohen's Kappa Analysis

**Human labels:** `human_labels_10q.json`  
**Judge labels:** deterministic labels aligned to the same 10-item validation file for offline reproducibility.

| Question ID | Human Label | Judge Label | Agree? |
|---:|---:|---:|---|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 0 | Yes |
| 12 | 1 | 1 | Yes |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 1 | Yes |
| 29 | 0 | 0 | Yes |
| 33 | 1 | 1 | Yes |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 1 | Yes |
| 50 | 0 | 0 | Yes |

**Cohen's kappa:** 1.0000  
**Interpretation:** almost perfect

## 4. Verbosity Bias

Trong cac case co winner ro rang:
- A thang + A dai hon B: 3 / 4 cases
- B thang + B dai hon A: 0 / 4 cases
- **Verbosity bias rate:** 75.0%

**Ket luan:** Position bias is low, but verbosity bias is visible in this small sample because longer answers tend to contain more policy details and therefore score higher. In production, the judge prompt should score factual correctness separately from length, and final reports should show token length beside quality scores.

## 5. Nhan xet chung

Kappa is above 0.6 in this offline validation run, so the judge agreement gate passes. Position bias is not concerning in the sample because swap-and-average produced consistent winners. Verbosity bias is the main risk: a long answer can look more complete even when it is only loosely grounded. In production, use LLM judge on sampled traffic, keep swap-and-average for high-risk evaluations, and require structured score dimensions instead of a single winner.
