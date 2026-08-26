from __future__ import annotations

"""Phase A: RAGAS-style production evaluation for the 50-question test set."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ANSWERS_PATH, TEST_SET_PATH

Distribution = str

DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating", "Tighten system prompt and cite only retrieved evidence"),
    "context_recall": ("Missing relevant chunks", "Improve chunking, hybrid search coverage, or parent retrieval"),
    "context_precision": ("Too many irrelevant chunks", "Add stronger reranking and metadata/version filters"),
    "answer_relevancy": ("Answer does not match question", "Improve query-focused answer synthesis"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (
            self.faithfulness
            + self.answer_relevancy
            + self.context_precision
            + self.context_recall
        ) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }
        return min(scores, key=scores.get)


def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json not found at {path}. Run: python setup_answers.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        dist = item.get("distribution")
        if dist not in groups:
            raise ValueError(f"Unknown distribution: {dist}")
        groups[dist].append(item)
    return groups


def _metric_value(metrics, key: str) -> float:
    if isinstance(metrics, dict):
        value = metrics.get(key, 0.0)
    else:
        value = getattr(metrics, key, 0.0)
    return float(value or 0.0)


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    from src.m4_eval import evaluate_ragas

    raw = evaluate_ragas(
        [a["question"] for a in answers],
        [a["answer"] for a in answers],
        [a.get("contexts", []) for a in answers],
        [a["ground_truth"] for a in answers],
    )
    per_question = raw.get("per_question", [])

    results = []
    for answer, metrics in zip(answers, per_question):
        results.append(RagasResult(
            question_id=int(answer["id"]),
            distribution=answer["distribution"],
            question=answer["question"],
            answer=answer["answer"],
            contexts=answer.get("contexts", []),
            ground_truth=answer["ground_truth"],
            faithfulness=_metric_value(metrics, "faithfulness"),
            answer_relevancy=_metric_value(metrics, "answer_relevancy"),
            context_precision=_metric_value(metrics, "context_precision"),
            context_recall=_metric_value(metrics, "context_recall"),
        ))
    return results


def bottom_10(results: list[RagasResult]) -> list[dict]:
    output = []
    for rank, result in enumerate(sorted(results, key=lambda r: r.avg_score)[:10], start=1):
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[result.worst_metric]
        output.append({
            "rank": rank,
            "question_id": result.question_id,
            "distribution": result.distribution,
            "question": result.question,
            "avg_score": round(result.avg_score, 4),
            "worst_metric": result.worst_metric,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    matrix = {
        metric: {"factual": 0, "multi_hop": 0, "adversarial": 0}
        for metric in DIAGNOSTIC_TREE
    }
    for result in results:
        matrix[result.worst_metric][result.distribution] += 1

    distributions = ["factual", "multi_hop", "adversarial"]
    dominant_dist = max(distributions, key=lambda d: sum(matrix[m][d] for m in matrix))
    dominant_metric = max(matrix, key=lambda m: sum(matrix[m].values()))
    insight = (
        f"{dominant_dist} has the most weakest-metric cases. "
        f"The dominant weak metric is {dominant_metric}; recommended fix: "
        f"{DIAGNOSTIC_TREE[dominant_metric][1]}."
    )
    return {
        "matrix": matrix,
        "dominant_failure_distribution": dominant_dist,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }


def save_phase_a_report(
    results: list[RagasResult],
    clusters: dict,
    path: str = "reports/ragas_50q.json",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if not subset:
            continue
        per_dist[dist] = {
            "count": len(subset),
            "faithfulness": round(sum(r.faithfulness for r in subset) / len(subset), 4),
            "answer_relevancy": round(sum(r.answer_relevancy for r in subset) / len(subset), 4),
            "context_precision": round(sum(r.context_precision for r in subset) / len(subset), 4),
            "context_recall": round(sum(r.context_recall for r in subset) / len(subset), 4),
            "avg_score": round(sum(r.avg_score for r in subset) / len(subset), 4),
        }

    report = {
        "total_questions": len(results),
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": bottom_10(results),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved -> {path}")


if __name__ == "__main__":
    test_set = load_test_set_50q()
    groups = group_by_distribution(test_set)
    print(f"Loaded {len(test_set)} questions")
    for dist, qs in groups.items():
        print(f"  {dist}: {len(qs)} questions")

    results = run_ragas_50q(load_answers())
    clusters = cluster_analysis(results)
    save_phase_a_report(results, clusters)
    for item in bottom_10(results):
        print(
            f"#{item['rank']} [{item['distribution']}] "
            f"avg={item['avg_score']:.3f} worst={item['worst_metric']}"
        )
