from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, re, math
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("questions, answers, contexts, and ground_truths must have the same length")

    try:
        if not OPENAI_API_KEY or os.getenv("LAB18_USE_RAGAS", "0") != "1":
            raise RuntimeError("set LAB18_USE_RAGAS=1 with OPENAI_API_KEY to run RAGAS")
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            )
            for _, row in df.iterrows()
        ]
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed, using lexical evaluator: {e}")
        per_question = [
            _heuristic_eval(q, a, c, gt)
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]

    aggregate = {
        "faithfulness": _avg([r.faithfulness for r in per_question]),
        "answer_relevancy": _avg([r.answer_relevancy for r in per_question]),
        "context_precision": _avg([r.context_precision for r in per_question]),
        "context_recall": _avg([r.context_recall for r in per_question]),
        "per_question": [r.__dict__ for r in per_question],
    }
    return aggregate


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    normalized = [
        item if isinstance(item, EvalResult) else EvalResult(**item)
        for item in eval_results
    ]
    diagnostic_tree = {
        "faithfulness": ("Answer is not sufficiently grounded in retrieved context",
                         "Tighten generation prompt and cite only retrieved evidence"),
        "answer_relevancy": ("Answer does not match the user question",
                             "Improve query-focused answer synthesis and refuse missing information"),
        "context_precision": ("Retriever returned too many irrelevant chunks",
                              "Use stronger reranking, metadata filters, or lower top-k"),
        "context_recall": ("Retriever missed required evidence",
                           "Improve chunking, hybrid search coverage, or return parent context"),
    }
    scored = []
    for result in normalized:
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        avg_score = _avg(metrics.values())
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        scored.append({
            "question": result.question,
            "answer": result.answer,
            "ground_truth": result.ground_truth,
            "contexts": result.contexts,
            "worst_metric": worst_metric,
            "score": round(avg_score, 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
            "error_tree": _error_tree(worst_metric),
        })
    return sorted(scored, key=lambda item: item["score"])[:bottom_n]


def _tokens(text: str) -> set[str]:
    stopwords = {"là", "và", "có", "cho", "theo", "được", "phải", "trong", "khi", "của", "nhân", "viên"}
    return {
        token for token in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
        if len(token) > 1 and token not in stopwords
    }


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / math.sqrt(len(ta) * len(tb))


def _avg(values) -> float:
    vals = [float(v) for v in values]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _heuristic_eval(question: str, answer: str, contexts: list[str], ground_truth: str) -> EvalResult:
    joined_context = "\n\n".join(contexts)
    per_context = [_similarity(ground_truth, ctx) for ctx in contexts] or [0.0]
    return EvalResult(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
        faithfulness=min(1.0, _similarity(answer, joined_context) * 1.2),
        answer_relevancy=min(1.0, (_similarity(answer, question) + _similarity(answer, ground_truth)) / 2 * 1.5),
        context_precision=min(1.0, sum(1 for s in per_context if s > 0.15) / max(len(per_context), 1)),
        context_recall=min(1.0, max(per_context) * 1.4),
    )


def _error_tree(worst_metric: str) -> str:
    mapping = {
        "context_recall": "Output sai -> Context thiếu bằng chứng -> kiểm tra M1/M2/top-k",
        "context_precision": "Output sai -> Context có nhiễu -> kiểm tra M2/RRF/M3",
        "faithfulness": "Output sai -> Context đúng nhưng answer không bám context -> kiểm tra prompt/generation",
        "answer_relevancy": "Output sai -> Answer lệch trọng tâm query -> kiểm tra prompt/query intent",
    }
    return mapping[worst_metric]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
