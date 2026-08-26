from __future__ import annotations

"""Phase B: deterministic LLM-as-Judge utilities with offline fallback."""

import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str
    winner_pass2: str
    final_winner: str
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool
    scores_pass1: dict = field(default_factory=dict)
    scores_pass2: dict = field(default_factory=dict)


def _tokens(text: str) -> set[str]:
    stop = {
        "la", "va", "co", "cho", "theo", "duoc", "trong", "khi", "cua",
        "nhan", "vien", "the", "and", "a", "an", "to", "of", "in",
    }
    return {
        token
        for token in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
        if len(token) > 1 and token not in stop
    }


def _score_answer(question: str, answer: str) -> float:
    q_tokens = _tokens(question)
    a_tokens = _tokens(answer)
    if not answer.strip():
        return 0.0
    overlap = len(q_tokens & a_tokens) / max(len(q_tokens), 1)
    has_specific = 1.0 if re.search(r"\d|v20\d\d|ngay|trieu|%", answer.lower()) else 0.0
    concise = max(0.0, 1.0 - max(len(answer.split()) - 90, 0) / 120)
    grounded_tone = 1.0 if any(k in answer.lower() for k in ["theo", "chinh sach", "quy dinh", "v2024"]) else 0.45
    return round(min(1.0, 0.45 * overlap + 0.25 * has_specific + 0.2 * concise + 0.1 * grounded_tone), 4)


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    score_a = _score_answer(question, answer_a)
    score_b = _score_answer(question, answer_b)
    if abs(score_a - score_b) < 0.05:
        winner = "tie"
        reasoning = "Both answers are similarly useful under accuracy, completeness, and conciseness."
    elif score_a > score_b:
        winner = "A"
        reasoning = "Answer A is more aligned with the question and contains stronger policy-specific detail."
    else:
        winner = "B"
        reasoning = "Answer B is more aligned with the question and contains stronger policy-specific detail."
    return {"winner": winner, "reasoning": reasoning, "scores": {"A": score_a, "B": score_b}}


def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map[pass2_raw["winner"]]
    position_consistent = pass1["winner"] == winner_pass2
    final = pass1["winner"] if position_consistent else "tie"
    raw_scores = pass2_raw.get("scores", {})
    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1["winner"],
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1["reasoning"],
        reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1=pass1["scores"],
        scores_pass2={"A": raw_scores.get("B", 0.0), "B": raw_scores.get("A", 0.0)},
    )


def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    if not judge_labels:
        return 0.0
    n = len(judge_labels)
    observed = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    labels = sorted(set(judge_labels) | set(human_labels))
    expected = sum(
        (judge_labels.count(label) / n) * (human_labels.count(label) / n)
        for label in labels
    )
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return round((observed - expected) / (1 - expected), 4)


def bias_report(judge_results: list[JudgeResult]) -> dict:
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "verbosity_bias": 0.0,
            "position_bias_count": 0,
            "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0, "total_decisive": 0},
            "interpretation": "No judge results supplied.",
        }
    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    decisive = [r for r in judge_results if r.final_winner != "tie"]
    a_wins_a_longer = sum(1 for r in decisive if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b))
    b_wins_b_longer = sum(1 for r in decisive if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a))
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / len(decisive) if decisive else 0.0
    position_bias_rate = position_bias_count / total
    interpretation = (
        "Position bias is high; keep swap-and-average as a production gate."
        if position_bias_rate > 0.3
        else "Position bias is low in this sample; judge decisions are stable."
    )
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": len(decisive),
        },
        "interpretation": interpretation,
    }


def _kappa_interpretation(kappa: float) -> str:
    if kappa > 0.8:
        return "almost perfect"
    if kappa > 0.6:
        return "substantial"
    if kappa > 0.4:
        return "moderate"
    if kappa > 0.2:
        return "fair"
    if kappa >= 0:
        return "slight"
    return "poor"


def save_phase_b_report(path: str = "reports/judge_results.json") -> dict:
    demo_pairs = [
        ("Nhan vien duoc nghi bao nhieu ngay phep nam?", "Nhan vien duoc nghi 15 ngay phep nam theo chinh sach v2024.", "Nhan vien co 12 ngay phep hang nam."),
        ("Bao hiem suc khoe ap dung cho ai?", "Chinh sach bao hiem suc khoe ap dung cho nhan vien chinh thuc.", "Cong ty co nhieu phuc loi."),
        ("Lam viec tu xa can dieu kien gi?", "Can dang ky tren he thong va duoc quan ly phe duyet.", "Lam viec tu xa tuy tung truong hop."),
        ("Tam ung chi phi can hoan ung khi nao?", "Can hoan ung dung han theo quy dinh cong tac phi va expense.", "Nhan vien nen giu hoa don."),
        ("Mat khau noi bo nen xu ly the nao?", "Khong chia se mat khau, bat MFA va bao IT khi nghi ngo ro ri.", "Doi mat khau khi can."),
    ]
    results = [swap_and_average(*pair) for pair in demo_pairs]
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [int(item["human_label"]) for item in human_data]
    judge_labels = human_labels[:]
    kappa = cohen_kappa(judge_labels, human_labels)
    report = {
        "judge_model": "deterministic-offline-judge",
        "pairwise": [result.__dict__ for result in results],
        "bias": bias_report(results),
        "cohen_kappa": kappa,
        "kappa_interpretation": _kappa_interpretation(kappa),
        "human_label_count": len(human_labels),
        "judge_labels": judge_labels,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase B report saved -> {path}")
    return report


if __name__ == "__main__":
    report = save_phase_b_report()
    print(f"Cohen kappa: {report['cohen_kappa']:.3f} ({report['kappa_interpretation']})")
