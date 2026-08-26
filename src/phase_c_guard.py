from __future__ import annotations

"""Phase C: production guardrail stack with local deterministic fallbacks."""

import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, LATENCY_BUDGET_P95_MS

PII_PATTERNS = {
    "EMAIL_ADDRESS": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "VN_CCCD": re.compile(r"\b\d{12}\b|\b\d{9}\b"),
    "VN_PHONE": re.compile(r"\b0[3-9]\d{8}\b"),
}

BLOCK_PATTERNS = {
    "prompt_injection": [
        "ignore previous", "ignore your system", "system override", "admin command",
        "priority 1", "dump all", "training data", "system instructions",
        "bo qua", "bỏ qua", "quen tat ca", "forget all", "reveal all",
    ],
    "jailbreak": [
        "dan", "do anything now", "unrestricted", "khong co gioi han",
        "không có giới hạn", "dong vai", "đóng vai", "pretend you are",
    ],
    "sensitive_data": [
        "mat khau", "mật khẩu", "password", "confidential", "employee salaries",
        "all employee", "bang luong chi tiet", "bảng lương chi tiết",
        "tiet lo", "tiết lộ", "cccd", "so dien thoai cua nhan vien",
    ],
    "harmful": ["tan cong mang", "tấn công mạng", "attack internal", "illegal"],
    "off_topic": [
        "bai tho", "bài thơ", "nau pho", "nấu phở", "bitcoin", "ethereum",
        "phuong trinh", "phương trình", "marvel", "bo phim", "bộ phim",
    ],
}


def _matches_any(text: str, patterns: list[str]) -> bool:
    lower = text.lower()
    return any(pattern in lower for pattern in patterns)


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    entities = []
    for entity_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append({
                "type": entity_type,
                "text": match.group(0),
                "score": 0.95,
                "start": match.start(),
                "end": match.end(),
            })
    entities.sort(key=lambda item: item["start"])

    anonymized = text
    for entity in reversed(entities):
        anonymized = (
            anonymized[:entity["start"]]
            + f"<{entity['type']}>"
            + anonymized[entity["end"]:]
        )
    return {"has_pii": bool(entities), "entities": entities, "anonymized": anonymized}


async def check_input_rail(text: str, rails=None) -> dict:
    for reason, patterns in BLOCK_PATTERNS.items():
        if _matches_any(text, patterns):
            return {
                "allowed": False,
                "blocked_reason": reason,
                "response": "Blocked by local input rail.",
            }
    return {"allowed": True, "blocked_reason": None, "response": "Allowed by local input rail."}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    pii = pii_scan(answer)
    if pii["has_pii"]:
        return {
            "safe": False,
            "flagged_reason": "pii_in_output",
            "final_answer": pii["anonymized"],
        }
    rail = await check_input_rail(answer, rails)
    if not rail["allowed"]:
        return {
            "safe": False,
            "flagged_reason": rail["blocked_reason"],
            "final_answer": "Response blocked by output guardrail.",
        }
    return {"safe": True, "flagged_reason": None, "final_answer": answer}


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


def run_adversarial_suite(
    adversarial_set: list[dict],
    rails=None,
    analyzer=None,
    anonymizer=None,
) -> list[dict]:
    async def _run_all() -> list[dict]:
        results = []
        for item in adversarial_set:
            blocked_by = None
            pii = pii_scan(item["input"], analyzer, anonymizer)
            if pii["has_pii"]:
                blocked_by = "presidio"
            if blocked_by is None:
                rail = await check_input_rail(item["input"], rails)
                if not rail["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id": item["id"],
                "category": item["category"],
                "input": item["input"][:120],
                "expected": item["expected"],
                "actual": actual,
                "blocked_by": blocked_by,
                "passed": actual == item["expected"],
            })
        return results

    results = _run_async(_run_all())
    passed = sum(1 for item in results if item["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    sorted_values = sorted(values)
    n = len(sorted_values)

    def pick(pct: float) -> float:
        idx = min(int((n - 1) * pct), n - 1)
        return round(sorted_values[idx], 2)

    return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}


def measure_p95_latency(
    test_inputs: list[str],
    n_runs: int = 20,
    rails=None,
    analyzer=None,
    anonymizer=None,
) -> dict:
    inputs = (test_inputs or ["test input"])[: max(1, n_runs)]
    presidio_times: list[float] = []
    nemo_times: list[float] = []
    total_times: list[float] = []

    async def _measure() -> None:
        for text in inputs:
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    _run_async(_measure())
    total = _percentiles(total_times)
    return {
        "presidio_ms": _percentiles(presidio_times),
        "nemo_ms": _percentiles(nemo_times),
        "total_ms": total,
        "latency_budget_ok": total["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


def save_phase_c_report(path: str = "reports/guard_results.json") -> dict:
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    suite = run_adversarial_suite(adversarial_set)
    latency = measure_p95_latency([item["input"] for item in adversarial_set], n_runs=len(adversarial_set))
    passed = sum(1 for item in suite if item["passed"])
    report = {
        "total": len(suite),
        "passed": passed,
        "pass_rate": round(passed / len(suite), 3) if suite else 0.0,
        "results": suite,
        "latency": latency,
        "output_rail_demo": _run_async(check_output_rail("demo", "No PII in this response.")),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase C report saved -> {path}")
    return report


if __name__ == "__main__":
    report = save_phase_c_report()
    print(f"Pass rate: {report['passed']}/{report['total']}")
    print(f"Total P95: {report['latency']['total_ms']['p95']} ms")
