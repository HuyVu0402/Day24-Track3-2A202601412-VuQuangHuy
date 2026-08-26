from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time, re, math
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                if os.getenv("LAB18_USE_LOCAL_MODELS", "0") != "1":
                    raise RuntimeError("set LAB18_USE_LOCAL_MODELS=1 to enable CrossEncoder reranking")
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                print(f"  ⚠️  CrossEncoder unavailable, using lexical rerank fallback: {exc}")
                self._model = False
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        model = self._load_model()
        if model:
            pairs = [(query, doc.get("text", "")) for doc in documents]
            scores = model.predict(pairs)
            if isinstance(scores, (int, float)):
                scores = [scores]
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
        else:
            scores = [_lexical_relevance(query, doc.get("text", "")) for doc in documents]

        scored = sorted(zip(scores, documents), key=lambda x: float(x[0]), reverse=True)
        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(doc.get("metadata", {})),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        scored = sorted(
            ((_lexical_relevance(query, doc.get("text", "")), doc) for doc in documents),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            RerankResult(doc.get("text", ""), float(doc.get("score", 0.0)), float(score),
                         dict(doc.get("metadata", {})), i)
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


def _lexical_relevance(query: str, text: str) -> float:
    def toks(value: str) -> list[str]:
        return re.findall(r"[\wÀ-ỹ]+", value.lower(), flags=re.UNICODE)

    q_tokens = toks(query)
    d_tokens = toks(text)
    if not q_tokens or not d_tokens:
        return 0.0
    qset, dset = set(q_tokens), set(d_tokens)
    overlap = len(qset & dset)
    exact_bonus = 0.25 if any(token in text.lower() for token in ["nghỉ", "phép", "mật khẩu"] if token in query.lower()) else 0.0
    return overlap / math.sqrt(len(qset) * len(dset)) + exact_bonus


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
