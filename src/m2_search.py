from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys, re, math
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").replace("_", " ")
    except Exception:
        return text


def _tokens(text: str) -> list[str]:
    segmented = segment_vietnamese(text.lower())
    return re.findall(r"[\wÀ-ỹ]+", segmented, flags=re.UNICODE)


def _lexical_score(query: str, text: str) -> float:
    q = _tokens(query)
    d = _tokens(text)
    if not q or not d:
        return 0.0
    qset, dset = set(q), set(d)
    overlap = len(qset & dset)
    return overlap / math.sqrt(len(qset) * len(dset))


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = [_tokens(chunk.get("text", "")) for chunk in self.documents]
        if not self.corpus_tokens:
            self.bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception as exc:
            print(f"  ⚠️  BM25 unavailable, using lexical fallback: {exc}")
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if not self.documents:
            return []
        if self.bm25 is not None:
            scores = self.bm25.get_scores(_tokens(query))
        else:
            scores = [_lexical_score(query, doc.get("text", "")) for doc in self.documents]
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        results = []
        for idx in ranked[:top_k]:
            score = float(scores[idx])
            if score <= 0:
                continue
            doc = self.documents[idx]
            results.append(SearchResult(doc.get("text", ""), score, dict(doc.get("metadata", {})), "bm25"))
        return results


class DenseSearch:
    def __init__(self):
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except Exception as exc:
            print(f"  ⚠️  Qdrant client unavailable, using in-memory dense fallback: {exc}")
            self.client = None
        self._encoder = None
        self._fallback_documents: list[dict] = []

    def _get_encoder(self):
        if self._encoder is None:
            if os.getenv("LAB18_USE_LOCAL_MODELS", "0") != "1":
                raise RuntimeError("set LAB18_USE_LOCAL_MODELS=1 to enable dense embedding model")
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        self._fallback_documents = list(chunks)
        if not chunks or self.client is None:
            return
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams
            if hasattr(self.client, "recreate_collection"):
                self.client.recreate_collection(
                    collection,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                )
            texts = [c.get("text", "") for c in chunks]
            vectors = self._get_encoder().encode(texts, show_progress_bar=False)
            points = [
                PointStruct(
                    id=i,
                    vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                    payload={**c.get("metadata", {}), "text": c.get("text", "")},
                )
                for i, (c, vector) in enumerate(zip(chunks, vectors))
            ]
            self.client.upsert(collection, points)
        except Exception as exc:
            print(f"  ⚠️  Dense indexing fallback activated: {exc}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if self.client is not None:
            try:
                query_vector = self._get_encoder().encode(query)
                if hasattr(query_vector, "tolist"):
                    query_vector = query_vector.tolist()
                response = self.client.query_points(collection, query=query_vector, limit=top_k)
                return [
                    SearchResult(
                        text=pt.payload.get("text", ""),
                        score=float(pt.score),
                        metadata={k: v for k, v in pt.payload.items() if k != "text"},
                        method="dense",
                    )
                    for pt in response.points
                ]
            except Exception as exc:
                print(f"  ⚠️  Dense search fallback activated: {exc}")

        scored = [
            (doc, _lexical_score(query, doc.get("text", "")))
            for doc in self._fallback_documents
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            SearchResult(doc.get("text", ""), float(score), dict(doc.get("metadata", {})), "dense")
            for doc, score in scored[:top_k]
            if score > 0
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = result.metadata.get("id") or result.metadata.get("source", "") + "::" + result.text[:120]
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "result": result}
            rrf_scores[key]["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.values(), key=lambda item: item["score"], reverse=True)
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=dict(item["result"].metadata),
            method="hybrid",
        )
        for item in ranked[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
