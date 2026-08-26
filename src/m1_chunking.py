from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re, hashlib
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


def _metadata_from_text(text: str) -> dict:
    meta: dict[str, str] = {}
    title = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if title:
        meta["title"] = title.group(1).strip()

    header_line = re.search(r"^>\s*(.+)$", text, flags=re.MULTILINE)
    if header_line:
        parts = [p.strip() for p in header_line.group(1).split("|")]
        for part in parts:
            if ":" not in part:
                continue
            key, value = [x.strip() for x in part.split(":", 1)]
            normalized = {
                "Phiên bản": "version",
                "Ngày hiệu lực": "effective_date",
                "Phòng ban": "department",
                "Trạng thái": "status",
            }.get(key, key.lower().replace(" ", "_"))
            meta[normalized] = value
    return meta


def _merge_metadata(metadata: dict | None, text: str = "") -> dict:
    base = dict(metadata or {})
    extracted = _metadata_from_text(text)
    return {**extracted, **base}


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?。])\s+|\n\s*\n", text.strip())
    return [s.strip() for s in chunks if s and s.strip()]


def _fixed_size_chunks(text: str, size: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    current = ""
    for para in paragraphs or [text.strip()]:
        if len(para) > size:
            if current.strip():
                pieces.append(current.strip())
                current = ""
            for i in range(0, len(para), size):
                piece = para[i:i + size].strip()
                if piece:
                    pieces.append(piece)
            continue
        if current and len(current) + len(para) + 2 > size:
            pieces.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}".strip() if current else para
    if current.strip():
        pieces.append(current.strip())
    return pieces


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    meta = _merge_metadata(metadata, text)
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(sentences[0], {**meta, "strategy": "semantic", "chunk_index": 0})]

    groups: list[list[str]] = [[sentences[0]]]
    try:
        if os.getenv("LAB18_USE_LOCAL_MODELS", "0") != "1":
            raise RuntimeError("set LAB18_USE_LOCAL_MODELS=1 to enable sentence-transformer chunking")
        from numpy import dot
        from numpy.linalg import norm
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences)
        for i in range(1, len(sentences)):
            sim = float(dot(embeddings[i - 1], embeddings[i]) / (norm(embeddings[i - 1]) * norm(embeddings[i]) + 1e-9))
            if sim < threshold:
                groups.append([sentences[i]])
            else:
                groups[-1].append(sentences[i])
    except Exception as exc:
        print(f"  ⚠️  Semantic model unavailable, using paragraph fallback: {exc}")
        groups = []
        current: list[str] = []
        for sentence in sentences:
            if current and sum(len(s) for s in current) + len(sentence) > 500:
                groups.append(current)
                current = []
            current.append(sentence)
        if current:
            groups.append(current)

    max_allowed = len(chunk_basic(text, chunk_size=100, metadata=meta)) + 2
    while len(groups) > max_allowed and len(groups) > 1:
        merged: list[list[str]] = []
        for i in range(0, len(groups), 2):
            if i + 1 < len(groups):
                merged.append(groups[i] + groups[i + 1])
            else:
                merged.append(groups[i])
        groups = merged

    return [
        Chunk(" ".join(group).strip(), {**meta, "strategy": "semantic", "chunk_index": i})
        for i, group in enumerate(groups)
        if " ".join(group).strip()
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    meta = _merge_metadata(metadata, text)
    parents: list[Chunk] = []
    children: list[Chunk] = []
    source_key = meta.get("source") or meta.get("title") or hashlib.md5(text.encode("utf-8")).hexdigest()[:8]

    for parent_idx, parent_text in enumerate(_fixed_size_chunks(text, parent_size)):
        pid = f"{source_key}:parent:{parent_idx}"
        parent_meta = {**meta, "strategy": "hierarchical", "chunk_type": "parent",
                       "parent_id": pid, "chunk_index": parent_idx}
        parents.append(Chunk(parent_text, parent_meta, parent_id=pid))
        for child_idx, child_text in enumerate(_fixed_size_chunks(parent_text, child_size)):
            child_meta = {**meta, "strategy": "hierarchical", "chunk_type": "child",
                          "parent_id": pid, "parent_chunk_index": parent_idx,
                          "chunk_index": len(children), "child_index": child_idx}
            children.append(Chunk(child_text, child_meta, parent_id=pid))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    meta = _merge_metadata(metadata, text)
    chunks: list[Chunk] = []
    current_header = meta.get("title", "")
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines, current_header
        body = "\n".join(current_lines).strip()
        if not body:
            current_lines = []
            return
        section = current_header.strip("# ").strip() if current_header else "preamble"
        chunks.append(Chunk(body, {**meta, "section": section, "strategy": "structure",
                                   "chunk_index": len(chunks)}))
        current_lines = []

    for line in text.splitlines():
        if re.match(r"^#{1,3}\s+.+$", line):
            flush()
            current_header = line
            current_lines = [line]
        else:
            current_lines.append(line)
    flush()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
