"""倒排索引与 BM25 稀疏检索实现。"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from voice_optimized_rag.retrieval.vector_store import SearchResult

ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
HAN_SEGMENT_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize_text(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text for sparse retrieval.

    中文使用 2-gram/3-gram 和整词回退，英文使用字母数字词。
    这是轻量实现，避免为仓库强制引入额外分词依赖。
    """
    normalized = text.lower().strip()
    if not normalized:
        return []

    tokens: list[str] = []
    tokens.extend(ASCII_TOKEN_RE.findall(normalized))

    for segment in HAN_SEGMENT_RE.findall(normalized):
        if len(segment) <= 2:
            tokens.append(segment)
            continue

        tokens.append(segment)
        tokens.extend(segment[i:i + 2] for i in range(len(segment) - 1))
        if len(segment) >= 4:
            tokens.extend(segment[i:i + 3] for i in range(len(segment) - 2))

    return [token for token in tokens if token]


class InvertedIndex:
    """基于 BM25 的轻量倒排索引。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._documents: list[SearchResult] = []
        self._doc_lengths: list[int] = []
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._avg_doc_length: float = 0.0

    @property
    def size(self) -> int:
        return len(self._documents)

    def rebuild(self, documents: list[SearchResult]) -> None:
        self._documents = []
        self._doc_lengths = []
        self._postings.clear()

        for document in documents:
            self.add_document(document)

        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)
        else:
            self._avg_doc_length = 0.0

    def add_document(self, document: SearchResult) -> None:
        doc_id = len(self._documents)
        tokens = tokenize_text(document.text)
        term_counts = Counter(tokens)

        self._documents.append(document)
        self._doc_lengths.append(max(len(tokens), 1))

        for token, count in term_counts.items():
            self._postings[token][doc_id] = count

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._documents:
            return []

        query_tokens = list(dict.fromkeys(tokenize_text(query)))
        if not query_tokens:
            return []

        scores: dict[int, float] = defaultdict(float)
        total_docs = len(self._documents)
        avg_doc_length = self._avg_doc_length or 1.0

        for token in query_tokens:
            postings = self._postings.get(token)
            if not postings:
                continue

            document_frequency = len(postings)
            idf = math.log((total_docs - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0)

            for doc_id, term_frequency in postings.items():
                doc_length = self._doc_lengths[doc_id]
                numerator = term_frequency * (self._k1 + 1.0)
                denominator = term_frequency + self._k1 * (1.0 - self._b + self._b * doc_length / avg_doc_length)
                scores[doc_id] += idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        results: list[SearchResult] = []
        for doc_id, score in ranked:
            document = self._documents[doc_id]
            results.append(SearchResult(
                text=document.text,
                metadata=document.metadata,
                score=float(score),
                index=document.index,
                embedding=document.embedding,
                sparse_score=float(score),
                retrieval_source="sparse",
            ))
        return results