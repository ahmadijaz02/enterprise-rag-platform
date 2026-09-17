from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from rank_bm25 import BM25Okapi


def reciprocal_rank_fusion(result_lists: Iterable[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            scores[item] += 1.0 / (k + rank)
    return [item for item, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)]


class KeywordIndex:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.bm25: BM25Okapi | None = None

    def build(self, documents: list[tuple[str, str]]) -> None:
        self.ids = [doc_id for doc_id, _ in documents]
        tokens = [text.lower().split() for _, text in documents]
        self.bm25 = BM25Okapi(tokens) if tokens else None

    def search(self, query: str, top_k: int = 8) -> list[str]:
        if not self.bm25 or not self.ids:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return [self.ids[i] for i in ranked[:top_k]]
