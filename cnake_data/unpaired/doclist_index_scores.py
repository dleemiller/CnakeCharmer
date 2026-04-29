from __future__ import annotations


def score_documents(query_terms: list[int], doc_term_ids: list[list[int]]) -> list[int]:
    q = set(query_terms)
    out = [0] * len(doc_term_ids)
    for i, terms in enumerate(doc_term_ids):
        s = 0
        for t in terms:
            if t in q:
                s += 1
        out[i] = s
    return out


def top_k_docs(scores: list[int], k: int) -> list[int]:
    idx = list(range(len(scores)))
    idx.sort(key=lambda i: scores[i], reverse=True)
    return idx[: max(0, k)]
