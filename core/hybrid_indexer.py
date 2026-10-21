"""
core/hybrid_indexer.py  -  HYBRID RETRIEVAL (BM25 sparse + dense vectors, merged with RRF)
=========================================================================================

Why hybrid?
    - Sparse / BM25 is precise on exact terms people type into a helpdesk:
      product names, "2FA", "PTO", "SOX", section numbers, error messages.
    - Dense / embeddings understand meaning: "I can't get into my account"
      ~ "password reset"; "time off when I'm ill" ~ "sick leave".
    Running both and fusing the rankings beats either one alone.

How a query flows through search()
    queries ─┬─> BM25 over all chunks        ─> ranking #1 (per query)
             └─> VectorStore (FAISS/Pinecone) ─> ranking #2 (per query)
                          │
                          └─> Reciprocal Rank Fusion: score = Σ 1/(60 + rank)
                              × personalisation / feedback multipliers
                          └─> top_k candidates for the reranker

    RRF fuses RANK POSITIONS, not raw scores - BM25 scores are unbounded while
    cosine similarity sits in 0-1, so adding them directly would let one engine
    drown out the other (HireFlow HW7).

Multipliers (soft, never filters)
    category_boosts : e.g. {"customer": 1.2} for a customer, or the category the
                      query analyser detected ("hr" for a leave question)
    feedback_boosts : {chunk_id: +/-x} learned from thumbs up/down (core/feedback.py)

Public API
    index(chunks)                  -> build BM25 + sync the vector store
    add(chunks)                    -> merge new chunks (uploads) without a rebuild
    search(queries, top_k, filters, category_boosts, feedback_boosts, trace)
    idf(token) / get_index_stats()
"""

import math
from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from core.vector_store import VectorStore, _matches
from utils.utils import get_logger, tokenize

logger = get_logger(__name__)

RRF_K = 60


def _bm25_text(doc: Document) -> List[str]:
    """Tokens for BM25: title and section heading are repeated so a match on the
    section heading ("Sick Leave") outweighs a passing mention in another chunk."""
    md = doc.metadata
    return (tokenize(doc.page_content) + tokenize(md.get("section_title", "")) * 2
            + tokenize(md.get("parent_section", "")) + tokenize(md.get("title", "")))


class HybridIndexer:
    """Owns the BM25 index and coordinates it with the shared VectorStore."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        self.chunks: List[Document] = []
        self.by_id: Dict[str, Document] = {}
        self.bm25: Optional[BM25Okapi] = None
        self._corpus_tokens: List[List[str]] = []

    # ------------------------------------------------------------------ indexing

    def index(self, chunks: List[Document]) -> Dict[str, Any]:
        """(Re)build BM25 over `chunks` and sync the vector store to them."""
        self.chunks = [c for c in chunks if c.page_content.strip()]
        self.by_id = {c.metadata["chunk_id"]: c for c in self.chunks}
        self._build_bm25()
        sync = self.vector_store.sync(self.chunks) if self.vector_store.is_ready() else None
        return {"chunks": len(self.chunks), "vector_sync": sync}

    def add(self, chunks: List[Document]) -> Dict[str, Any]:
        """Merge new/updated chunks (e.g. an uploaded document) by chunk_id."""
        for c in chunks:
            self.by_id[c.metadata["chunk_id"]] = c
        self.chunks = list(self.by_id.values())
        self._build_bm25()
        sync = self.vector_store.add_chunks(chunks) if self.vector_store.is_ready() else None
        return {"chunks": len(self.chunks), "vector_sync": sync}

    def remove_document(self, doc_id: str) -> int:
        removed = [cid for cid, c in self.by_id.items() if c.metadata.get("doc_id") == doc_id]
        for cid in removed:
            self.by_id.pop(cid)
        self.chunks = list(self.by_id.values())
        self._build_bm25()
        if removed and self.vector_store.is_ready():
            self.vector_store.sync(self.chunks)
        return len(removed)

    def _build_bm25(self) -> None:
        self._corpus_tokens = [_bm25_text(c) for c in self.chunks]
        self.bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def idf(self, token: str) -> float:
        """IDF of a (tokenized) term; unseen terms get the maximum IDF (they are
        maximally informative - and their absence signals a knowledge gap)."""
        if not self.bm25:
            return 1.0
        n = len(self._corpus_tokens)
        return float(self.bm25.idf.get(token, math.log((n + 0.5) / 0.5)))

    # ------------------------------------------------------------------ search

    def _bm25_ranking(self, query: str, filters: Optional[Dict[str, Any]], limit: int):
        tokens = tokenize(query)
        if not self.bm25 or not tokens:
            return tokens, []
        scores = self.bm25.get_scores(tokens)
        hits = [(i, s) for i, s in enumerate(scores)
                if s > 0 and _matches(self.chunks[i].metadata, filters)]   # no shared word -> not a hit
        hits.sort(key=lambda p: p[1], reverse=True)
        return tokens, [(self.chunks[i].metadata["chunk_id"], float(s)) for i, s in hits[:limit]]

    def search(self, queries: Union[str, List[str]], top_k: int = 12,
               filters: Optional[Dict[str, Any]] = None,
               category_boosts: Optional[Dict[str, float]] = None,
               feedback_boosts: Optional[Dict[str, float]] = None,
               trace: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Hybrid search for one or more phrasings of the question.

        Returns candidate dicts: chunk_id, text, metadata, combined_score, rrf_score,
        bm25_score, vector_score, bm25_rank, vector_rank, matched_by.
        """
        if isinstance(queries, str):
            queries = [queries]
        queries = [q for q in queries if q and q.strip()]
        if not queries or not self.chunks:
            return []

        merged: Dict[str, Dict[str, Any]] = {}

        def entry(cid: str, text: str = "", metadata: Optional[Dict] = None) -> Dict[str, Any]:
            if cid not in merged:
                doc = self.by_id.get(cid)
                merged[cid] = {"chunk_id": cid,
                               "text": doc.page_content if doc else text,
                               "metadata": dict(doc.metadata) if doc else dict(metadata or {}),
                               "rrf_score": 0.0, "bm25_score": 0.0, "vector_score": 0.0,
                               "bm25_rank": None, "vector_rank": None, "matched_by": set()}
            return merged[cid]

        per_query_trace = []
        dense_ready = self.vector_store.is_ready()
        for q in queries:
            tokens, bm25_hits = self._bm25_ranking(q, filters, limit=top_k * 2)
            for rank, (cid, score) in enumerate(bm25_hits, start=1):
                e = entry(cid)
                e["rrf_score"] += 1.0 / (RRF_K + rank)
                e["bm25_score"] = max(e["bm25_score"], score)
                e["bm25_rank"] = min(e["bm25_rank"] or rank, rank)
                e["matched_by"].add("bm25")
            dense_hits = self.vector_store.search(q, top_k * 2, filters) if dense_ready else []
            for rank, hit in enumerate(dense_hits, start=1):
                cid = hit.get("chunk_id") or hit["metadata"].get("chunk_id")
                if not cid:
                    continue
                e = entry(cid, hit.get("page_content", ""), hit.get("metadata"))
                e["rrf_score"] += 1.0 / (RRF_K + rank)
                e["vector_score"] = max(e["vector_score"], hit.get("score", 0.0))
                e["vector_rank"] = min(e["vector_rank"] or rank, rank)
                e["matched_by"].add("dense")
            if trace is not None:
                per_query_trace.append({
                    "query": q, "bm25_tokens": tokens,
                    "bm25_top": [{"chunk": self._label(cid), "score": round(s, 3)} for cid, s in bm25_hits[:6]],
                    "dense_top": [{"chunk": self._label(h.get("chunk_id")), "cosine": round(h.get("score", 0), 3)}
                                  for h in dense_hits[:6]],
                })

        for e in merged.values():
            mult = 1.0
            if category_boosts:
                mult *= category_boosts.get(e["metadata"].get("category"), 1.0)
            if feedback_boosts:
                mult *= 1.0 + feedback_boosts.get(e["chunk_id"], 0.0)
            e["boost"] = round(mult, 3)
            e["combined_score"] = e["rrf_score"] * mult
            e["matched_by"] = sorted(e["matched_by"])

        results = sorted(merged.values(), key=lambda r: r["combined_score"], reverse=True)[:top_k]
        if trace is not None:
            trace["retrieval"] = {
                "dense_enabled": dense_ready,
                "dense_status": self.vector_store.status_message,
                "queries": per_query_trace,
                "fusion": f"Reciprocal Rank Fusion (k={RRF_K}) over {len(queries)} quer"
                          f"{'y' if len(queries) == 1 else 'ies'} x {'2 engines' if dense_ready else 'BM25 only'}",
                "fused_top": [{"chunk": self._label(r["chunk_id"]), "rrf": round(r["rrf_score"], 4),
                               "boost": r["boost"], "matched_by": r["matched_by"]} for r in results[:8]],
            }
        return results

    def _label(self, cid: Optional[str]) -> str:
        doc = self.by_id.get(cid or "")
        if not doc:
            return str(cid)
        md = doc.metadata
        return f"{md.get('title')} · {md.get('section')}"

    def get_index_stats(self) -> Dict[str, Any]:
        docs = {c.metadata.get("doc_id") for c in self.chunks}
        return {"chunks": len(self.chunks), "documents": len(docs), "bm25_ready": self.bm25 is not None,
                "dense": self.vector_store.get_stats(),
                "hybrid_ready": self.bm25 is not None and self.vector_store.is_ready()}
