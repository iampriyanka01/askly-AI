"""
core/search_router.py  -  THE RETRIEVAL "FRONT DOOR"
=====================================================

Every knowledge-base lookup goes through SearchRouter.search(). It decides how
much work a question deserves (same idea as HireFlow's shallow/deep router):

    standalone query ──> mode? ──┬─ "shallow": hybrid (BM25 + dense, RRF) ─> rerank
                                 └─ "deep"   : multi-query expansion ─> hybrid for
                                               every phrasing ─> RRF ─> rerank
        "auto" (default): run shallow first; if the best reranked chunk is weak
        (relevance < GAP_THRESHOLD + 0.15) AND an LLM is available to generate
        better phrasings, escalate to deep. Easy questions cost nothing extra;
        oddly phrased ones get a second, wider search before Askly gives up.

Personalisation
    category_boosts gently favour the documents most relevant to the user
    (customers -> customer FAQ + account security; employees -> internal policies)
    and the category the query analyser detected. They multiply the fused score;
    they never hide a document.

Output of search()
    {"results": top_n reranked chunks (each with 'relevance' 0-1),
     "candidates": all fused candidates before reranking,
     "search_type": "hybrid" | "hybrid_multi_query" | "bm25_only" | "bm25_multi_query",
     "queries": [phrasings used], "top_relevance": float}
"""

from typing import Dict, Any, List, Optional
from core.hybrid_indexer import HybridIndexer
from core.re_ranker import ReRanker
from utils.config import RETRIEVAL_TOP_K, RERANK_TOP_N, GAP_THRESHOLD
from utils.multi_query import MultiQueryGenerator
from utils.utils import get_logger

logger = get_logger(__name__)

PERSONA_BOOSTS = {
    "customer": {"customer": 1.25, "security": 1.10, "hr": 0.80, "finance": 0.80, "onboarding": 0.80, "it": 0.9},
    "employee": {"customer": 0.85},
}


class SearchRouter:
    """Chooses shallow vs deep retrieval and applies reranking."""

    def __init__(self, hybrid_indexer: HybridIndexer, reranker: Optional[ReRanker] = None,
                 multi_query: Optional[MultiQueryGenerator] = None):
        self.indexer = hybrid_indexer
        self.reranker = reranker or ReRanker(hybrid_indexer)
        self.multi_query = multi_query or MultiQueryGenerator()

    @staticmethod
    def category_boosts(user_type: Optional[str], category_hint: Optional[str]) -> Dict[str, float]:
        boosts = dict(PERSONA_BOOSTS.get((user_type or "").lower(), {}))
        if category_hint:
            boosts[category_hint] = boosts.get(category_hint, 1.0) * 1.15
        return boosts

    def _run(self, queries: List[str], top_k: int, top_n: int, filters, boosts, feedback_boosts, trace,
             persona: Optional[str] = None):
        candidates = self.indexer.search(queries, top_k=top_k, filters=filters, category_boosts=boosts,
                                         feedback_boosts=feedback_boosts, trace=trace)
        # The reranker always judges against the user's (standalone) question, not the
        # LLM's paraphrases - paraphrases are for recall, not for deciding relevance.
        results = self.reranker.rerank(queries[0], candidates, top_n=top_n, trace=trace, persona=persona)
        return candidates, results

    def search(self, query: str, mode: str = "auto", top_k: int = RETRIEVAL_TOP_K, top_n: int = RERANK_TOP_N,
               filters: Optional[Dict[str, Any]] = None, user_type: Optional[str] = None,
               category_hint: Optional[str] = None, feedback_boosts: Optional[Dict[str, float]] = None,
               trace: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        mode = (mode or "auto").lower()
        boosts = self.category_boosts(user_type, category_hint)
        dense = self.indexer.vector_store.is_ready()
        base = "hybrid" if dense else "bm25_only"
        shallow_trace = {} if trace is not None else None

        queries = [query]
        if mode == "deep":
            queries = self.multi_query.generate_queries(query, 3)
        candidates, results = self._run(queries, top_k, top_n, filters, boosts, feedback_boosts, shallow_trace,
                                        user_type)
        search_type = base if len(queries) == 1 else ("hybrid_multi_query" if dense else "bm25_multi_query")
        top = results[0]["relevance"] if results else 0.0
        escalated_to_deep = False

        if mode == "auto" and top < GAP_THRESHOLD + 0.15 and self.multi_query.llm is not None:
            deep_trace = {} if trace is not None else None
            deep_queries = self.multi_query.generate_queries(query, 3)
            if len(deep_queries) > 1:
                d_cands, d_results = self._run(deep_queries, top_k, top_n, filters, boosts, feedback_boosts,
                                               deep_trace, user_type)
                d_top = d_results[0]["relevance"] if d_results else 0.0
                if d_top >= top:
                    candidates, results, top, queries = d_cands, d_results, d_top, deep_queries
                    search_type = "hybrid_multi_query" if dense else "bm25_multi_query"
                    shallow_trace = deep_trace if deep_trace is not None else shallow_trace
                escalated_to_deep = True

        if trace is not None:
            trace.update(shallow_trace or {})
            trace["router"] = {"mode_requested": mode, "search_type": search_type, "queries": queries,
                               "escalated_to_deep": escalated_to_deep, "category_boosts": boosts,
                               "filters": filters or {}, "top_relevance": round(top, 3)}
        return {"results": results, "candidates": candidates, "search_type": search_type,
                "queries": queries, "top_relevance": top}
