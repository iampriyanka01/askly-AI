"""
core/re_ranker.py  -  STAGE 2 of retrieval: RERANKING candidates by true relevance
==================================================================================

Hybrid retrieval is tuned for RECALL: it returns ~12 plausible chunks. Only the
best few should reach the LLM - extra, off-topic context makes answers longer,
slower and less faithful. The reranker reads each (question, chunk) pair and
assigns a relevance in 0-1, which is also the signal the gap detector uses to
decide "the documentation does not cover this".

Strategies (RERANKER in .env)
    "auto" (default) = cross-encoder if RERANKER_MODEL is set and installed, else lexical.
    LLM reranking is opt-in (RERANKER=llm) because it costs one extra LLM call per
    question - on a 5-requests/minute free tier that is the difference between a
    ~15 s and a ~30 s answer.

    cross-encoder : a dedicated reranking model, e.g.
                    cross-encoder/ms-marco-MiniLM-L-6-v2 (pip install sentence-transformers,
                    set RERANKER_MODEL). Local, fast, no API cost.
    llm           : ONE LLM call scores all candidates 0-10 as JSON. The reply is
                    accepted only if it scores every candidate; otherwise lexical.
    lexical       : IDF-weighted coverage of the question's terms in the chunk body
                    and heading, blended with the hybrid rank and (if available) the
                    dense cosine. Free, deterministic, works offline.

Public API
    ReRanker(hybrid_indexer).rerank(query, candidates, top_n, trace) -> list (relevance added)
    ReRanker.method  -> which strategy is active
"""

import math
from typing import List, Dict, Any, Optional
from utils.config import RERANKER, RERANKER_MODEL
from utils.utils import get_logger, get_llm, llm_text, parse_json_block, tokenize, query_term_weights

logger = get_logger(__name__)


class ReRanker:
    """Scores (question, chunk) relevance and keeps the top_n chunks."""

    def __init__(self, hybrid_indexer=None, strategy: str = RERANKER):
        self.indexer = hybrid_indexer
        self.cross_encoder = None
        self.llm = None
        self.method = "lexical"
        strategy = (strategy or "auto").lower()

        if strategy in ("auto", "cross-encoder", "cross_encoder") and RERANKER_MODEL:
            try:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder(RERANKER_MODEL)
                self.method = "cross-encoder"
            except Exception as e:
                logger.warning(f"Cross-encoder '{RERANKER_MODEL}' unavailable ({e}); falling back.")
        if self.method == "lexical" and strategy == "llm":
            self.llm = get_llm(temperature=0.0)
            if self.llm:
                self.method = "llm"

    # ------------------------------------------------------------------ public

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 4,
               trace: Optional[Dict[str, Any]] = None, persona: Optional[str] = None) -> List[Dict[str, Any]]:
        """Score, sort and keep the best `top_n` candidates.

        persona: "employee" / "customer". Sections written for one audience
        ("Customer Password Reset" vs "Employee Password Reset") get a small nudge
        (x1.08 / x0.92) for the matching / other audience - personalisation that only
        breaks near-ties, never overrides a clearly better passage.
        """
        if not candidates:
            if trace is not None:
                trace["rerank"] = {"method": self.method, "note": "no candidates"}
            return []
        candidates = [dict(c) for c in candidates]
        lexical = self._lexical_scores(query, candidates)
        for c, s in zip(candidates, lexical):
            c["lexical_relevance"] = round(s, 3)

        used, scores, note = self.method, None, ""
        if self.method == "cross-encoder":
            scores = self._cross_encoder_scores(query, candidates)
        elif self.method == "llm":
            scores, note = self._llm_scores(query, candidates)
        if scores is None:
            if used != "lexical":
                note = note or f"{used} failed"
            used, scores = "lexical", lexical
        other = {"employee": "customer", "customer": "employee"}.get((persona or "").lower())
        for c, s in zip(candidates, scores):
            heading = (c.get("metadata", {}).get("section_title") or "").lower()
            if other and persona.lower() in heading:
                s = min(1.0, float(s) * 1.08)
            elif other and other in heading:
                s = float(s) * 0.92
            c["relevance"] = round(float(s), 3)
        ranked = sorted(candidates, key=lambda c: (c["relevance"], c.get("combined_score", 0)), reverse=True)
        if trace is not None:
            trace["rerank"] = {
                "method": used, "note": note,
                "scores": [{"chunk": f"{c['metadata'].get('title')} · {c['metadata'].get('section')}",
                            "relevance": c["relevance"], "lexical": c["lexical_relevance"]} for c in ranked[:8]],
                "kept": top_n,
            }
        return ranked[:top_n]

    # ------------------------------------------------------------------ strategies

    def _idf(self, token: str) -> float:
        return self.indexer.idf(token) if self.indexer else 1.0

    def _lexical_scores(self, query: str, candidates: List[Dict[str, Any]]) -> List[float]:
        weights = query_term_weights(query, self._idf)
        if not weights:
            return [0.0] * len(candidates)
        total = sum(weights.values())
        max_rrf = max((c.get("combined_score", 0) for c in candidates), default=0) or 1.0
        out = []
        for c in candidates:
            md = c.get("metadata", {})
            body = set(tokenize(c.get("text", "")))
            heading = set(tokenize(md.get("section_title", "")))     # the chunk's own heading
            coverage = sum(w for t, w in weights.items() if t in body or t in heading) / total
            head_cov = sum(w for t, w in weights.items() if t in heading) / total
            rank_prior = c.get("combined_score", 0) / max_rrf
            # coverage**1.5 punishes partial matches: a chunk that shares only one of
            # two rare words with "how do I get my money back" is probably about something
            # else, and should fall below the gap threshold rather than be answered from.
            score = 0.70 * coverage ** 1.5 + 0.18 * head_cov + 0.12 * rank_prior
            cos = c.get("vector_score", 0.0)
            if cos > 0:   # dense signal available: cosine ~0.5 (unrelated) .. ~0.9 (same topic)
                score = 0.75 * score + 0.25 * min(max((cos - 0.5) / 0.4, 0.0), 1.0)
            out.append(min(score, 1.0))
        return out

    def _cross_encoder_scores(self, query: str, candidates: List[Dict[str, Any]]) -> Optional[List[float]]:
        try:
            logits = self.cross_encoder.predict([(query, c["text"][:2000]) for c in candidates])
            return [1 / (1 + math.exp(-float(x))) for x in logits]
        except Exception as e:
            logger.warning(f"Cross-encoder rerank failed: {e}")
            return None

    def _llm_scores(self, query: str, candidates: List[Dict[str, Any]]):
        passages = "\n\n".join(f"[{i}] {c['text'][:700]}" for i, c in enumerate(candidates, start=1))
        prompt = f"""You are reranking passages from an enterprise knowledge base for a helpdesk.
Score how useful each passage is for answering the question, from 0 (irrelevant) to 10
(directly answers it). Be strict: a passage on a related but different topic scores <= 4.

Question: "{query}"

Passages:
{passages}

Return ONLY JSON: {{"scores": [{{"id": 1, "score": 7}}, ...]}} with one entry per passage."""
        try:
            data = parse_json_block(llm_text(self.llm.invoke(prompt)))
            items = data.get("scores", []) if isinstance(data, dict) else (data or [])
            by_id = {int(it["id"]): float(it["score"]) for it in items if "id" in it and "score" in it}
            if set(by_id) != set(range(1, len(candidates) + 1)):
                return None, f"LLM scored {len(by_id)}/{len(candidates)} passages - reply rejected"
            return [min(max(by_id[i] / 10.0, 0.0), 1.0) for i in range(1, len(candidates) + 1)], ""
        except Exception as e:
            logger.warning(f"LLM rerank failed: {e}")
            return None, f"LLM rerank error: {e}"
