"""
core/feedback.py  -  USER FEEDBACK LOOP ("users rate answers -> Askly learns")
==============================================================================

Every assistant answer can be rated 👍 / 👎 (optionally with a comment). Ratings
are stored in data/state/feedback.jsonl together with the chunk ids the answer
was grounded in, and feed back into Askly in two ways:

  1. Retrieval boosts  - chunk_boosts() turns the ratings into a small multiplier
     per chunk (+/-30% at most, smoothed so one vote moves little). HybridIndexer
     applies it to the fused score, so chunks that keep producing helpful answers
     rise and chunks behind unhelpful ones sink - without retraining anything.
  2. Gap signals       - a 👎 is logged as a knowledge-gap candidate by the
     pipeline, so documentation owners see which answers people disliked.

Public API
    FeedbackStore.record(...) / all() / stats() / chunk_boosts()
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional
from utils.config import STATE_DIR
from utils.schemas import FeedbackRecord
from utils.utils import get_logger, append_jsonl, read_jsonl

logger = get_logger(__name__)

MAX_BOOST = 0.30
SMOOTHING = 2.0   # pseudo-votes: a single 👍 gives +0.10, not +0.30


class FeedbackStore:
    def __init__(self, state_dir: str = STATE_DIR):
        self.path = Path(state_dir) / "feedback.jsonl"
        self._cache: Optional[Dict[str, float]] = None

    def record(self, session_id: str, turn_id: str, question: str, answer: str, rating: int,
               chunk_ids: List[str], comment: Optional[str] = None) -> FeedbackRecord:
        rec = FeedbackRecord(session_id=session_id, turn_id=turn_id, question=question, answer=answer,
                             rating=1 if rating > 0 else -1, comment=comment or None, chunk_ids=chunk_ids)
        append_jsonl(self.path, rec.model_dump())
        self._cache = None
        return rec

    def all(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.path)

    def chunk_boosts(self) -> Dict[str, float]:
        """{chunk_id: boost in [-0.3, 0.3]} - the latest rating per (turn, chunk) counts once."""
        if self._cache is not None:
            return self._cache
        latest: Dict[tuple, int] = {}
        for r in self.all():
            for cid in r.get("chunk_ids", []):
                latest[(r.get("turn_id"), cid)] = r.get("rating", 0)
        votes: Dict[str, List[int]] = defaultdict(list)
        for (_, cid), rating in latest.items():
            votes[cid].append(rating)
        self._cache = {cid: max(-MAX_BOOST, min(MAX_BOOST, MAX_BOOST * sum(v) / (len(v) + SMOOTHING)))
                       for cid, v in votes.items()}
        return self._cache

    def stats(self) -> Dict[str, Any]:
        rows = self.all()
        up = sum(1 for r in rows if r.get("rating") == 1)
        down = sum(1 for r in rows if r.get("rating") == -1)
        return {"total": len(rows), "helpful": up, "not_helpful": down,
                "helpful_rate": round(up / len(rows), 3) if rows else None,
                "comments": [r for r in rows if r.get("comment")][-20:]}
