"""
core/gap_detector.py  -  KNOWLEDGE-GAP DETECTION + INTELLIGENT ESCALATION
=========================================================================

Problem it solves
    "When documentation is incomplete, systems fail silently" - or worse, the
    LLM fills the hole with a confident guess. Askly checks twice:

      1. BEFORE generation  - retrieval confidence: is the best reranked chunk
                              relevant enough (>= GAP_THRESHOLD)?
      2. AFTER generation   - the LLM itself reports answerable=false when the
                              context does not actually contain the answer.

    Either way Askly says it doesn't know, logs the gap for the documentation
    owners, and offers (or automatically opens) a ticket with the right team.

When does Askly escalate on its own?
    - the user asked for a human ("talk to an agent", "raise a ticket")
    - the question is urgent / security-sensitive (locked out, stolen, hacked...)
    - it is the 2nd unanswered question in a row in the same conversation
    Otherwise it OFFERS escalation as a one-click follow-up, so the user decides.

Storage (data/state/)
    knowledge_gaps.jsonl   one line per detected gap  (reviewed by doc owners)
    tickets.jsonl          escalation tickets         (the human helpdesk queue)

Public API
    KnowledgeGapDetector.assess(results)            -> {confidence, is_gap, reason}
    KnowledgeGapDetector.log_gap(...) / list_gaps() / gap_summary()
    EscalationManager.create_ticket(...) / list_tickets() / update_status(...)
    EscalationManager.route(category) / priority_for(text) / should_auto_escalate(...)
"""

import re
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional
from utils.config import GAP_THRESHOLD, STATE_DIR
from utils.schemas import KnowledgeGap, EscalationTicket
from utils.utils import get_logger, append_jsonl, read_jsonl, write_jsonl

logger = get_logger(__name__)

# Category -> team that receives escalations. Adjust to your organisation (or replace
# create_ticket() with a call to Jira / ServiceNow / Zendesk).
TEAM_ROUTING = {
    "hr": "HR Team",
    "onboarding": "HR Onboarding Team",
    "it": "IT Service Desk",
    "security": "Information Security",
    "finance": "Finance Team",
    "customer": "Customer Success & Support",
    "operations": "Operations & Facilities",
    "engineering": "Platform Engineering",
}
DEFAULT_TEAM = "General Helpdesk"
_URGENT = re.compile(r"\b(urgent|asap|immediately|emergency|locked out|stolen|lost (my )?(phone|laptop)|hacked|"
                     r"compromised|breach|phishing|can'?t (log|sign) ?in|outage|down for everyone)\b", re.I)


class KnowledgeGapDetector:
    """Decides whether retrieved context is good enough to answer from."""

    def __init__(self, threshold: float = GAP_THRESHOLD, state_dir: str = STATE_DIR):
        self.threshold = threshold
        self.path = Path(state_dir) / "knowledge_gaps.jsonl"

    def assess(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Retrieval confidence = best rerank relevance, nudged up slightly when the
        2nd-best chunk agrees (same document), down when nothing was retrieved."""
        if not results:
            return {"confidence": 0.0, "is_gap": True, "reason": "no matching documentation found"}
        top = results[0].get("relevance", 0.0)
        agree = (len(results) > 1 and results[1].get("relevance", 0) >= self.threshold
                 and results[1]["metadata"].get("doc_id") == results[0]["metadata"].get("doc_id"))
        confidence = min(1.0, top + (0.05 if agree else 0.0))
        is_gap = top < self.threshold
        reason = (f"best match relevance {top:.2f} is below the {self.threshold:.2f} threshold"
                  if is_gap else f"best match relevance {top:.2f}")
        return {"confidence": round(confidence, 3), "is_gap": is_gap, "reason": reason}

    def log_gap(self, query: str, reason: str, top_relevance: float = 0.0,
                category_hint: Optional[str] = None, session_id: Optional[str] = None) -> KnowledgeGap:
        gap = KnowledgeGap(query=query, reason=reason, top_relevance=round(top_relevance, 3),
                           category_hint=category_hint, session_id=session_id)
        append_jsonl(self.path, gap.model_dump())
        logger.info(f"Knowledge gap logged: {query!r} ({reason})")
        return gap

    def list_gaps(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.path)

    def gap_summary(self, top: int = 10) -> List[Dict[str, Any]]:
        """Most frequent unanswered questions - the documentation team's to-do list."""
        counts = Counter(g["query"].strip().lower() for g in self.list_gaps())
        return [{"query": q, "count": n} for q, n in counts.most_common(top)]


class EscalationManager:
    """Creates and tracks hand-offs to human support teams."""

    def __init__(self, state_dir: str = STATE_DIR):
        self.path = Path(state_dir) / "tickets.jsonl"

    @staticmethod
    def route(category: Optional[str]) -> str:
        return TEAM_ROUTING.get(category or "", DEFAULT_TEAM)

    @staticmethod
    def priority_for(text: str) -> str:
        return "high" if _URGENT.search(text or "") else "normal"

    @staticmethod
    def should_auto_escalate(question: str, consecutive_gaps: int) -> bool:
        return bool(_URGENT.search(question or "")) or consecutive_gaps >= 2

    def _next_id(self) -> str:
        return f"ASK-{1001 + len(read_jsonl(self.path))}"

    def create_ticket(self, session_id: str, question: str, standalone_query: str, reason: str,
                      category: Optional[str] = None, conversation: Optional[List[dict]] = None,
                      best_sources: Optional[List[str]] = None, user_name: Optional[str] = None,
                      user_type: Optional[str] = None) -> EscalationTicket:
        ticket = EscalationTicket(
            ticket_id=self._next_id(), session_id=session_id, user_name=user_name, user_type=user_type,
            question=question, standalone_query=standalone_query, reason=reason,
            team=self.route(category), priority=self.priority_for(f"{question} {standalone_query}"),
            conversation=(conversation or [])[-10:], best_sources=best_sources or [])
        append_jsonl(self.path, ticket.model_dump())
        logger.info(f"Escalation ticket {ticket.ticket_id} -> {ticket.team} ({ticket.priority})")
        return ticket

    def list_tickets(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.path)

    def update_status(self, ticket_id: str, status: str) -> bool:
        rows = self.list_tickets()
        changed = False
        for r in rows:
            if r.get("ticket_id") == ticket_id:
                r["status"] = status
                changed = True
        if changed:
            write_jsonl(self.path, rows)
        return changed
