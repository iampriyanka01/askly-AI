"""Multi-query expansion for Askly ("deep" retrieval).

Helpdesk questions are often phrased very differently from the policy text
that answers them ("my laptop won't connect to the office network" vs. a
section titled "VPN troubleshooting"). Generating a few alternative phrasings
and retrieving for each raises recall; the results are merged with Reciprocal
Rank Fusion in HybridIndexer.

The user's own wording is ALWAYS kept as the first query (lesson from HireFlow
HW19), and numbering/bullets the LLM adds are stripped.
"""

import re
from typing import List
from utils.utils import get_logger, get_llm, llm_text

logger = get_logger(__name__)

# Domain synonyms used when no LLM is available.
_SYNONYMS = {
    "password": ["login credentials", "sign in"],
    "reset": ["forgot", "recover"],
    "leave": ["time off", "absence"],
    "sick": ["illness", "medical certificate"],
    "vpn": ["remote network access", "vpn client"],
    "vacation": ["pto", "paid time off"],
    "pto": ["vacation", "annual leave"],
    "wifi": ["wireless network"],
    "expense": ["reimbursement", "claim"],
    "refund": ["money back", "cancellation"],
    "2fa": ["two-factor authentication", "mfa"],
    "mfa": ["two-factor authentication", "authenticator app"],
    "laptop": ["device", "computer"],
    "salary": ["payroll", "payslip"],
    "invoice": ["billing", "receipt"],
}


class MultiQueryGenerator:
    """Generate alternative phrasings of a helpdesk question."""

    def __init__(self):
        self.llm = get_llm(temperature=0.4)

    def generate_queries(self, original_query: str, num_queries: int = 3) -> List[str]:
        """Return `num_queries` queries; the first one is always the original."""
        original_query = (original_query or "").strip()
        if not original_query:
            return [original_query]
        queries: List[str] = []
        if self.llm:
            queries = self._generate_with_llm(original_query, num_queries - 1)
        if not queries:
            queries = self._generate_fallback(original_query, num_queries - 1)
        out = [original_query]
        for q in queries:
            if q.lower() != original_query.lower() and q.lower() not in (o.lower() for o in out):
                out.append(q)
        return out[:num_queries]

    def _generate_with_llm(self, query: str, n: int) -> List[str]:
        prompt = f"""You help an enterprise helpdesk search its internal documentation
(HR policies, IT troubleshooting wikis, security policies, finance policies, customer FAQs).

Rewrite the employee/customer question below into {n} alternative search queries that use
the vocabulary such documents typically use (policy names, section titles, product terms).
Return ONLY the queries, one per line, no numbering.

Question: "{query}"
"""
        try:
            lines = llm_text(self.llm.invoke(prompt)).strip().splitlines()
            cleaned = []
            for line in lines:
                line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip().strip('"')
                if len(line) > 3:
                    cleaned.append(line)
            return cleaned[:n]
        except Exception as e:
            logger.warning(f"Multi-query LLM generation failed, using fallback: {e}")
            return []

    @staticmethod
    def _generate_fallback(query: str, n: int) -> List[str]:
        """Rule-based expansion with a small helpdesk synonym table."""
        words = re.findall(r"[a-z0-9]+", query.lower())
        expansions = []
        for w in words:
            for syn in _SYNONYMS.get(w, []):
                expansions.append(re.sub(rf"\b{re.escape(w)}\b", syn, query, flags=re.IGNORECASE))
        return expansions[:n]
