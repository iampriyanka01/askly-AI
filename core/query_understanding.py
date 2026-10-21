"""
core/query_understanding.py  -  INTENT DETECTION + FOLLOW-UP RESOLUTION
======================================================================

Role in the pipeline
    Rigid helpdesk bots fail on follow-ups: "And what about sick leave?" means
    nothing to a search engine on its own. Before retrieval, Askly works out

        intent            greeting | kb_question | escalation_request | thanks | out_of_scope
        is_follow_up      does this turn depend on the conversation so far?
        standalone_query  the question rewritten to stand alone, e.g.
                          "How do I apply for sick leave?"
        category_hint     hr / it / security / finance / customer / onboarding

    The standalone query - not the raw text - is what gets retrieved.

Two implementations
    LLM        one structured JSON call with the recent conversation
    heuristic  regex intents + follow-up cues ("and", "what about", pronouns,
               very short turns) merged with the previous question. Used offline
               and whenever the LLM reply cannot be parsed.

Public API
    QueryAnalyzer().analyze(question, history) -> QueryAnalysis
        history: [{"role": "user"|"assistant", "content": str, "standalone": str?}, ...]
"""

import re
from typing import List, Dict, Optional
from utils.schemas import QueryAnalysis
from utils.utils import get_logger, get_llm, llm_text, parse_json_block

logger = get_logger(__name__)

_GREETING = re.compile(r"^\s*(hi|hello|hey|hiya|good (morning|afternoon|evening)|greetings|namaste|yo)\b[\s!.,]*(there|askly|team)?[\s!.,]*$", re.I)
_THANKS = re.compile(r"^\s*(thanks|thank you|thx|ty|cheers|great|perfect|awesome|got it|ok(ay)?)\b[\s\w!.,]{0,25}$", re.I)
_ESCALATE = re.compile(r"\b(talk|speak|chat|connect)\b.{0,20}\b(human|agent|person|someone|representative|support team|staff)\b"
                       r"|\bescalate\b|\b(raise|open|create|log)\b.{0,10}\bticket\b|\blive agent\b|\breal person\b", re.I)
# "How do I escalate a ticket?" / "What is the escalation process?" are QUESTIONS about the
# documented process, not requests for a human - unless they ask for a person explicitly.
_HOW_QUESTION = re.compile(r"^\s*(how|what|when|where|why|which|who|is there|are there|does|do)\b", re.I)
_WANTS_PERSON = re.compile(r"\b(human|agent|person|someone|representative|real person|live agent)\b", re.I)
_FOLLOW_UP_START = re.compile(r"^\s*(and|also|what about|how about|and what|what if|same for|or|but|then|so)\b", re.I)
_PRONOUN = re.compile(r"\b(it|that|this|those|these|they|them|there|its|one)\b", re.I)

CATEGORY_KEYWORDS = {
    "hr": ["leave", "holiday", "sick", "maternity", "paternity", "parental", "payroll", "salary", "payslip",
           "benefit", "insurance", "probation", "resign", "notice period", "performance review", "remote",
           "hybrid", "work from", "bereavement", "sabbatical", "overtime", "grievance"],
    "it": ["vpn", "wifi", "wi-fi", "laptop", "outlook", "email", "printer", "print", "teams", "software",
           "install", "monitor", "keyboard", "slow", "mailbox", "computer", "zoom", "remote desktop",
           "backup", "recover", "device", "phone"],
    "security": ["password", "2fa", "mfa", "two-factor", "two factor", "authenticator", "phishing", "locked out",
                 "lockout", "hacked", "compromised", "backup code", "login", "log in", "sign in"],
    "finance": ["expense", "reimburse", "receipt", "travel", "hotel", "flight", "per diem",
                "mileage", "corporate card", "invoice", "budget", "purchase order", "pay date", "payroll",
                "timesheet", "tax"],
    "customer": ["refund", "billing", "subscription", "pricing", "cancel", "export", "my data", "trial",
                 "sla", "response time", "resolution time", "platinum", "gold tier", "silver", "bronze",
                 "csat", "nps", "customer", "escalation matrix", "support ticket"],
    "operations": ["office", "facilit", "parking", "badge", "building", "fire", "evacuat", "emergency",
                   "first aid", "vendor", "procurement", "visitor", "shuttle", "business continuity"],
    "engineering": ["api", "endpoint", "database", "deploy", "rollback", "kubernetes", "aws", "monitoring",
                    "log level", "alert", "schema", "rate limit", "microservice", "staging"],
    "onboarding": ["new hire", "first day", "onboarding", "new starter", "joining", "mandatory training", "buddy"],
}


def detect_category(text: str) -> Optional[str]:
    text = (text or "").lower()
    best, hits = None, 0
    for cat, words in CATEGORY_KEYWORDS.items():
        n = sum(1 for w in words if re.search(rf"\b{re.escape(w)}", text))
        if n > hits:
            best, hits = cat, n
    return best


class QueryAnalyzer:
    """Classify the user's turn and rewrite follow-ups into standalone questions."""

    def __init__(self, use_llm: bool = True):
        self.llm = get_llm(temperature=0.0) if use_llm else None

    def analyze(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> QueryAnalysis:
        history = history or []
        question = (question or "").strip()
        quick = self._quick_intent(question)
        if quick:                       # greetings / thanks / "talk to a human" need no LLM call
            return QueryAnalysis(intent=quick, standalone_query=question, method="heuristic",
                                 category_hint=detect_category(self._previous_user_query(history) or ""))
        if self.llm and history:        # rewriting only matters when there IS a conversation
            result = self._analyze_with_llm(question, history)
            if result:
                return result
        return self._analyze_heuristic(question, history)

    # ------------------------------------------------------------------ heuristics

    @staticmethod
    def _quick_intent(question: str) -> Optional[str]:
        if _GREETING.match(question):
            return "greeting"
        if _ESCALATE.search(question) and (not _HOW_QUESTION.match(question) or _WANTS_PERSON.search(question)):
            return "escalation_request"
        if _THANKS.match(question) and len(question.split()) <= 5 and "?" not in question:
            return "thanks"
        return None

    @staticmethod
    def _previous_user_query(history: List[Dict[str, str]]) -> Optional[str]:
        for msg in reversed(history):
            if msg.get("role") == "user":
                return msg.get("standalone") or msg.get("content")
        return None

    def _analyze_heuristic(self, question: str, history: List[Dict[str, str]]) -> QueryAnalysis:
        prev = self._previous_user_query(history)
        words = re.findall(r"\w+", question)
        follow = bool(prev) and (
            bool(_FOLLOW_UP_START.match(question))
            or (len(words) <= 6 and bool(_PRONOUN.search(question)))
            or len(words) <= 3
        )
        standalone = question
        if follow:
            core = _FOLLOW_UP_START.sub("", question).strip(" ,?") or question
            # Keep the new topic first so it dominates retrieval; the earlier question
            # supplies the missing action ("apply for ...", "reset ...").
            standalone = f"{core} (follow-up to: {prev})"
        category = detect_category(question) or (detect_category(prev) if follow else None)
        return QueryAnalysis(intent="kb_question", is_follow_up=follow, standalone_query=standalone,
                             category_hint=category, method="heuristic")

    # ------------------------------------------------------------------ LLM

    def _analyze_with_llm(self, question: str, history: List[Dict[str, str]]) -> Optional[QueryAnalysis]:
        convo = "\n".join(f"{m['role'].upper()}: {m['content'][:400]}" for m in history[-6:])
        prompt = f"""You are the query-understanding step of an enterprise helpdesk assistant.

Conversation so far:
{convo}

New user message: "{question}"

Decide:
- intent: one of "kb_question" (a question the company documentation could answer),
  "escalation_request" (wants a human/ticket), "greeting", "thanks", "out_of_scope"
  (chit-chat or unrelated to work/company/product).
- is_follow_up: true if the message depends on the conversation to be understood.
- standalone_query: the message rewritten as a complete, self-contained question that keeps
  the user's intent (resolve "it", "that", "what about X" using the conversation).
  If it is not a follow-up, repeat the message unchanged.
- category: one of hr, it, security, finance, customer, onboarding, or null.

Return ONLY JSON: {{"intent": "...", "is_follow_up": true/false, "standalone_query": "...", "category": "..."}}"""
        try:
            data = parse_json_block(llm_text(self.llm.invoke(prompt)))
            if not isinstance(data, dict) or not data.get("standalone_query"):
                return None
            intent = data.get("intent") if data.get("intent") in (
                "greeting", "kb_question", "escalation_request", "thanks", "out_of_scope") else "kb_question"
            cat = data.get("category")
            cat = cat if cat in CATEGORY_KEYWORDS else detect_category(data["standalone_query"])
            return QueryAnalysis(intent=intent, is_follow_up=bool(data.get("is_follow_up")),
                                 standalone_query=str(data["standalone_query"]).strip(),
                                 category_hint=cat, method="llm")
        except Exception as e:
            logger.warning(f"LLM query analysis failed, using heuristics: {e}")
            return None
