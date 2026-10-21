"""
core/rag_pipeline.py  -  THE ASKLY ASSISTANT (orchestrates one conversational turn)
===================================================================================

    user message
      │ 1. QueryAnalyzer      intent + follow-up resolution -> standalone query
      │ 2. SearchRouter       hybrid BM25 + dense retrieval -> RRF -> rerank
      │ 3. KnowledgeGap check is the best chunk relevant enough?   ── no ─┐
      │ 4. Generation         grounded answer with [n] citations         │
      │                       (LLM, or extractive sentences offline)     │
      │ 5. Answerable check   LLM says the context lacks the answer? ─ yes┤
      │ 6. Proactive layer    follow-up offers + related KB sections     │
      │ 7. Memory             store turn (feeds the next follow-up)      │
      ▼                                                                  ▼
    AskResponse(answer, citations, follow_ups, related, confidence)   "I don't know" +
                                                                       gap log + escalation

Grounding rules enforced in the prompt
    - answer ONLY from the numbered context passages, cite every fact as [n]
    - if the passages don't contain the answer -> answerable=false (never guess)
    - step-by-step instructions for "how do I" questions, names exact menu paths/links
    - tone and examples adapted to the user (employee vs. customer)

Offline mode (no API key)
    Generation falls back to EXTRACTIVE answers: the sentences of the top passages
    that best cover the question, each cited. Still grounded, never invented.

Public API
    AsklyAssistant.load_knowledge_base() / add_document(path, category)
    AsklyAssistant.ask(question, session_id, mode="auto", filters=None, debug=False) -> AskResponse
    AsklyAssistant.escalate(session_id, reason) / record_feedback(session_id, turn_id, rating, comment)
    AsklyAssistant.status()
"""

import json
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from core.ingestion import load_knowledge_base, load_file, build_catalog
from core.vector_store import VectorStore
from core.hybrid_indexer import HybridIndexer
from core.re_ranker import ReRanker
from core.search_router import SearchRouter
from core.query_understanding import QueryAnalyzer, detect_category
from core.gap_detector import KnowledgeGapDetector, EscalationManager
from core.feedback import FeedbackStore
from core.memory_rag import MemoryRAG
from utils.config import KNOWLEDGE_BASE_DIR, GAP_THRESHOLD, RERANKER, COMPANY_NAME, EXAMPLES_FILE
from utils.schemas import AskResponse, Citation, RelatedResource, QueryAnalysis
from utils.utils import (get_logger, get_llm, llm_text, parse_json_block, tokenize, answer_units, query_term_weights,
                         resolve_provider, default_llm_model)

logger = get_logger(__name__)

_DEFAULT_EXAMPLES = {
    "employee": ["How many sick days do I get?", "How do I set up the VPN?",
                 "What are the per diem rates for travel?"],
    "customer": ["What is the response time for a critical issue?", "How do I escalate a support ticket?",
                 "How is my data protected?"],
}


def load_examples(path: str = EXAMPLES_FILE) -> Dict[str, List[str]]:
    """Example questions per audience (data/examples.json), so they match YOUR knowledge base."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {k: [str(q) for q in v][:3] for k, v in data.items() if isinstance(v, list)} or _DEFAULT_EXAMPLES
    except Exception:
        return _DEFAULT_EXAMPLES


EXAMPLES = load_examples()


class AsklyAssistant:
    """Context-aware, grounded helpdesk assistant over an enterprise knowledge base."""

    def __init__(self, kb_dir: str = KNOWLEDGE_BASE_DIR, vector_store: Optional[VectorStore] = None,
                 use_llm: bool = True):
        self.kb_dir = kb_dir
        self.vector_store = vector_store or VectorStore()
        if not self.vector_store.is_ready():
            self.vector_store.initialize()
        self.indexer = HybridIndexer(self.vector_store)
        self.reranker = ReRanker(self.indexer, strategy=RERANKER if use_llm else "lexical")
        self.router = SearchRouter(self.indexer, self.reranker)
        if not use_llm:
            self.router.multi_query.llm = None
        self.analyzer = QueryAnalyzer(use_llm=use_llm)
        self.memory = MemoryRAG()
        self.gaps = KnowledgeGapDetector()
        self.escalations = EscalationManager()
        self.feedback = FeedbackStore()
        self.llm = get_llm() if use_llm else None
        self.source_docs = []
        self.catalog: Dict[str, Dict[str, Any]] = {}
        self.company = COMPANY_NAME or "the company"

    def _detect_company(self) -> str:
        """COMPANY_NAME from .env, else the organisation most documents name."""
        if COMPANY_NAME:
            return COMPANY_NAME
        orgs = Counter(d.get("organization") for d in self.catalog.values() if d.get("organization"))
        if not orgs:
            return "the company"
        return re.sub(r",?\s+(Inc|Ltd|LLC|GmbH|Corp|Corporation|plc)\.?$", "", orgs.most_common(1)[0][0]).strip()

    # ------------------------------------------------------------------ knowledge base

    def load_knowledge_base(self) -> Dict[str, Any]:
        docs, chunks = load_knowledge_base(self.kb_dir)
        self.source_docs = docs
        self.catalog = build_catalog(docs)
        self.company = self._detect_company()
        result = self.indexer.index(chunks)
        result["documents"] = len(docs)
        return result

    def add_document(self, path: str, category: Optional[str] = None) -> Dict[str, Any]:
        doc, chunks = load_file(path, category)
        if not doc:
            return {"ok": False, "error": "Could not read any text from this file."}
        self.source_docs = [d for d in self.source_docs if d.metadata["doc_id"] != doc.metadata["doc_id"]] + [doc]
        self.catalog.update(build_catalog([doc]))
        res = self.indexer.add(chunks)
        return {"ok": True, "doc_id": doc.metadata["doc_id"], "title": doc.metadata["title"],
                "chunks": len(chunks), "vector_sync": res.get("vector_sync")}

    # ------------------------------------------------------------------ main entry point

    def ask(self, question: str, session_id: str, mode: str = "auto",
            filters: Optional[Dict[str, Any]] = None, debug: bool = False) -> AskResponse:
        question = (question or "").strip()
        trace: Dict[str, Any] = {}
        profile = self.memory.get_profile(session_id)
        history = self.memory.get_history(session_id)
        analysis = self.analyzer.analyze(question, history)
        trace["analysis"] = analysis.model_dump()
        self.memory.add_user_message(session_id, question, analysis.standalone_query)
        turn_id = uuid.uuid4().hex[:10]

        if analysis.intent == "greeting":
            resp = self._greeting(profile)
        elif analysis.intent == "thanks":
            resp = AskResponse(answer="You're welcome! Is there anything else I can help you with?",
                               follow_ups=EXAMPLES.get(profile.get("user_type", "employee"), [])[:2],
                               intent="thanks", grounded=False)
        elif analysis.intent == "escalation_request":
            resp = self._escalation_response(session_id, question, profile, "User asked to speak to a human")
        else:
            resp = self._answer(question, analysis, session_id, profile, mode, filters, trace)

        resp.turn_id = turn_id
        resp.intent = resp.intent or analysis.intent
        resp.standalone_query = analysis.standalone_query
        resp.is_follow_up = analysis.is_follow_up
        resp.trace = trace if debug else {}
        self.memory.add_assistant_message(session_id, resp)
        return resp

    # ------------------------------------------------------------------ answering

    def _answer(self, question: str, analysis: QueryAnalysis, session_id: str, profile: Dict[str, Any],
                mode: str, filters: Optional[Dict[str, Any]], trace: Dict[str, Any]) -> AskResponse:
        search = self.router.search(analysis.standalone_query, mode=mode, filters=filters,
                                    user_type=profile.get("user_type"), category_hint=analysis.category_hint,
                                    feedback_boosts=self.feedback.chunk_boosts(), trace=trace)
        results = search["results"]
        assessment = self.gaps.assess(results)
        trace["gap_check"] = assessment

        if assessment["is_gap"]:
            return self._gap_response(question, analysis, session_id, profile, results,
                                      assessment["reason"], search["search_type"],
                                      out_of_scope=analysis.intent == "out_of_scope")

        # Drop the weak tail: passages far below the best one only dilute the answer.
        top = results[0]["relevance"]
        context = [r for r in results if r["relevance"] >= max(GAP_THRESHOLD * 0.8, top * 0.55)] or results[:1]

        if self.llm:
            gen = self._generate_llm(question, analysis, context, session_id, profile, trace)
        else:
            gen = None
        if gen is None:
            gen = self._generate_extractive(analysis.standalone_query, context)
        trace["generation"] = {k: v for k, v in gen.items() if k != "answer"}

        if not gen["answerable"]:
            return self._gap_response(question, analysis, session_id, profile, results,
                                      "retrieved passages do not contain the answer (model check)",
                                      search["search_type"], out_of_scope=analysis.intent == "out_of_scope")

        answer, cited = self._renumber(gen["answer"], gen["cited"], len(context))
        citations = [self._citation(i + 1, context[n - 1]) for i, n in enumerate(cited)]
        related = self._related_resources([context[n - 1] for n in cited], results)
        follow_ups = [f for f in gen.get("follow_ups", []) if f][:2] or self._follow_ups_from(related)
        return AskResponse(answer=answer, citations=citations, follow_ups=follow_ups, related=related,
                           confidence=assessment["confidence"], grounded=True, intent="kb_question",
                           search_type=search["search_type"], generation_mode=gen["mode"],
                           contexts=[c["text"] for c in context])

    def _generate_llm(self, question: str, analysis: QueryAnalysis, context: List[Dict[str, Any]],
                      session_id: str, profile: Dict[str, Any], trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        passages = "\n\n".join(f"[{i}] ({self._label(c)})\n{c['text']}" for i, c in enumerate(context, start=1))
        related = self._related_resources(context, context)
        related_txt = "; ".join(r.title for r in related) or "none"
        user_type = profile.get("user_type", "employee")
        name = profile.get("name") or "there"
        prompt = f"""You are Askly, the smart helpdesk assistant of {self.company}. You answer {user_type}s' questions
using ONLY the approved company documentation passages below.

Rules:
1. Use only facts stated in the passages. Cite every fact with its passage number, e.g. [1] or [2].
2. If the passages do not contain the answer, set "answerable" to false and do not guess.
3. For "how do I" questions give short numbered steps with the exact menu paths, links and contacts.
4. Mention important conditions (limits, deadlines, documents required).
5. Be concise (under 170 words), friendly and direct. The user's name is {name}; you may use it once.
6. If passages from different documents disagree (e.g. different numbers for the same policy), say so
   plainly, cite both, and suggest confirming with the document owner - never silently pick one.
7. "follow_ups": up to 2 short offers of the next thing the user will probably need, phrased like
   "Would you also like guidance on updating your 2FA settings?". Prefer these related sections: {related_txt}.

Conversation so far:
{self.memory.history_for_prompt(session_id)}

Current question: "{question}"
(Interpreted as: "{analysis.standalone_query}")

Passages:
{passages}

Return ONLY JSON:
{{"answerable": true, "answer": "text with [n] citations", "cited": [1], "follow_ups": ["..."]}}"""
        try:
            raw = llm_text(self.llm.invoke(prompt))
            data = parse_json_block(raw)
            trace["generation_raw"] = raw[:1500]
            if not isinstance(data, dict) or "answer" not in data:
                # Model ignored the JSON format - keep its text, recover [n] markers.
                cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", raw)})
                return {"mode": "llm", "answer": raw.strip(), "answerable": bool(raw.strip()),
                        "cited": cited, "follow_ups": [], "format_ok": False}
            cited = [int(n) for n in data.get("cited", []) if str(n).isdigit()]
            cited = cited or sorted({int(n) for n in re.findall(r"\[(\d+)\]", str(data.get("answer", "")))})
            return {"mode": "llm", "answer": str(data.get("answer", "")).strip(),
                    "answerable": bool(data.get("answerable", True)) and bool(str(data.get("answer", "")).strip()),
                    "cited": cited, "follow_ups": [str(f) for f in data.get("follow_ups", [])][:2],
                    "format_ok": True, "model": default_llm_model(resolve_provider())}
        except Exception as e:
            logger.error(f"LLM generation failed, using extractive answer: {e}")
            trace["generation_error"] = str(e)
            return None

    def _generate_extractive(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Offline answer: the best-covering sentences / bullet points of the top passage(s),
        each cited. Units scoring under 60% of the best unit are dropped as off-topic."""
        weights = query_term_weights(query, self.indexer.idf)
        picked: List[Tuple[int, List[str]]] = []
        used_sections = set()
        for n, c in enumerate(context[:2], start=1):
            if n == 2 and (c["relevance"] < context[0]["relevance"] * 0.75 or c["chunk_id"] in used_sections):
                break
            used_sections.add(c["chunk_id"])
            body = c["text"].split("\n", 1)[1] if "\n" in c["text"] else c["text"]   # drop context header
            units = answer_units(body)
            scored = [(sum(w for t, w in weights.items() if t in set(tokenize(u))), i, u) for i, u in enumerate(units)]
            best_score = max((x[0] for x in scored), default=0)
            limit = 5 if n == 1 else 3
            best = sorted([x for x in scored if x[0] > 0 and x[0] >= 0.6 * best_score], reverse=True)[:limit]
            best = best or scored[:2]
            if best:
                picked.append((n, [u for _, _, u in sorted(best, key=lambda x: x[1])]))
        parts = []
        for n, units in picked:
            label = self._label(context[n - 1])
            same = n > 1 and context[n - 1]["metadata"].get("section") == context[0]["metadata"].get("section")
            intro = f"From **{label}**:" if n == 1 else (f"Also from **{label}**:" if same else f"Also relevant - **{label}**:")
            parts.append(intro + "\n" + "\n".join(f"- {u} [{n}]" for u in units))
        return {"mode": "extractive", "answer": "\n\n".join(parts), "answerable": bool(parts),
                "cited": [n for n, _ in picked], "follow_ups": []}

    # ------------------------------------------------------------------ gaps & escalation

    def _gap_response(self, question: str, analysis: QueryAnalysis, session_id: str, profile: Dict[str, Any],
                      results: List[Dict[str, Any]], reason: str, search_type: str,
                      out_of_scope: bool = False) -> AskResponse:
        if out_of_scope:
            return AskResponse(
                answer=f"I'm Askly, the {self.company} helpdesk assistant, so I can only help with questions "
                       f"covered by our documentation ({', '.join(sorted({d['category'] for d in self.catalog.values()}))}). "
                       "What can I help you with?",
                follow_ups=EXAMPLES.get(profile.get("user_type", "employee"), [])[:2],
                grounded=False, intent="out_of_scope", search_type=search_type)

        top = results[0]["relevance"] if results else 0.0
        category = analysis.category_hint or detect_category(analysis.standalone_query)
        self.gaps.log_gap(analysis.standalone_query, reason, top, category, session_id)
        team = self.escalations.route(category)
        closest = [self._related_from_chunk(r) for r in results[:2] if r["relevance"] >= GAP_THRESHOLD * 0.8]

        # consecutive_gaps counts earlier turns; this turn is a gap too.
        if self.escalations.should_auto_escalate(question, self.memory.consecutive_gaps(session_id) + 1):
            ticket = self.escalations.create_ticket(
                session_id, question, analysis.standalone_query, f"Knowledge gap: {reason}", category,
                self.memory.transcript(session_id), [self._label(r) for r in results[:3]],
                profile.get("name"), profile.get("user_type"))
            answer = (f"I couldn't find this in the approved documentation, and I don't want to guess. "
                      f"I've escalated it to **{team}** as ticket **{ticket.ticket_id}** "
                      f"({ticket.priority} priority) - they'll follow up with you directly.")
            return AskResponse(answer=answer, related=closest, confidence=top, grounded=False, knowledge_gap=True,
                               escalated=True, ticket_id=ticket.ticket_id, intent="kb_question",
                               search_type=search_type, generation_mode="none")

        answer = ("I couldn't find an answer to that in the approved documentation, so I'd rather not guess. "
                  f"I've logged it as a documentation gap. Would you like me to pass your question to **{team}**?")
        if closest:
            answer += " In the meantime, the closest related topics I found are listed below."
        return AskResponse(answer=answer, related=closest, confidence=top, grounded=False, knowledge_gap=True,
                           follow_ups=["Yes, connect me with a human agent"], intent="kb_question",
                           search_type=search_type, generation_mode="none")

    def _last_question_context(self, session_id: str) -> Tuple[str, str, Optional[str], List[str]]:
        """Most recent knowledge-base question in this session (what an escalation is about)."""
        turns = list(self.memory._session(session_id)["turns"].values())
        for t in reversed(turns):
            if t.intent == "kb_question":
                sources = [c.label() for c in t.citations] or [
                    f"{r.doc_title} · {r.section}" if r.section else r.doc_title for r in t.related]
                return t.standalone_query, t.standalone_query, detect_category(t.standalone_query), sources
        users = [m for m in self.memory.get_history(session_id, 20) if m["role"] == "user"]
        last = users[-1]["content"] if users else ""
        return last, last, detect_category(last), []

    def escalate(self, session_id: str, reason: str = "User requested a human agent",
                 question: Optional[str] = None):
        profile = self.memory.get_profile(session_id)
        q, standalone, category, sources = self._last_question_context(session_id)
        if question:
            q, standalone, category = question, question, detect_category(question) or category
        return self.escalations.create_ticket(session_id, q, standalone, reason, category,
                                              self.memory.transcript(session_id), sources,
                                              profile.get("name"), profile.get("user_type"))

    def _escalation_response(self, session_id: str, question: str, profile: Dict[str, Any], reason: str) -> AskResponse:
        ticket = self.escalate(session_id, reason)
        name = profile.get("name")
        answer = (f"{'Sure, ' + name if name else 'Sure'} - I've created ticket **{ticket.ticket_id}** for "
                  f"**{ticket.team}** with our conversation attached, so you won't have to repeat yourself. "
                  f"Priority: {ticket.priority}. They'll get back to you shortly.")
        return AskResponse(answer=answer, escalated=True, ticket_id=ticket.ticket_id, grounded=False,
                           intent="escalation_request")

    def record_feedback(self, session_id: str, turn_id: str, rating: int, comment: Optional[str] = None):
        turn = self.memory.get_turn(session_id, turn_id)
        if not turn:
            return None
        rec = self.feedback.record(session_id, turn_id, turn.standalone_query, turn.answer, rating,
                                   [c.chunk_id for c in turn.citations], comment)
        if rating < 0:
            self.gaps.log_gap(turn.standalone_query, f"user rated answer unhelpful{': ' + comment if comment else ''}",
                              turn.confidence, detect_category(turn.standalone_query), session_id)
        return rec

    # ------------------------------------------------------------------ proactive layer

    @staticmethod
    def _label(c: Dict[str, Any]) -> str:
        md = c["metadata"]
        sec = md.get("section", "")
        parts = [md.get("title", "")]
        if sec and sec != "Overview":
            parts.append(f"Section {sec}" if sec[:1].isdigit() else sec)
        if md.get("page"):
            parts.append(f"p. {md['page']}")
        return ", ".join(p for p in parts if p)

    def _citation(self, number: int, c: Dict[str, Any]) -> Citation:
        md = c["metadata"]
        body = c["text"].split("\n", 1)[1] if "\n" in c["text"] else c["text"]
        return Citation(number=number, chunk_id=c["chunk_id"], doc_title=md.get("title", ""),
                        section=md.get("section") if md.get("section") != "Overview" else None,
                        page=md.get("page") or None, category=md.get("category"), source=md.get("source", ""),
                        url=md.get("url") or None, snippet=body[:600], relevance=c.get("relevance", 0.0))

    def _related_from_chunk(self, c: Dict[str, Any]) -> RelatedResource:
        md = c["metadata"]
        return RelatedResource(title=md.get("section_title") or md.get("title"), doc_title=md.get("title", ""),
                               section=md.get("section"), url=md.get("url") or None)

    def _related_resources(self, cited: List[Dict[str, Any]], results: List[Dict[str, Any]],
                           limit: int = 3) -> List[RelatedResource]:
        """Neighbouring sections of the cited documents (same parent first, then the next
        top-level section), plus other documents that retrieval surfaced."""
        out: List[RelatedResource] = []
        used = {c["metadata"].get("section") for c in cited}
        seen = set(used)
        for c in cited:
            md = c["metadata"]
            doc = self.catalog.get(md.get("doc_id"))
            if not doc:
                continue
            sections = doc["sections"]
            idx = next((i for i, s in enumerate(sections) if s["heading"] == md.get("section")), None)
            if idx is None:
                continue
            parent = sections[idx].get("parent")
            siblings = [s for s in sections if parent and s.get("parent") == parent and s["heading"] not in seen]
            after = [s for s in sections[idx + 1:] if s["heading"] not in seen and s.get("parent") != parent]
            for s in (siblings[:2] + after[:1]):
                seen.add(s["heading"])
                out.append(RelatedResource(title=s["title"], doc_title=doc["title"], section=s["heading"],
                                           url=doc.get("url")))
        for r in results:
            md = r["metadata"]
            if md.get("section") not in seen and md.get("doc_id") not in {c["metadata"].get("doc_id") for c in cited}:
                seen.add(md.get("section"))
                out.append(self._related_from_chunk(r))
        return out[:limit]

    @staticmethod
    def _follow_ups_from(related: List[RelatedResource]) -> List[str]:
        def phrase(t: str) -> str:
            # sentence case, but keep acronyms / product names like "2FA", "VPN", "Wi-Fi"
            return " ".join(w if (w.isupper() or any(ch.isdigit() for ch in w) or "-" in w) else w.lower()
                            for w in t.split())
        return [f"Would you also like guidance on {phrase(r.title)}?" for r in related[:2]]

    @staticmethod
    def _renumber(answer: str, cited: List[int], n_context: int) -> Tuple[str, List[int]]:
        """Keep only valid citation numbers, renumber them 1..k in order of first use."""
        order: List[int] = []
        for n in [int(x) for x in re.findall(r"\[(\d+)\]", answer)] + list(cited):
            if 1 <= n <= n_context and n not in order:
                order.append(n)
        if not order:
            order = [1]
            answer = answer.rstrip() + " [1]"
        mapping = {old: new for new, old in enumerate(order, start=1)}
        answer = re.sub(r"\[(\d+)\]", lambda m: f"[{mapping[int(m.group(1))]}]" if int(m.group(1)) in mapping else "",
                        answer)
        return answer, order

    def _greeting(self, profile: Dict[str, Any]) -> AskResponse:
        name = profile.get("name")
        user_type = profile.get("user_type", "employee")
        cats = sorted({d["category"] for d in self.catalog.values()})
        answer = (f"Hi{' ' + name if name else ''}! I'm **Askly**, the {self.company} helpdesk assistant. I answer from "
                  f"{len(self.catalog)} approved documents ({', '.join(cats)}), always with sources, and I'll "
                  "hand you over to a human when the documentation doesn't cover something. What do you need?")
        return AskResponse(answer=answer, follow_ups=EXAMPLES.get(user_type, EXAMPLES["employee"]),
                           intent="greeting", grounded=False)

    # ------------------------------------------------------------------ status

    def status(self) -> Dict[str, Any]:
        provider = resolve_provider()
        return {
            "company": self.company,
            "llm": f"{provider}:{default_llm_model(provider)}" if self.llm else "offline (extractive answers)",
            "reranker": self.reranker.method,
            "query_analysis": "llm" if self.analyzer.llm else "heuristic",
            "index": self.indexer.get_index_stats(),
            "memory": self.memory.get_memory_stats(),
            "feedback": {k: v for k, v in self.feedback.stats().items() if k != "comments"},
            "open_tickets": sum(1 for t in self.escalations.list_tickets() if t.get("status") != "resolved"),
            "knowledge_gaps": len(self.gaps.list_gaps()),
        }
