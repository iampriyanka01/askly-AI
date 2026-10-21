"""
main.py  -  Askly command-line demo and orchestrator
====================================================

Wires every module together and demonstrates them (same role as HireFlow's main.py):

    1. check_prerequisites()   knowledge base present? which API keys are set?
    2. initialize_system()     AsklyAssistant (vector store, hybrid index, reranker, router,
                               query analyser, memory, gap detector, feedback)
    3. load_knowledge_base()   documents -> chunks -> BM25 + dense index (embed-once)
    4. run_demo_conversation() the scenarios from the problem statement:
                               multi-turn HR question, customer password reset with a
                               proactive 2FA suggestion, and a knowledge gap -> escalation
    5. interactive_menu()      chat, RAGAS evaluation, tickets & gaps, status, re-index

Usage
    python main.py              full demo + menu
    python main.py --chat       straight into chat
    python main.py --eval       run the evaluation and exit
    python main.py --offline    ignore API keys (BM25 + extractive answers)
"""

import argparse
import sys
import textwrap
from pathlib import Path
from core.rag_pipeline import AsklyAssistant
from core.evaluator import RAGEvaluator
from utils.config import KNOWLEDGE_BASE_DIR, VECTOR_BACKEND
from utils.utils import get_logger, resolve_provider

logger = get_logger(__name__)
WIDTH = 92


def hr(char: str = "-") -> None:
    print(char * WIDTH)


def print_response(resp) -> None:
    print()
    for para in resp.answer.split("\n"):
        print(textwrap.fill(para, WIDTH, subsequent_indent="  ") if para.strip() else "")
    if resp.citations:
        print("\n  Sources:")
        for c in resp.citations:
            print(f"   [{c.number}] {c.label()}" + (f"  <{c.url}>" if c.url else ""))
    if resp.related:
        print("  Related: " + "; ".join(f"{r.title} ({r.doc_title})" for r in resp.related))
    if resp.follow_ups:
        print("  Suggested next: " + " | ".join(resp.follow_ups))
    meta = [f"confidence {resp.confidence:.2f}", resp.search_type or resp.intent]
    if resp.is_follow_up:
        meta.append(f"follow-up -> \"{resp.standalone_query}\"")
    if resp.ticket_id:
        meta.append(f"ticket {resp.ticket_id}")
    print("  (" + " · ".join(m for m in meta if m) + ")")


class AsklyDemo:
    def __init__(self, offline: bool = False):
        self.offline = offline
        self.assistant: AsklyAssistant = None
        self.evaluator: RAGEvaluator = None
        self.session_id = None

    def check_prerequisites(self) -> bool:
        kb = Path(KNOWLEDGE_BASE_DIR)
        files = [p for p in kb.rglob("*") if p.suffix.lower() in {".pdf", ".md", ".markdown", ".txt"}] if kb.exists() else []
        print(f"Knowledge base : {kb}  ({len(files)} files)")
        if not files:
            print("  No documents found - add PDFs / Markdown files under data/knowledge_base/<category>/")
            return False
        provider = None if self.offline else resolve_provider()
        print(f"LLM provider   : {provider or 'none -> offline mode (BM25 + extractive answers)'}")
        if not provider and not self.offline:
            from utils.key_guide import KEY_GUIDE
            print("  Get a key (one is enough), put it in .env, and re-run:")
            for k in KEY_GUIDE:
                print(f"   - {k['env']:<17} {k['url']}   ({k['needed'].split(' - ')[0].lower()})")
        print(f"Vector backend : {VECTOR_BACKEND}{'' if provider else ' (disabled without an embedding key)'}")
        return True

    def initialize_system(self) -> None:
        self.assistant = AsklyAssistant(use_llm=not self.offline)
        self.evaluator = RAGEvaluator(self.assistant)

    def load_knowledge_base(self) -> None:
        res = self.assistant.load_knowledge_base()
        print(f"Indexed {res['documents']} documents as {res['chunks']} chunks for {self.assistant.company}. "
              f"Dense sync: {res['vector_sync']}")

    def run_demo_conversation(self) -> None:
        scenarios = [
            ("Employee HR query (multi-turn)", {"name": "Priya", "user_type": "employee"},
             ["How do I apply for medical leave for a planned surgery?", "And what about sick leave?"]),
            ("IT support + proactive help", {"name": "Priya", "user_type": "employee"},
             ["My account is locked, what should I do?"]),
            ("Customer support", {"name": "Sam", "user_type": "customer"},
             ["What is the response time for a critical issue on the Platinum tier?"]),
            ("Conflicting documentation", {"name": "Priya", "user_type": "employee"},
             ["How many days a week do I need to be in the office?"]),
            ("Knowledge gap -> escalation", {"name": "Priya", "user_type": "employee"},
             ["Can I bring my dog to the office?", "Yes, connect me with a human agent"]),
        ]
        for title, profile, turns in scenarios:
            hr("=")
            print(f"SCENARIO: {title}   (user: {profile['name']}, {profile['user_type']})")
            sid = self.assistant.memory.new_session(profile)
            for q in turns:
                hr()
                print(f"{profile['name']}: {q}")
                print_response(self.assistant.ask(q, sid))

    def chat(self) -> None:
        if not self.session_id:
            name = input("Your name (optional): ").strip() or None
            kind = input("Are you an employee or a customer? [employee]: ").strip().lower() or "employee"
            self.session_id = self.assistant.memory.new_session(
                {"name": name, "user_type": "customer" if kind.startswith("c") else "employee"})
        print("\nChat with Askly. Commands: /new  /escalate  /good  /bad [comment]  /quit\n")
        last = None
        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not q:
                continue
            if q in ("/quit", "/exit", "/q"):
                return
            if q == "/new":
                self.assistant.memory.clear(self.session_id)
                print("(conversation cleared)")
                continue
            if q == "/escalate":
                t = self.assistant.escalate(self.session_id)
                print(f"Ticket {t.ticket_id} created for {t.team}.")
                continue
            if q.startswith(("/good", "/bad")) and last:
                rating = 1 if q.startswith("/good") else -1
                comment = q.split(" ", 1)[1] if " " in q else None
                self.assistant.record_feedback(self.session_id, last.turn_id, rating, comment)
                print("Thanks - feedback recorded.")
                continue
            last = self.assistant.ask(q, self.session_id)
            print_response(last)
            print()

    def run_evaluation(self) -> None:
        items = self.evaluator.load_testset()
        use_ragas = not self.offline and resolve_provider() is not None
        est = self.evaluator.estimate_cost(len(items), use_ragas)
        print(f"{len(items)} test questions. " + (
            f"RAGAS with an LLM judge: ~{est['estimated_llm_calls']} LLM calls, ~{est['estimated_minutes']} min."
            if use_ragas else "No LLM key -> offline proxy metrics (not RAGAS)."))
        if use_ragas and input("Run it? [y/N]: ").strip().lower() != "y":
            return
        report = self.evaluator.run(use_ragas=use_ragas)
        hr("=")
        print(f"Method: {report.method}   Overall: {report.overall_score:.3f}"
              + ("   (judge failed on some metrics - treat 0.0 as unknown)" if report.judge_failed else ""))
        for k, v in report.metrics.items():
            print(f"  {k:<20} {v:.3f}")
        for k, v in report.retrieval.items():
            print(f"  {k:<20} {v:.3f}")
        hr()
        for r in report.rows:
            flag = "gap ok" if r.get("gap_correct") else ("hit" if r.get("source_hit") else "miss")
            print(f"  {r['id']:<8} {flag:<7} {r['question'][:70]}")

    def show_tickets_and_gaps(self) -> None:
        tickets = self.assistant.escalations.list_tickets()
        print(f"\nEscalation tickets ({len(tickets)}):")
        for t in tickets[-10:]:
            print(f"  {t['ticket_id']}  [{t['status']}/{t['priority']}]  {t['team']}: {t['question'][:60]}")
        print("\nTop knowledge gaps:")
        for g in self.assistant.gaps.gap_summary():
            print(f"  {g['count']}x  {g['query'][:80]}")

    def check_system_status(self) -> None:
        import json
        print(json.dumps(self.assistant.status(), indent=2, default=str))

    def interactive_menu(self) -> None:
        options = {"1": ("Chat with Askly", self.chat), "2": ("Run the demo conversations", self.run_demo_conversation),
                   "3": ("Evaluate quality (RAGAS / offline proxies)", self.run_evaluation),
                   "4": ("Escalation tickets & knowledge gaps", self.show_tickets_and_gaps),
                   "5": ("System status", self.check_system_status),
                   "6": ("Re-index knowledge base", self.load_knowledge_base)}
        while True:
            hr("=")
            for k, (label, _) in options.items():
                print(f"  {k}. {label}")
            print("  0. Exit")
            choice = input("Choose: ").strip()
            if choice == "0":
                return
            if choice in options:
                options[choice][1]()


def main() -> int:
    parser = argparse.ArgumentParser(description="Askly - Smart HelpDesk Assistant")
    parser.add_argument("--chat", action="store_true", help="go straight to chat")
    parser.add_argument("--eval", action="store_true", help="run the evaluation and exit")
    parser.add_argument("--offline", action="store_true", help="ignore API keys")
    parser.add_argument("--no-demo", action="store_true", help="skip the demo conversations")
    args = parser.parse_args()

    hr("=")
    print("ASKLY - Smart HelpDesk Assistant")
    hr("=")
    demo = AsklyDemo(offline=args.offline)
    if not demo.check_prerequisites():
        return 1
    demo.initialize_system()
    demo.load_knowledge_base()
    if args.eval:
        demo.run_evaluation()
        return 0
    if args.chat:
        demo.chat()
        return 0
    if not args.no_demo:
        demo.run_demo_conversation()
    demo.interactive_menu()
    return 0


if __name__ == "__main__":
    sys.exit(main())
