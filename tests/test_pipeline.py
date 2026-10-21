"""End-to-end tests for Askly on the Orion Digital Services knowledge base. Run: pytest -q

No API keys needed: dense retrieval uses LangChain's deterministic fake embeddings
and LLM generation uses a fake chat model, so every code path runs offline.
"""

import json
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from core.ingestion import load_knowledge_base, infer_category
from core.vector_store import VectorStore
from core.hybrid_indexer import HybridIndexer
from core.rag_pipeline import AsklyAssistant
from core.evaluator import RAGEvaluator
from utils.config import KNOWLEDGE_BASE_DIR, PROJECT_ROOT

SAMPLE_KB = PROJECT_ROOT / "data" / "sample_knowledge_base"


@pytest.fixture(scope="module")
def assistant():
    a = AsklyAssistant(use_llm=False)
    a.load_knowledge_base()
    return a


def test_ingestion_flat_markdown_kb():
    docs, chunks = load_knowledge_base(KNOWLEDGE_BASE_DIR)
    assert len(docs) == 7
    cats = {d.metadata["source"]: d.metadata["category"] for d in docs}
    assert cats["it_support_guide.md"] == "it" and cats["customer_service_guide.md"] == "customer"
    assert all(d.metadata["organization"] == "Orion Digital Services, Inc." for d in docs)
    assert all(d.metadata["owner"] for d in docs)
    headings = {c.metadata["section"] for c in chunks}
    # '# 1. Pull latest code' lives inside a ```bash fence and must not become a section
    assert not any(h.startswith(("1. ", "2. ")) for h in headings)
    assert "Orion Digital Services, Inc." not in headings
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks), "chunk ids must be unique"


def test_ingestion_pdf_pages_and_numbered_sections():
    docs, chunks = load_knowledge_base(str(SAMPLE_KB))
    sick = [c for c in chunks if c.metadata["section"] == "4.2 Sick Leave"]
    assert sick and sick[0].metadata["page"] > 0 and sick[0].metadata["source"] == "HR_Handbook.pdf"


def test_infer_category_from_filename():
    assert infer_category("technical_documentation.md") == "engineering"
    assert infer_category("company_operations.md") == "operations"
    assert infer_category("sample_policy.md") == "hr"


def test_company_name_detected(assistant):
    assert assistant.company == "Orion Digital Services"


def test_multi_turn_follow_up(assistant):
    sid = assistant.memory.new_session({"user_type": "employee"})
    first = assistant.ask("How many sick days do I get?", sid)
    assert "10" in first.answer and first.citations
    second = assistant.ask("And what about parental leave?", sid)
    assert second.is_follow_up
    assert "12 weeks" in second.answer


def test_escalation_question_is_not_a_human_request(assistant):
    sid = assistant.memory.new_session({"user_type": "customer"})
    r = assistant.ask("How do I escalate a support ticket?", sid)
    assert r.intent == "kb_question" and not r.escalated
    assert any(c.source == "customer_service_guide.md" for c in r.citations)


def test_knowledge_gap_then_escalation(assistant):
    sid = assistant.memory.new_session({"user_type": "employee", "name": "Test"})
    gap = assistant.ask("Can I bring my dog to the office?", sid)
    assert gap.knowledge_gap and not gap.citations and not gap.escalated
    esc = assistant.ask("Yes, connect me with a human agent", sid)
    assert esc.escalated and esc.ticket_id.startswith("ASK-")
    ticket = [t for t in assistant.escalations.list_tickets() if t["ticket_id"] == esc.ticket_id][0]
    assert "dog" in ticket["question"].lower() and ticket["conversation"]


def test_urgent_gap_auto_escalates_with_high_priority(assistant):
    sid = assistant.memory.new_session({"user_type": "employee"})
    r = assistant.ask("URGENT: the coffee machine on floor 3 is leaking", sid)
    assert r.knowledge_gap and r.escalated
    t = [t for t in assistant.escalations.list_tickets() if t["ticket_id"] == r.ticket_id][0]
    assert t["priority"] == "high"


def test_feedback_boosts_cited_chunks(assistant):
    sid = assistant.memory.new_session({"user_type": "employee"})
    r = assistant.ask("What are the per diem rates for international travel?", sid)
    assistant.record_feedback(sid, r.turn_id, +1)
    assert assistant.feedback.chunk_boosts()[r.citations[0].chunk_id] > 0


def test_faiss_dense_sync_embeds_once(tmp_path):
    docs, chunks = load_knowledge_base(KNOWLEDGE_BASE_DIR)
    vs = VectorStore(backend="faiss", embeddings=DeterministicFakeEmbedding(size=64), index_dir=str(tmp_path))
    assert vs.initialize()
    assert vs.sync(chunks)["added"] == len(chunks)
    again = VectorStore(backend="faiss", embeddings=DeterministicFakeEmbedding(size=64), index_dir=str(tmp_path))
    again.initialize()
    second = again.sync(chunks)
    assert second["added"] == 0 and second["unchanged"] == len(chunks)
    hits = again.search("sick leave", top_k=5, filters={"category": "hr"})
    assert hits and all(h["metadata"]["category"] == "hr" for h in hits)
    hi = HybridIndexer(again)
    hi.index(chunks)
    assert any("dense" in r["matched_by"] for r in hi.search("sick leave", top_k=5))


def test_llm_generation_path_uses_company_and_renumbers(assistant):
    reply = {"answerable": True, "cited": [2], "answer": "You get 10 paid sick days a year [2].",
             "follow_ups": ["Would you also like guidance on parental leave?"]}
    fake = FakeListChatModel(responses=[json.dumps(reply)])
    assistant.llm = fake
    try:
        sid = assistant.memory.new_session({"user_type": "employee"})
        r = assistant.ask("How many sick days do I get?", sid, debug=True)
        assert r.generation_mode == "llm"
        assert "[1]" in r.answer and "[2]" not in r.answer and len(r.citations) == 1
        assert r.follow_ups == ["Would you also like guidance on parental leave?"]
    finally:
        assistant.llm = None


def test_llm_says_not_answerable_becomes_gap(assistant):
    assistant.llm = FakeListChatModel(responses=[json.dumps({"answerable": False, "answer": "", "cited": []})])
    try:
        sid = assistant.memory.new_session({"user_type": "employee"})
        r = assistant.ask("What is the stock option vesting schedule?", sid)
        assert r.knowledge_gap and not r.citations
    finally:
        assistant.llm = None


def test_offline_evaluation_runs(assistant):
    report = RAGEvaluator(assistant).run(use_ragas=False)
    assert report.method == "offline_proxy"
    assert report.retrieval["source_hit_rate"] >= 0.85
    assert report.retrieval["false_gap_rate"] == 0.0
