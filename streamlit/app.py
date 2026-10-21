# streamlit/app.py
"""Askly - Smart HelpDesk Assistant (Streamlit UI).

    streamlit run streamlit/app.py

Pages
    Chat              multi-turn conversation with sources, confidence, follow-up
                      suggestions, 👍/👎 feedback, one-click escalation, optional RAGAS
                      scoring of an answer and a step-by-step retrieval trace
    Knowledge base    documents and sections, upload a new document, try retrieval
    Escalations       human-handoff tickets and the knowledge-gap log
    Quality           RAGAS evaluation on the test set, feedback analytics, past runs
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # so `import ui_helpers` works
import ui_helpers as uh   # noqa: E402

from core.rag_pipeline import AsklyAssistant, EXAMPLES   # noqa: E402
from core.evaluator import RAGEvaluator                   # noqa: E402
from utils.config import KNOWLEDGE_BASE_DIR               # noqa: E402
from utils.utils import tokenize                          # noqa: E402
from utils.key_guide import KEY_GUIDE, ENV_SNIPPET, key_status, mask   # noqa: E402
from utils import config as app_config                    # noqa: E402

st.set_page_config(page_title="Askly - HelpDesk Assistant", page_icon="💬", layout="wide")
st.markdown(uh.CSS, unsafe_allow_html=True)


# ============================================================================
# SHARED RESOURCES (built once per server process)
# ============================================================================

@st.cache_resource(show_spinner="Starting Askly - loading and indexing the knowledge base...")
def get_assistant() -> AsklyAssistant:
    assistant = AsklyAssistant()
    assistant.load_knowledge_base()
    return assistant


@st.cache_resource(show_spinner=False)
def get_evaluator() -> RAGEvaluator:
    return RAGEvaluator(get_assistant())


assistant = get_assistant()
status = assistant.status()


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("user_type", "employee")
    ss.setdefault("user_name", "")
    ss.setdefault("messages", [])          # [{"role", "content", "resp": dict|None}]
    ss.setdefault("pending", None)         # a question queued by a button click
    ss.setdefault("page", "Chat")
    if "session_id" not in ss:
        ss.session_id = assistant.memory.new_session({"user_type": ss.user_type})


init_state()
ss = st.session_state


def reset_conversation() -> None:
    assistant.memory.clear(ss.session_id)
    ss.messages = []
    ss.pending = None


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 💬 Askly")
    st.caption("Smart HelpDesk Assistant")
    PAGES = ["Chat", "Knowledge base", "Escalations & gaps", "Quality & feedback", "API keys"]
    ss.page = st.radio("Go to", PAGES, index=PAGES.index(ss.page), label_visibility="collapsed",
                       format_func=lambda p: "🔑 API keys" if p == "API keys" else p)
    st.divider()
    st.markdown("**You**")
    ss.user_type = st.radio("I am a", ["employee", "customer"], horizontal=True,
                            index=0 if ss.user_type == "employee" else 1,
                            help="Personalises retrieval (customers see product docs first) and tone.")
    ss.user_name = st.text_input("Name (optional)", value=ss.user_name)
    assistant.memory.set_profile(ss.session_id, user_type=ss.user_type, name=ss.user_name or None)

    st.markdown("**Retrieval**")
    mode = st.selectbox("Search mode", ["auto", "shallow", "deep"],
                        help="shallow = one hybrid search · deep = multi-query expansion · "
                             "auto = shallow, then deep only if the first result is weak")
    cats = sorted({d["category"] for d in assistant.catalog.values()})
    cat_filter = st.multiselect("Limit to categories", cats, help="Hard filter; leave empty to search everything")
    show_trace = st.checkbox("Show retrieval trace", help="BM25 vs dense hits, RRF fusion, rerank scores")
    if st.button("🧹 New conversation", width="stretch"):
        reset_conversation()
        st.rerun()

    st.divider()
    st.markdown("**System**")
    idx = status["index"]
    dense = idx["dense"]
    st.caption(f"LLM: `{status['llm']}`")
    st.caption(f"Dense ({dense['backend']}): {'✅' if dense['ready'] else '⚪'} {dense['status']}")
    st.caption(f"BM25: {'✅' if idx['bm25_ready'] else '❌'} · {idx['chunks']} chunks / {idx['documents']} docs")
    st.caption(f"Reranker: `{status['reranker']}` · Query analysis: `{status['query_analysis']}`")
    keys = key_status()
    st.caption("Keys: " + " · ".join(f"{'✅' if keys[k['env']] else '⚪'} {k['provider'].split()[0]}" for k in KEY_GUIDE))
    if not (keys["GOOGLE_API_KEY"] or keys["OPENAI_API_KEY"]):
        st.info("Running in **offline mode** (keyword search + quoted answers). Add a free Gemini key "
                "for written answers, semantic search and RAGAS.", icon="🔑")
        if st.button("🔑 How to get an API key", width="stretch", type="primary"):
            ss.page = "API keys"
            st.rerun()
    elif not dense["ready"]:
        st.warning(f"Dense search is off: {dense['status']}", icon="⚠️")


# ============================================================================
# CHAT PAGE
# ============================================================================

def render_assistant_message(msg: dict, i: int, is_last: bool) -> None:
    resp = msg["resp"] or {}
    st.markdown(uh.md_safe(msg["content"]))
    st.markdown(uh.badges_html(resp), unsafe_allow_html=True)

    if resp.get("citations"):
        terms = tokenize(resp.get("standalone_query", ""))
        with st.expander(f"📚 Sources ({len(resp['citations'])})", expanded=False):
            for c in resp["citations"]:
                st.markdown(uh.source_card_html(c, terms), unsafe_allow_html=True)
    if resp.get("related"):
        st.markdown(uh.related_html(resp["related"]), unsafe_allow_html=True)

    if is_last and resp.get("follow_ups"):
        cols = st.columns(min(len(resp["follow_ups"]), 3))
        for j, f in enumerate(resp["follow_ups"][:3]):
            if cols[j].button(f, key=f"fu-{i}-{j}", width="stretch"):
                ss.pending = f
                st.rerun()

    turn_id = resp.get("turn_id")
    if resp.get("intent") == "kb_question" and turn_id:
        c1, c2, c3 = st.columns([1.3, 1.3, 3])
        with c1:
            rating = st.feedback("thumbs", key=f"fb-{turn_id}")
            if rating is not None and not msg.get("rated"):
                assistant.record_feedback(ss.session_id, turn_id, 1 if rating == 1 else -1)
                msg["rated"] = True
                st.toast("Thanks - your feedback improves future answers.")
        with c2:
            if not resp.get("escalated") and st.button("🙋 Ask a human", key=f"esc-{turn_id}"):
                t = assistant.escalate(ss.session_id, reason="User clicked 'Ask a human' on an answer")
                ss.messages.append({"role": "assistant", "resp": None,
                                    "content": f"I've created ticket **{t.ticket_id}** for **{t.team}** with this "
                                               f"conversation attached. Priority: {t.priority}."})
                st.rerun()
        with c3:
            if resp.get("citations"):
                with st.popover("📏 Score this answer (RAGAS)"):
                    st.caption("Faithfulness + answer relevancy for this one answer. Uses ~4-6 LLM calls "
                               "(offline: token-overlap proxies).")
                    if st.button("Run", key=f"rg-{turn_id}"):
                        with st.spinner("Judging..."):
                            msg["ragas"] = get_evaluator().score_answer(
                                resp.get("standalone_query", ""), resp["answer"], resp.get("contexts", []))
                    if msg.get("ragas"):
                        st.json(msg["ragas"])

    if resp.get("trace"):
        with st.expander("🔎 Retrieval trace"):
            render_trace(resp["trace"])


def render_trace(trace: dict) -> None:
    a = trace.get("analysis", {})
    st.markdown(f"**1 · Query understanding** ({a.get('method')}): intent `{a.get('intent')}`, "
                f"follow-up `{a.get('is_follow_up')}`, category hint `{a.get('category_hint')}`")
    st.code(a.get("standalone_query", ""), language=None)
    r = trace.get("router", {})
    if r:
        st.markdown(f"**2 · Router**: `{r.get('search_type')}` (requested `{r.get('mode_requested')}`, "
                    f"escalated to deep: `{r.get('escalated_to_deep')}`), boosts `{r.get('category_boosts')}`")
    ret = trace.get("retrieval", {})
    for q in ret.get("queries", []):
        st.markdown(f"**3 · Retrieval for** _{q['query']}_ - BM25 tokens `{q['bm25_tokens']}`")
        c1, c2 = st.columns(2)
        c1.caption("BM25 (sparse)")
        c1.dataframe(pd.DataFrame(q["bm25_top"]), hide_index=True, width="stretch")
        c2.caption("Dense (embeddings)" if ret.get("dense_enabled") else f"Dense: off - {ret.get('dense_status')}")
        if q["dense_top"]:
            c2.dataframe(pd.DataFrame(q["dense_top"]), hide_index=True, width="stretch")
    if ret.get("fused_top"):
        st.markdown(f"**4 · Fusion** - {ret.get('fusion')}")
        st.dataframe(pd.DataFrame(ret["fused_top"]), hide_index=True, width="stretch")
    rr = trace.get("rerank", {})
    if rr.get("scores"):
        st.markdown(f"**5 · Rerank** (`{rr.get('method')}`) {rr.get('note') or ''} - top {rr.get('kept')} kept")
        st.dataframe(pd.DataFrame(rr["scores"]), hide_index=True, width="stretch")
    if trace.get("gap_check"):
        g = trace["gap_check"]
        st.markdown(f"**6 · Gap check**: confidence `{g['confidence']}` - {'⚠️ gap' if g['is_gap'] else '✅ answerable'} "
                    f"({g['reason']})")
    if trace.get("generation"):
        st.markdown(f"**7 · Generation**: `{trace['generation']}`")
    if trace.get("generation_raw"):
        with st.popover("Raw LLM reply"):
            st.code(trace["generation_raw"], language="json")


def ask(question: str) -> None:
    ss.messages.append({"role": "user", "content": question, "resp": None})
    filters = {"category": cat_filter} if cat_filter else None
    with st.spinner("Searching the knowledge base..."):
        resp = assistant.ask(question, ss.session_id, mode=mode, filters=filters, debug=show_trace)
    data = resp.model_dump()
    ss.messages.append({"role": "assistant", "content": resp.answer, "resp": data})


def chat_page() -> None:
    if not ss.messages:
        st.markdown(uh.hero_html(idx["documents"], idx["chunks"], status["llm"]), unsafe_allow_html=True)
        st.markdown("**Try asking**")
        examples = EXAMPLES.get(ss.user_type, EXAMPLES["employee"])
        cols = st.columns(len(examples))
        for j, ex in enumerate(examples):
            if cols[j].button(ex, key=f"ex-{j}", width="stretch"):
                ss.pending = ex
                st.rerun()

    for i, msg in enumerate(ss.messages):
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "💬"):
            if msg["role"] == "user" or not msg.get("resp"):
                st.markdown(uh.md_safe(msg["content"]))
            else:
                render_assistant_message(msg, i, is_last=(i == len(ss.messages) - 1))

    typed = st.chat_input("Ask about HR policies, IT issues, your account, expenses...")
    question = typed or ss.pending
    if question:
        ss.pending = None
        ask(question)
        st.rerun()


# ============================================================================
# KNOWLEDGE BASE PAGE
# ============================================================================

def kb_page() -> None:
    st.header("📚 Knowledge base")
    cols = st.columns(4)
    for col, (v, label) in zip(cols, [(idx["documents"], "documents"), (idx["chunks"], "indexed passages"),
                                     (dense.get("embedded_chunks", 0), "passages embedded"),
                                     (len({d['category'] for d in assistant.catalog.values()}), "categories")]):
        col.markdown(uh.kpi_html(v, label), unsafe_allow_html=True)
    st.write("")

    chunk_counts = pd.Series([c.metadata["doc_id"] for c in assistant.indexer.chunks]).value_counts()
    rows = [{"Title": d["title"], "Category": d["category"], "File": d["source"], "Sections": len(d["sections"]),
             "Passages": int(chunk_counts.get(doc_id, 0)), "Owner": d.get("owner") or "",
             "Updated": d.get("last_updated") or "", "Link": d.get("url") or ""}
            for doc_id, d in sorted(assistant.catalog.items())]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Link": st.column_config.LinkColumn()})

    with st.expander("Browse document structure"):
        doc_id = st.selectbox("Document", sorted(assistant.catalog), format_func=lambda d: assistant.catalog[d]["title"])
        for s in assistant.catalog[doc_id]["sections"]:
            indent = "&nbsp;" * 6 if s.get("parent") else ""
            page = f" · p. {s['page']}" if s.get("page") else ""
            st.markdown(f"{indent}• {s['heading']}{page}", unsafe_allow_html=True)

    st.subheader("Add a document")
    st.caption("PDF, Markdown or text. It is chunked by section, added to BM25 immediately and embedded once.")
    c1, c2 = st.columns([3, 1])
    up = c1.file_uploader("Document", type=["pdf", "md", "markdown", "txt"], label_visibility="collapsed")
    category = c2.selectbox("Category", sorted(set(cats) | {"hr", "it", "security", "finance", "customer",
                                                             "onboarding", "general"}))
    if up and st.button("Index document", type="primary"):
        dest = Path(KNOWLEDGE_BASE_DIR) / category / Path(up.name).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(up.getvalue())
        with st.spinner("Indexing..."):
            res = assistant.add_document(str(dest), category)
        if res["ok"]:
            st.success(f"Indexed **{res['title']}** as {res['chunks']} passages. Dense: {res['vector_sync']}")
        else:
            dest.unlink(missing_ok=True)
            st.error(res["error"])

    st.subheader("Try retrieval")
    q = st.text_input("Query", placeholder="e.g. doctor's certificate")
    if q:
        trace = {}
        out = assistant.router.search(q, mode=mode, user_type=ss.user_type, trace=trace)
        st.caption(f"{out['search_type']} · queries: {out['queries']}")
        st.dataframe(pd.DataFrame([{"Document": r["metadata"]["title"], "Section": r["metadata"]["section"],
                                    "Relevance": r["relevance"], "RRF": round(r["combined_score"], 4),
                                    "BM25": round(r["bm25_score"], 2), "Cosine": round(r["vector_score"], 3),
                                    "Matched by": ", ".join(r["matched_by"])} for r in out["results"]]),
                     hide_index=True, width="stretch")


# ============================================================================
# ESCALATIONS & GAPS PAGE
# ============================================================================

def escalations_page() -> None:
    st.header("🙋 Escalations & knowledge gaps")
    tickets = assistant.escalations.list_tickets()
    gaps = assistant.gaps.list_gaps()
    c = st.columns(3)
    c[0].markdown(uh.kpi_html(sum(1 for t in tickets if t["status"] != "resolved"), "open tickets"), unsafe_allow_html=True)
    c[1].markdown(uh.kpi_html(len(tickets), "tickets total"), unsafe_allow_html=True)
    c[2].markdown(uh.kpi_html(len(gaps), "knowledge gaps logged"), unsafe_allow_html=True)

    st.subheader("Tickets")
    if not tickets:
        st.caption("No tickets yet. Askly opens one when a user asks for a human, when a question is urgent, "
                   "or after two unanswered questions in a row.")
    else:
        df = pd.DataFrame([{k: t.get(k) for k in ("ticket_id", "created_at", "status", "priority", "team",
                                                    "user_name", "user_type", "question", "reason")}
                           for t in reversed(tickets)])
        st.dataframe(df, hide_index=True, width="stretch")
        c1, c2, c3 = st.columns([2, 2, 1])
        tid = c1.selectbox("Ticket", [t["ticket_id"] for t in reversed(tickets)])
        new_status = c2.selectbox("Set status", ["open", "in_progress", "resolved"])
        if c3.button("Update", width="stretch"):
            assistant.escalations.update_status(tid, new_status)
            st.rerun()
        t = next(t for t in tickets if t["ticket_id"] == tid)
        with st.expander(f"Conversation attached to {tid}"):
            for m in t.get("conversation", []):
                st.markdown(f"**{m['role']}:** {m['content']}")
            if t.get("best_sources"):
                st.caption("Closest sources: " + "; ".join(t["best_sources"]))

    st.subheader("Most frequent knowledge gaps")
    st.caption("Questions the documentation could not answer (or answers rated 👎) - the doc owners' to-do list.")
    summary = assistant.gaps.gap_summary(15)
    if summary:
        st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
        with st.expander("All gap events"):
            st.dataframe(pd.DataFrame(gaps[::-1]), hide_index=True, width="stretch")
    else:
        st.caption("No gaps logged yet.")


# ============================================================================
# QUALITY PAGE
# ============================================================================

def quality_page() -> None:
    st.header("📏 Quality & feedback")
    fb = assistant.feedback.stats()
    c = st.columns(4)
    c[0].markdown(uh.kpi_html(fb["total"], "ratings"), unsafe_allow_html=True)
    c[1].markdown(uh.kpi_html(fb["helpful"], "👍 helpful"), unsafe_allow_html=True)
    c[2].markdown(uh.kpi_html(fb["not_helpful"], "👎 not helpful"), unsafe_allow_html=True)
    rate = f"{fb['helpful_rate']:.0%}" if fb["helpful_rate"] is not None else "-"
    c[3].markdown(uh.kpi_html(rate, "helpful rate"), unsafe_allow_html=True)
    boosts = assistant.feedback.chunk_boosts()
    if boosts:
        with st.expander("How feedback is changing retrieval"):
            rows = [{"Passage": f"{assistant.indexer.by_id[cid].metadata['title']} · "
                                f"{assistant.indexer.by_id[cid].metadata['section']}" if cid in assistant.indexer.by_id else cid,
                     "Boost": f"{b:+.0%}"} for cid, b in sorted(boosts.items(), key=lambda x: -abs(x[1]))]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("RAGAS evaluation")
    evaluator = get_evaluator()
    items = evaluator.load_testset()
    llm_on = assistant.llm is not None
    limit = st.slider("Questions from data/eval/testset.json", 1, len(items), min(len(items), 5 if llm_on else len(items)))
    est = evaluator.estimate_cost(limit, llm_on)
    if llm_on:
        st.warning(f"About {est['estimated_llm_calls']} LLM calls - roughly {est['estimated_minutes']} min on a "
                   "5 requests/minute free tier.", icon="⏱️")
    else:
        st.info("No LLM key - this runs the offline token-overlap proxies (not RAGAS) plus retrieval metrics.")
    if st.button("Run evaluation", type="primary"):
        with st.spinner("Evaluating..."):
            report = evaluator.run(limit=limit, use_ragas=llm_on)
        ss.last_report = report.to_dict()

    rep = ss.get("last_report")
    if rep:
        st.caption(f"Method: **{rep['method']}** · overall **{rep['overall_score']:.3f}**"
                   + (" · ⚠️ the judge failed on some metrics (0 = unknown)" if rep["judge_failed"] else ""))
        mcols = st.columns(len(rep["metrics"]))
        for col, (k, v) in zip(mcols, rep["metrics"].items()):
            col.metric(k.replace("_", " "), f"{v:.2f}")
        rcols = st.columns(len(rep["retrieval"]))
        for col, (k, v) in zip(rcols, rep["retrieval"].items()):
            col.metric(k.replace("_", " "), f"{v:.2f}")
        cols = ["id", "question", "faithfulness", "answer_relevancy", "context_precision", "context_recall",
                "answer_correctness", "source_hit", "knowledge_gap", "confidence"]
        st.dataframe(pd.DataFrame([{k: r.get(k) for k in cols} for r in rep["rows"]]),
                     hide_index=True, width="stretch")

    past = evaluator.load_past_runs()
    if len(past) > 1:
        st.subheader("Past runs")
        st.line_chart(past.set_index("timestamp")[["overall", "faithfulness", "answer_relevancy"]])


# ============================================================================
# API KEYS PAGE
# ============================================================================

def _test_llm_connection() -> str:
    from utils.utils import get_llm, get_embeddings, llm_text, resolve_provider
    provider = resolve_provider()
    if not provider:
        return "❌ No Gemini or OpenAI key is set."
    try:
        reply = llm_text(get_llm().invoke("Reply with the single word: OK")).strip()
        vec = get_embeddings().embed_query("hello")
        return f"✅ {provider}: chat model replied '{reply[:20]}', embeddings returned {len(vec)} dimensions."
    except Exception as e:
        return f"❌ {provider} call failed: {str(e)[:400]}"


def _apply_session_keys(values: dict, save: bool) -> None:
    """Use pasted keys in THIS app process (and optionally persist them to the git-ignored .env)."""
    import os
    import core.vector_store as vs_mod
    from utils import utils as u
    for env, val in values.items():
        if not val:
            continue
        os.environ[env] = val
        setattr(app_config, env, val)
        if env == "PINECONE_API_KEY":
            vs_mod.PINECONE_API_KEY = val
    u._llm_cache.clear()
    if save:
        env_path = app_config.PROJECT_ROOT / ".env"
        lines = env_path.read_text().splitlines() if env_path.exists() else []
        for env, val in values.items():
            if val:
                lines = [ln for ln in lines if not ln.startswith(f"{env}=")] + [f"{env}={val}"]
        env_path.write_text("\n".join(lines) + "\n")
    st.cache_resource.clear()          # rebuild the assistant with the new keys


def api_keys_page() -> None:
    st.header("🔑 API keys")
    st.markdown(
        "Askly runs **without any key** (offline: BM25 keyword search + quoted, cited answers). "
        "Add **one LLM key - Gemini _or_ OpenAI** - to unlock written answers, semantic (dense) search, "
        "follow-up rewriting and RAGAS evaluation. **Pinecone is optional**; FAISS is used locally by default.")
    keys = key_status()
    cols = st.columns(len(KEY_GUIDE))
    for col, k in zip(cols, KEY_GUIDE):
        val = getattr(app_config, k["env"], "")
        col.markdown(uh.kpi_html("✅ set" if keys[k["env"]] else "not set", k["env"]), unsafe_allow_html=True)
        if val:
            col.caption(f"`{mask(val)}`")
    st.write("")

    for k in KEY_GUIDE:
        with st.expander(f"{'✅' if keys[k['env']] else '⚪'} **{k['name']}** - {k['needed']}",
                         expanded=not keys[k["env"]] and k["env"] == "GOOGLE_API_KEY"):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**Unlocks:** {k['unlocks']}")
            c2.link_button("Get API key ↗", k["url"], width="stretch", type="primary")
            st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(k["steps"], start=1)))
            st.caption(f"💰 {k['cost']}")
            st.caption(f"💡 {k['tip']}")
            st.markdown(f"[Official docs ↗]({k['docs']})")

    st.subheader("Where to put the keys")
    st.markdown("Copy `.env.example` to `.env` in the project root, fill in the key(s) and restart the app. "
                "`.env` is git-ignored, so keys never end up on GitHub.")
    st.code(ENV_SNIPPET, language="bash")

    st.subheader("Test the connection")
    if st.button("Run a test call"):
        with st.spinner("Calling the provider..."):
            st.write(_test_llm_connection())

    st.subheader("Or paste a key for this run")
    st.caption("For use on your own machine: the key is applied to this running app process (anyone using this "
               "same server would use it too). Tick 'save' to also write it to your local, git-ignored `.env`.")
    with st.form("keys_form", clear_on_submit=True):
        provider = st.radio("Provider", ["Gemini", "OpenAI"], horizontal=True)
        llm_key = st.text_input("API key", type="password", placeholder="AIza... or sk-...")
        pc_key = st.text_input("Pinecone API key (optional)", type="password", placeholder="pcsk_...")
        save = st.checkbox("Also save to .env")
        if st.form_submit_button("Apply keys", type="primary"):
            if not llm_key and not pc_key:
                st.error("Paste at least one key.")
            else:
                env = "GOOGLE_API_KEY" if provider == "Gemini" else "OPENAI_API_KEY"
                vals = {env: llm_key.strip(), "PINECONE_API_KEY": pc_key.strip()}
                if pc_key:
                    import core.vector_store as vs_mod
                    vals["VECTOR_BACKEND"] = "pinecone"
                    vs_mod.VECTOR_BACKEND = "pinecone"
                if provider == "OpenAI" and llm_key:
                    vals["LLM_PROVIDER"] = "openai"
                _apply_session_keys(vals, save)
                ss.messages = []
                st.success("Keys applied - restarting Askly with them...")
                st.rerun()


# ============================================================================
# ROUTER
# ============================================================================

{"Chat": chat_page, "Knowledge base": kb_page, "Escalations & gaps": escalations_page,
 "Quality & feedback": quality_page, "API keys": api_keys_page}[ss.page]()
