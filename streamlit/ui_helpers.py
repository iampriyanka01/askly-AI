"""UI building blocks for the Askly Streamlit app: CSS, badges and source cards.

Kept separate from app.py (same split as HireFlow) so the page code reads as
"what goes where" while the HTML/CSS details live here.
"""

import html
import re
from typing import Dict, Any, List

CATEGORY_STYLE = {
    "hr": ("HR", "#0f766e"), "it": ("IT", "#1d4ed8"), "security": ("Security", "#b45309"),
    "finance": ("Finance", "#7c3aed"), "customer": ("Customer", "#be123c"),
    "onboarding": ("Onboarding", "#0369a1"), "general": ("General", "#475569"),
}

CSS = """
<style>
:root { --ask-ink:#0f172a; --ask-muted:#64748b; --ask-line:#e2e8f0; --ask-soft:#f8fafc; --ask-accent:#4f46e5; }
@media (prefers-color-scheme: dark) {
  :root { --ask-ink:#e2e8f0; --ask-muted:#94a3b8; --ask-line:#334155; --ask-soft:#111827; --ask-accent:#818cf8; }
}
.block-container { padding-top: 1.6rem; max-width: 1100px; }
.ask-hero { padding: 1.2rem 1.4rem; border:1px solid var(--ask-line); border-radius: 14px;
            background: linear-gradient(135deg, rgba(79,70,229,.10), rgba(14,165,233,.06)); margin-bottom: 1rem; }
.ask-hero h1 { font-size: 1.7rem; margin: 0 0 .25rem 0; letter-spacing: -.01em; }
.ask-hero p { margin: 0; color: var(--ask-muted); }
.ask-badges { display:flex; flex-wrap:wrap; gap:.35rem; margin:.35rem 0 .2rem 0; }
.ask-badge { font-size:.74rem; padding:.12rem .55rem; border-radius:999px; border:1px solid var(--ask-line);
             color: var(--ask-muted); background: var(--ask-soft); white-space:nowrap; }
.ask-badge.good { color:#047857; border-color:rgba(4,120,87,.35); background:rgba(16,185,129,.08); }
.ask-badge.mid  { color:#b45309; border-color:rgba(180,83,9,.35);  background:rgba(245,158,11,.08); }
.ask-badge.low  { color:#b91c1c; border-color:rgba(185,28,28,.35); background:rgba(239,68,68,.08); }
.ask-src { border:1px solid var(--ask-line); border-left: 4px solid var(--cat, #475569); border-radius:10px;
           padding:.6rem .8rem; margin:.45rem 0; background: var(--ask-soft); }
.ask-src .head { display:flex; justify-content:space-between; gap:.6rem; font-weight:600; font-size:.9rem; }
.ask-src .head .num { color: var(--cat, #475569); margin-right:.35rem; }
.ask-src .meta { font-size:.76rem; color: var(--ask-muted); margin:.1rem 0 .35rem 0; }
.ask-src .snip { font-size:.84rem; line-height:1.45; color: var(--ask-ink); }
.ask-src a { font-size:.78rem; }
.ask-src mark { background: rgba(250,204,21,.30); color: inherit; padding: 0 .1em; border-radius: 3px; }
.ask-cat { display:inline-block; font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em;
           color:#fff; background: var(--cat, #475569); border-radius:6px; padding:.05rem .4rem; }
.ask-related { font-size:.84rem; color: var(--ask-muted); margin-top:.3rem; }
.ask-kpi { border:1px solid var(--ask-line); border-radius:12px; padding:.7rem .9rem; background: var(--ask-soft); }
.ask-kpi .v { font-size:1.45rem; font-weight:700; } .ask-kpi .l { font-size:.78rem; color: var(--ask-muted); }
.ask-trace code { font-size: .78rem; }
</style>
"""


def md_safe(text: str) -> str:
    """Streamlit renders $...$ as LaTeX - "$200K - $500K" would turn into math. Escape it."""
    return (text or "").replace("$", "\\$")


def esc(text: Any) -> str:
    return html.escape("" if text is None else str(text))


def confidence_class(conf: float) -> str:
    return "good" if conf >= 0.6 else ("mid" if conf >= 0.35 else "low")


def badges_html(resp: Dict[str, Any]) -> str:
    items = []
    if resp.get("intent") == "kb_question" and not resp.get("knowledge_gap"):
        conf = resp.get("confidence", 0.0)
        items.append(f'<span class="ask-badge {confidence_class(conf)}">grounded · confidence {conf:.0%}</span>')
    if resp.get("knowledge_gap"):
        items.append('<span class="ask-badge low">knowledge gap logged</span>')
    if resp.get("ticket_id"):
        items.append(f'<span class="ask-badge mid">ticket {esc(resp["ticket_id"])}</span>')
    if resp.get("is_follow_up"):
        items.append(f'<span class="ask-badge" title="{esc(resp.get("standalone_query"))}">'
                     f'follow-up understood as: {esc(resp.get("standalone_query"))[:90]}</span>')
    if resp.get("search_type"):
        items.append(f'<span class="ask-badge">{esc(resp["search_type"].replace("_", " "))}</span>')
    if resp.get("generation_mode") == "extractive":
        items.append('<span class="ask-badge" title="No LLM key configured - answer is quoted from the documents">'
                     'extractive (offline)</span>')
    return f'<div class="ask-badges">{"".join(items)}</div>' if items else ""


def _highlight(snippet: str, terms: List[str]) -> str:
    out = esc(snippet)
    for t in sorted({t for t in terms if len(t) > 3}, key=len, reverse=True)[:8]:
        out = re.sub(rf"(?i)\b({re.escape(t)}\w*)", r"<mark>\1</mark>", out)
    return out


def source_card_html(c: Dict[str, Any], terms: List[str] = ()) -> str:
    label, color = CATEGORY_STYLE.get(c.get("category") or "general", CATEGORY_STYLE["general"])
    where = " · ".join(x for x in [
        f"Section {c['section']}" if c.get("section") and c["section"][:1].isdigit() else c.get("section"),
        f"p. {c['page']}" if c.get("page") else None, c.get("source")] if x)
    link = f'<a href="{esc(c["url"])}" target="_blank">{esc(c["url"])}</a>' if c.get("url") else ""
    snippet = re.sub(r"\*\*(.+?)\*\*", r"\1", c.get("snippet", ""))   # drop Markdown bold markers
    snippet = snippet if len(snippet) < 420 else snippet[:420].rsplit(" ", 1)[0] + " …"
    return (f'<div class="ask-src" style="--cat:{color}"><div class="head"><span><span class="num">[{c["number"]}]</span>'
            f'{esc(c.get("doc_title"))}</span><span class="ask-cat">{label}</span></div>'
            f'<div class="meta">{esc(where)} · relevance {c.get("relevance", 0):.2f}</div>'
            f'<div class="snip">{_highlight(snippet, list(terms))}</div>{link}</div>')


def related_html(related: List[Dict[str, Any]]) -> str:
    if not related:
        return ""
    items = " · ".join(f'<b>{esc(r["title"])}</b> <span>({esc(r["doc_title"])})</span>' for r in related)
    return f'<div class="ask-related">📎 Related: {items}</div>'


def kpi_html(value: Any, label: str) -> str:
    return f'<div class="ask-kpi"><div class="v">{esc(value)}</div><div class="l">{esc(label)}</div></div>'


def hero_html(n_docs: int, n_chunks: int, mode: str) -> str:
    return (f'<div class="ask-hero"><h1>💬 Askly</h1><p>Your smart helpdesk assistant - answers grounded in '
            f'{n_docs} approved documents ({n_chunks} indexed passages), with sources, follow-ups and a human '
            f'hand-off when the docs don\'t cover it. <i>Mode: {esc(mode)}</i></p></div>')
