"""
Common utilities used across Askly: logging, text helpers, PDF loading,
LLM / embedding factories (Gemini or OpenAI), a shared rate limiter, the shared
BM25 tokenizer and small JSONL persistence helpers.
"""

import json
import logging
import re
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Create a standardized logger with timestamp formatting."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "google_genai", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Collapse runs of spaces/tabs, keep paragraph breaks, strip control characters."""
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\r", "\n")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_length: int = 8000) -> str:
    """Cut text at max length and add an ellipsis if truncated."""
    if not text:
        return ""
    return text if len(text) <= max_length else text[:max_length].rstrip() + "..."


_STOPWORDS = frozenset("""
    a an the and or but if is are was were be been being to of in on for with as at by
    from this that these those it its it's you your our we they he she his her them
    will would can could should shall may might must not no do does did doing have has
    had having i me my mine what which who whom how when where why there here about
    so than then too very just also any all some into over under up down out please
    tell know want need get am like guidance follow help thing
    won don doesn didn isn aren wasn couldn shouldn wouldn haven hasn ll ve re
    much many way ways anyone someone something got getting gotten
""".split())


# Spelling variants normalised on BOTH sides before tokenizing, so "Wi-Fi" matches
# "wifi" and "two-factor authentication" matches "2FA". Extend for your own domain.
_NORMALISE = [
    (r"\bwi-?fi\b", "wifi"), (r"\be-mail\b", "email"),
    (r"\b(two|2)[- ]?(factor|step)( authentication| verification)?\b", "2fa"),
    (r"\bmulti[- ]?factor( authentication)?\b", "2fa"), (r"\bmfa\b", "2fa"),
    (r"\b(log|sign)[- ]?(in|on)\b", "login"), (r"\bpay ?days?\b", "pay date"),
    (r"\bpay ?slips?\b", "payslip"), (r"\bpass ?words?\b", "password"),
    (r"\bmoney back\b", "refund"), (r"\bpaid time off\b", "pto"), (r"\bvacations?\b", "pto"), (r"\bcell ?phone|mobile phone\b", "phone"),
    (r"\b401\s*\(?k\)?", "401k"), (r"\bset[- ]?up\b", "setup"), (r"\broll[- ]?back\b", "rollback"), (r"\blos(e|es|ing|t)\b", "lost"), (r"\bstolen\b", "stolen"),
]


def normalise_text(text: str) -> str:
    text = (text or "").lower()
    for pattern, repl in _NORMALISE:
        text = re.sub(pattern, repl, text)
    return text


def tokenize(text: str) -> List[str]:
    """Shared BM25 tokenizer (same rule at index AND query time - lesson from HireFlow
    HW11): lowercase, split on non-alphanumerics, drop stopwords and 1-char tokens,
    and apply a tiny suffix-stripper so 'resetting'/'reset' and 'leaves'/'leave' match."""
    tokens = []
    for t in re.findall(r"[a-z0-9]+", normalise_text(text)):
        if t in _STOPWORDS or len(t) <= 1:
            continue
        tokens.append(_stem(t))
    return tokens


def _stem(token: str) -> str:
    """Very light stemmer - enough for BM25 on short helpdesk queries. Applied to both
    documents and queries, so stems only need to be consistent, not pretty:
    escalate / escalated / escalation(s) -> 'escalat'; leave / leaves -> 'leav'."""
    stem = _strip_suffix(token)
    if stem.endswith("ation") and len(stem) > 7:     # escalation -> escalate
        stem = stem[:-5] + "ate"
    if stem.endswith("e") and len(stem) > 4:         # escalate -> escalat (matches 'escalated')
        stem = stem[:-1]
    return stem


def _strip_suffix(token: str) -> str:
    for suffix, min_len in (("ing", 5), ("ies", 5), ("es", 5), ("ed", 4), ("s", 4)):
        if token.endswith(suffix) and len(token) > min_len:
            if suffix == "ies":
                return token[:-3] + "y"
            if suffix == "es" and not token.endswith(("ses", "xes", "ches", "shes")):
                return token[:-1]
            stem = token[: -len(suffix)]
            if suffix == "ed" and stem.endswith("i"):      # 'applied' -> 'apply'
                return stem[:-1] + "y"
            # 'resetting' -> 'resett' -> 'reset'
            if suffix in ("ing", "ed") and len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiouls":
                stem = stem[:-1]
            return stem
    return token


def query_term_weights(query: str, idf) -> Dict[str, float]:
    """IDF weight per query term. For heuristic follow-up rewrites
    ("what about sick leave (follow-up to: How do I apply ...)") the NEW part counts
    double and the inherited context half, so the answer is about sick leave, not the
    previous topic."""
    core, _, context = (query or "").partition("(follow-up to:")
    weights: Dict[str, float] = {}
    # "what about sick leave" names its own topic -> the old question barely matters;
    # "what about that one?" doesn't -> lean on the earlier question.
    ctx_weight = 0.15 if len(set(tokenize(core))) >= 2 else 0.5
    for t in set(tokenize(context)):
        weights[t] = max(idf(t), 0.05) * ctx_weight
    for t in set(tokenize(core)):
        weights[t] = max(idf(t), 0.05) * (2.0 if context else 1.0)
    return weights


def answer_units(text: str) -> List[str]:
    """Split a passage into answerable units: each bullet / numbered line is one unit,
    prose is split into sentences, code fences are skipped. Markdown emphasis removed."""
    units, in_fence, label = [], False, None
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence or not s:
            continue
        s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
        m = re.match(r"^(?:[-*\u2022]|\d+[.)])\s+(.*)$", s)
        if m:
            item = m.group(1).strip()
            if len(item) > 2:
                # keep the group a bullet belongs to: "Gold Tier (...) - Critical Issues: 30-minute ..."
                units.append(f"{label} - {item}" if label else item)
        elif len(s) <= 90 and not s.endswith("."):
            label = s.rstrip(":")          # a short header line like "Platinum Tier (ACV > $500K)"
        else:
            label = None
            units.extend(split_sentences(s))
    return units


def split_sentences(text: str) -> List[str]:
    """Rough sentence splitter used by the offline (extractive) answer mode."""
    text = re.sub(r"\s+", " ", text or "").strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p.strip() for p in parts if len(p.strip()) > 2]

# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------

def load_pdf_pages(file_path: str) -> List[str]:
    """Return the text of each page of a PDF (index 0 = page 1). [] on failure."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        return [clean_text(page.extract_text() or "") for page in reader.pages]
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
        return []


def load_pdf(file_path: str) -> str:
    """Extract all text content from a PDF as one string."""
    return "\n".join(load_pdf_pages(file_path)).strip()

# ---------------------------------------------------------------------------
# LLM / embeddings (Gemini or OpenAI)
# ---------------------------------------------------------------------------

def resolve_provider() -> Optional[str]:
    """Which LLM provider to use: 'gemini', 'openai' or None (offline).

    LLM_PROVIDER=auto picks Gemini if GOOGLE_API_KEY is set, else OpenAI if
    OPENAI_API_KEY is set. An explicit provider without its key -> None.
    """
    from utils.config import LLM_PROVIDER, GOOGLE_API_KEY, OPENAI_API_KEY
    if LLM_PROVIDER == "gemini":
        return "gemini" if GOOGLE_API_KEY else None
    if LLM_PROVIDER == "openai":
        return "openai" if OPENAI_API_KEY else None
    if GOOGLE_API_KEY:
        return "gemini"
    if OPENAI_API_KEY:
        return "openai"
    return None


def default_llm_model(provider: Optional[str]) -> str:
    from utils.config import LLM_MODEL
    if LLM_MODEL:
        return LLM_MODEL
    return {"gemini": "gemini-3.5-flash-lite", "openai": "gpt-4o-mini"}.get(provider or "", "")


_rate_limiter = None
_rate_lock = threading.Lock()


def get_rate_limiter():
    """ONE shared limiter for every LLM call in the app (if each component built its
    own, together they would still exceed the provider's per-minute quota)."""
    global _rate_limiter
    with _rate_lock:
        if _rate_limiter is None:
            from langchain_core.rate_limiters import InMemoryRateLimiter
            from utils.config import LLM_REQUESTS_PER_MINUTE
            _rate_limiter = InMemoryRateLimiter(
                requests_per_second=max(LLM_REQUESTS_PER_MINUTE, 1) / 60,
                check_every_n_seconds=0.25,
                max_bucket_size=1,
            )
    return _rate_limiter


_llm_cache: Dict[str, Any] = {}


def get_llm(temperature: Optional[float] = None):
    """Return a LangChain chat model for the configured provider, or None (offline)."""
    from utils.config import GOOGLE_API_KEY, OPENAI_API_KEY, LLM_TEMPERATURE
    provider = resolve_provider()
    if not provider:
        return None
    temp = LLM_TEMPERATURE if temperature is None else temperature
    key = f"{provider}:{temp}"
    if key in _llm_cache:
        return _llm_cache[key]
    try:
        model = default_llm_model(provider)
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=model, google_api_key=GOOGLE_API_KEY,
                                         temperature=temp, rate_limiter=get_rate_limiter())
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model, api_key=OPENAI_API_KEY, temperature=temp,
                             rate_limiter=get_rate_limiter())
        _llm_cache[key] = llm
        return llm
    except Exception as e:
        logger.error(f"Could not create LLM client ({provider}): {e}")
        return None


def get_embeddings():
    """Return a LangChain embeddings model for the configured provider, or None."""
    from utils.config import GOOGLE_API_KEY, OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSION
    provider = resolve_provider()
    if not provider:
        return None
    try:
        if provider == "gemini":
            import asyncio
            try:
                asyncio.get_event_loop()
            except RuntimeError:  # Streamlit runs scripts in threads without a loop
                asyncio.set_event_loop(asyncio.new_event_loop())
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL or "models/gemini-embedding-001",
                google_api_key=GOOGLE_API_KEY,
                # gemini-embedding-001 returns 3072 numbers unless told otherwise
                output_dimensionality=EMBEDDING_DIMENSION,
            )
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=EMBEDDING_MODEL or "text-embedding-3-small",
                                api_key=OPENAI_API_KEY, dimensions=EMBEDDING_DIMENSION)
    except Exception as e:
        if is_quota_error(e):
            logger.warning("API quota exceeded - dense retrieval disabled.")
        else:
            logger.error(f"Embeddings unavailable: {e}")
        return None


def is_quota_error(error: Exception) -> bool:
    """Detect API quota / rate-limit errors for graceful handling."""
    s = str(error).lower()
    return "429" in s or "quota" in s or "rate limit" in s or "resource_exhausted" in s


def llm_text(response: Any) -> str:
    """Normalise a chat-model response to plain text (Gemini 3 can return content parts)."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def parse_json_block(text: str) -> Optional[Any]:
    """Pull the first JSON object/array out of an LLM reply (tolerates ```json fences)."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None

# ---------------------------------------------------------------------------
# JSONL persistence (tickets, gaps, feedback)
# ---------------------------------------------------------------------------

_io_lock = threading.Lock()


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _io_lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _io_lock, path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
