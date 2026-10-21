"""Where to get each API key Askly can use - one source of truth for the Streamlit
"API keys" page, the CLI start-up check and the README.

Askly needs NO key to run (offline mode). One LLM key (Gemini OR OpenAI) unlocks
written answers, dense retrieval, LLM query rewriting and RAGAS. Pinecone is
optional - FAISS (local) is the default vector store.
"""

import os
from typing import Dict, List, Any

KEY_GUIDE: List[Dict[str, Any]] = [
    {
        "env": "GOOGLE_API_KEY",
        "name": "Google Gemini API key",
        "provider": "Google AI Studio",
        "needed": "Recommended - one LLM key (this OR OpenAI) is enough",
        "unlocks": "LLM answers, embeddings for dense search, follow-up rewriting, RAGAS judge",
        "url": "https://aistudio.google.com/app/apikey",
        "steps": [
            "Open Google AI Studio and sign in with a Google account.",
            "Accept the Gemini API terms (first visit only).",
            "Click **Create API key** (a new Google Cloud project is fine).",
            "Copy the key (starts with `AIza`) and paste it into `.env` as `GOOGLE_API_KEY=...`.",
        ],
        "cost": "Free tier available (rate-limited - keep LLM_REQUESTS_PER_MINUTE=5). Enabling billing "
                "on the Cloud project lifts the limits.",
        "tip": "If a model name returns 404 / permission errors on the free tier, set e.g. "
               "`LLM_MODEL=gemini-2.5-flash` in `.env`.",
        "docs": "https://ai.google.dev/gemini-api/docs/api-key",
    },
    {
        "env": "OPENAI_API_KEY",
        "name": "OpenAI API key",
        "provider": "OpenAI Platform",
        "needed": "Alternative to Gemini",
        "unlocks": "Same as Gemini (gpt-4o-mini + text-embedding-3-small by default)",
        "url": "https://platform.openai.com/api-keys",
        "steps": [
            "Sign up / log in at platform.openai.com and verify your email and phone.",
            "Add a payment method under **Settings -> Billing** (requests fail with "
            "`insufficient_quota` without it).",
            "Open **API keys** -> **Create new secret key**, copy it (shown only once).",
            "Paste it into `.env` as `OPENAI_API_KEY=...` (and `LLM_PROVIDER=openai` if you also have a Gemini key).",
        ],
        "cost": "Pay-as-you-go, no permanent free tier. Set a monthly budget limit under Billing.",
        "tip": "Billing page: https://platform.openai.com/settings/organization/billing",
        "docs": "https://platform.openai.com/docs/quickstart",
    },
    {
        "env": "PINECONE_API_KEY",
        "name": "Pinecone API key",
        "provider": "Pinecone",
        "needed": "Optional - only if VECTOR_BACKEND=pinecone (FAISS is the local default)",
        "unlocks": "Managed, persistent vector index instead of the local FAISS files",
        "url": "https://app.pinecone.io/organizations/-/projects",
        "steps": [
            "Sign up at app.pinecone.io (the Starter plan is enough for this project).",
            "Open your project -> **API keys** -> **+ API key**, name it, create it.",
            "Copy the key immediately - it can't be viewed again.",
            "In `.env`: `PINECONE_API_KEY=...` and `VECTOR_BACKEND=pinecone`. Askly creates the "
            "`askly` index (dimension 768) on first run.",
        ],
        "cost": "Starter plan available; check Pinecone pricing for current limits.",
        "tip": "Embeddings still come from Gemini/OpenAI, so Pinecone also needs one LLM key.",
        "docs": "https://docs.pinecone.io/guides/projects/manage-api-keys",
    },
]

ENV_SNIPPET = """# .env  (project root - git-ignored, never commit it)
GOOGLE_API_KEY=AIza...your-key...      # or OPENAI_API_KEY=sk-...
# VECTOR_BACKEND=pinecone              # optional
# PINECONE_API_KEY=pcsk_...            # optional
"""


def key_status() -> Dict[str, bool]:
    """Which keys are currently set (from .env or the process environment)."""
    from utils import config
    return {k["env"]: bool(getattr(config, k["env"], "") or os.environ.get(k["env"], "")) for k in KEY_GUIDE}


def mask(value: str) -> str:
    return (value[:4] + "…" + value[-4:]) if value and len(value) > 10 else ("set" if value else "")
