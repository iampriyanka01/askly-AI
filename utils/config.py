"""Configuration management for Askly.

Every setting can be overridden from a `.env` file in the project root (see
`.env.example`). Nothing here is required to *start* Askly: with no API keys
the assistant still runs in offline mode (BM25 keyword retrieval + extractive,
cited answers), and tells you so in the UI.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class AppConfig(BaseSettings):
    """All Askly settings in one place (read from environment / .env)."""

    model_config = SettingsConfigDict(env_file=str(PROJECT_ROOT / ".env"),
                                      case_sensitive=False, extra="ignore")

    # ---- LLM backbone -------------------------------------------------------
    # "gemini" (default, like HireFlow) or "openai". "auto" picks whichever key is set.
    LLM_PROVIDER: str = Field(default="auto")
    GOOGLE_API_KEY: str = Field(default="")
    OPENAI_API_KEY: str = Field(default="")
    # Leave blank to use the provider default (see utils.utils.default_llm_model()).
    LLM_MODEL: str = Field(default="")
    LLM_TEMPERATURE: float = Field(default=0.1)
    # Free tiers are tight (Gemini flash-lite: ~5 requests/minute). One shared limiter.
    LLM_REQUESTS_PER_MINUTE: int = Field(default=5)

    # ---- Embeddings ---------------------------------------------------------
    # Blank = provider default (gemini-embedding-001 / text-embedding-3-small).
    EMBEDDING_MODEL: str = Field(default="")
    EMBEDDING_DIMENSION: int = Field(default=768)

    # ---- Dense vector store -------------------------------------------------
    # "faiss" (local, default - no extra account needed) or "pinecone".
    VECTOR_BACKEND: str = Field(default="faiss")
    FAISS_INDEX_DIR: str = Field(default=str(PROJECT_ROOT / "data" / "index" / "faiss"))
    PINECONE_API_KEY: str = Field(default="")
    PINECONE_INDEX_NAME: str = Field(default="askly")
    PINECONE_NAMESPACE: str = Field(default="knowledge_base")
    PINECONE_METRIC: str = Field(default="cosine")

    # ---- Knowledge base + chunking -----------------------------------------
    KNOWLEDGE_BASE_DIR: str = Field(default=str(PROJECT_ROOT / "data" / "knowledge_base"))
    CHUNK_SIZE: int = Field(default=900)
    CHUNK_OVERLAP: int = Field(default=150)

    # ---- Retrieval ----------------------------------------------------------
    RETRIEVAL_TOP_K: int = Field(default=12)     # candidates pulled from BM25 + dense
    RERANK_TOP_N: int = Field(default=4)         # chunks that reach the LLM
    # "auto" -> cross-encoder if RERANKER_MODEL is set and installed, else LLM, else lexical
    RERANKER: str = Field(default="auto")
    RERANKER_MODEL: str = Field(default="")      # e.g. cross-encoder/ms-marco-MiniLM-L-6-v2

    # ---- Gap detection / escalation ---------------------------------------
    # Rerank relevance (0-1) below which Askly treats the KB as not covering the question.
    GAP_THRESHOLD: float = Field(default=0.35)

    # ---- Organisation -------------------------------------------------------
    # Blank = detected from the documents (e.g. "## Orion Digital Services, Inc." subtitles)
    COMPANY_NAME: str = Field(default="")
    # Example questions shown in the UI / greeting, per audience
    EXAMPLES_FILE: str = Field(default=str(PROJECT_ROOT / "data" / "examples.json"))

    # ---- Conversation -------------------------------------------------------
    MAX_HISTORY_TURNS: int = Field(default=6)

    # ---- Storage for feedback / tickets / gaps ------------------------------
    STATE_DIR: str = Field(default=str(PROJECT_ROOT / "data" / "state"))


config = AppConfig()

# Module-level aliases (same convenience pattern as HireFlow's config.py)
LLM_PROVIDER = config.LLM_PROVIDER.lower().strip()
GOOGLE_API_KEY = config.GOOGLE_API_KEY
OPENAI_API_KEY = config.OPENAI_API_KEY
LLM_MODEL = config.LLM_MODEL
LLM_TEMPERATURE = config.LLM_TEMPERATURE
LLM_REQUESTS_PER_MINUTE = config.LLM_REQUESTS_PER_MINUTE

EMBEDDING_MODEL = config.EMBEDDING_MODEL
EMBEDDING_DIMENSION = config.EMBEDDING_DIMENSION

VECTOR_BACKEND = config.VECTOR_BACKEND.lower().strip()
FAISS_INDEX_DIR = config.FAISS_INDEX_DIR
PINECONE_API_KEY = config.PINECONE_API_KEY
PINECONE_INDEX_NAME = config.PINECONE_INDEX_NAME
PINECONE_NAMESPACE = config.PINECONE_NAMESPACE
PINECONE_METRIC = config.PINECONE_METRIC

KNOWLEDGE_BASE_DIR = config.KNOWLEDGE_BASE_DIR
CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP

RETRIEVAL_TOP_K = config.RETRIEVAL_TOP_K
RERANK_TOP_N = config.RERANK_TOP_N
RERANKER = config.RERANKER.lower().strip()
RERANKER_MODEL = config.RERANKER_MODEL

GAP_THRESHOLD = config.GAP_THRESHOLD
MAX_HISTORY_TURNS = config.MAX_HISTORY_TURNS
COMPANY_NAME = config.COMPANY_NAME
EXAMPLES_FILE = config.EXAMPLES_FILE
STATE_DIR = config.STATE_DIR
