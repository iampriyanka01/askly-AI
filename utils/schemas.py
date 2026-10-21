"""Data schemas for Askly (Pydantic models shared across the pipeline and the UI)."""

from datetime import datetime, timezone
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """One source shown under an answer ("Transparent Answers")."""
    number: int = Field(..., description="Citation marker used in the answer text, e.g. [1]")
    chunk_id: str
    doc_title: str
    section: Optional[str] = None
    page: Optional[int] = None
    category: Optional[str] = None
    source: str = Field(..., description="File name of the source document")
    url: Optional[str] = None
    snippet: str = Field(..., description="The passage the answer is grounded in")
    relevance: float = Field(0.0, description="Rerank relevance 0-1")

    def label(self) -> str:
        """Human-readable reference, e.g. 'HR Handbook, Section 4.2 Sick Leave, p. 3'."""
        parts = [self.doc_title]
        if self.section:
            parts.append(f"Section {self.section}" if self.section[:1].isdigit() else self.section)
        if self.page:
            parts.append(f"p. {self.page}")
        return ", ".join(parts)


class RelatedResource(BaseModel):
    """A proactive pointer to something the user may want next."""
    title: str
    doc_title: str
    section: Optional[str] = None
    url: Optional[str] = None


# ---------------------------------------------------------------------------
# Query understanding
# ---------------------------------------------------------------------------

Intent = Literal["greeting", "kb_question", "escalation_request", "thanks", "out_of_scope"]


class QueryAnalysis(BaseModel):
    """Result of intent detection + follow-up resolution for one user turn."""
    intent: Intent = "kb_question"
    is_follow_up: bool = False
    standalone_query: str = Field(..., description="The question rewritten to stand on its own")
    category_hint: Optional[str] = Field(None, description="hr / it / security / finance / customer / onboarding")
    method: str = Field("heuristic", description="'llm' or 'heuristic'")


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------

class AskResponse(BaseModel):
    """Everything the UI / CLI needs to render one assistant turn."""
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    follow_ups: List[str] = Field(default_factory=list, description="Proactive next-step questions")
    related: List[RelatedResource] = Field(default_factory=list)
    confidence: float = 0.0
    grounded: bool = True
    knowledge_gap: bool = False
    escalated: bool = False
    ticket_id: Optional[str] = None
    intent: str = "kb_question"
    standalone_query: str = ""
    is_follow_up: bool = False
    search_type: str = ""
    generation_mode: str = Field("llm", description="'llm' or 'extractive' (offline)")
    contexts: List[str] = Field(default_factory=list, description="Raw retrieved text (for RAGAS)")
    trace: dict = Field(default_factory=dict)
    turn_id: str = ""


# ---------------------------------------------------------------------------
# Escalation, gaps, feedback
# ---------------------------------------------------------------------------

class EscalationTicket(BaseModel):
    ticket_id: str
    created_at: str = Field(default_factory=_now)
    session_id: str
    user_name: Optional[str] = None
    user_type: Optional[str] = None
    question: str
    standalone_query: str
    reason: str
    team: str
    priority: Literal["low", "normal", "high"] = "normal"
    status: Literal["open", "in_progress", "resolved"] = "open"
    conversation: List[dict] = Field(default_factory=list)
    best_sources: List[str] = Field(default_factory=list)


class KnowledgeGap(BaseModel):
    created_at: str = Field(default_factory=_now)
    query: str
    reason: str
    top_relevance: float = 0.0
    category_hint: Optional[str] = None
    session_id: Optional[str] = None


class FeedbackRecord(BaseModel):
    created_at: str = Field(default_factory=_now)
    session_id: str
    turn_id: str
    question: str
    answer: str
    rating: Literal[1, -1]
    comment: Optional[str] = None
    chunk_ids: List[str] = Field(default_factory=list)
