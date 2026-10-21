"""
core/memory_rag.py  -  CONVERSATION MEMORY (multi-turn context + personalisation)
=================================================================================

Unlike HireFlow's search log, Askly's memory is fed BACK into the pipeline:
the recent turns are what lets QueryAnalyzer turn "And what about sick leave?"
into "How do I apply for sick leave?", and what the answer prompt sees so
replies stay consistent across a conversation.

Per session it keeps
    messages : LangChain HumanMessage / AIMessage list. Each message carries
               additional_kwargs (standalone query, citations, turn_id, flags)
    profile  : {"name", "user_type": "employee"|"customer", "department"} -
               used to personalise retrieval and tone
    turns    : turn_id -> the full AskResponse (for feedback / escalation)
    consecutive_gaps : unanswered questions in a row (drives auto-escalation)

Memory lives in RAM (one MemoryRAG per app process; Streamlit keeps it in a
cached resource), so conversations reset when the process restarts.

Public API
    add_user_message / add_assistant_message / get_history / history_for_prompt
    set_profile / get_profile / get_turn / clear / get_memory_stats
"""

import uuid
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from utils.config import MAX_HISTORY_TURNS


class MemoryRAG:
    """Session-scoped conversation memory."""

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self.max_turns = max_turns
        self.sessions: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ sessions

    def new_session(self, profile: Optional[Dict[str, Any]] = None) -> str:
        sid = uuid.uuid4().hex[:12]
        self._session(sid)["profile"].update(profile or {})
        return sid

    def _session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {"messages": [], "profile": {"user_type": "employee"},
                                         "turns": {}, "consecutive_gaps": 0}
        return self.sessions[session_id]

    def set_profile(self, session_id: str, **profile) -> None:
        self._session(session_id)["profile"].update({k: v for k, v in profile.items() if v is not None})

    def get_profile(self, session_id: str) -> Dict[str, Any]:
        return dict(self._session(session_id)["profile"])

    def clear(self, session_id: str) -> None:
        profile = self.get_profile(session_id)
        self.sessions.pop(session_id, None)
        self._session(session_id)["profile"].update(profile)

    # ------------------------------------------------------------------ messages

    def add_user_message(self, session_id: str, content: str, standalone: Optional[str] = None) -> None:
        self._session(session_id)["messages"].append(
            HumanMessage(content=content, additional_kwargs={"standalone": standalone or content}))

    def add_assistant_message(self, session_id: str, response) -> None:
        s = self._session(session_id)
        s["messages"].append(AIMessage(content=response.answer, additional_kwargs={
            "turn_id": response.turn_id,
            "citations": [c.label() for c in response.citations],
            "knowledge_gap": response.knowledge_gap,
            "escalated": response.escalated,
        }))
        s["turns"][response.turn_id] = response
        if response.intent == "kb_question":
            s["consecutive_gaps"] = s["consecutive_gaps"] + 1 if response.knowledge_gap else 0

    def update_last_user_standalone(self, session_id: str, standalone: str) -> None:
        for msg in reversed(self._session(session_id)["messages"]):
            if isinstance(msg, HumanMessage):
                msg.additional_kwargs["standalone"] = standalone
                return

    def get_history(self, session_id: str, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """Last N turns as [{"role", "content", "standalone"?}, ...] (oldest first)."""
        msgs: List[BaseMessage] = self._session(session_id)["messages"]
        keep = (max_turns or self.max_turns) * 2
        out = []
        for m in msgs[-keep:]:
            item = {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            if isinstance(m, HumanMessage):
                item["standalone"] = m.additional_kwargs.get("standalone", m.content)
            out.append(item)
        return out

    def history_for_prompt(self, session_id: str, max_turns: int = 3, max_chars: int = 500) -> str:
        lines = []
        for m in self.get_history(session_id, max_turns):
            text = m["content"].replace("\n", " ")
            lines.append(f"{m['role'].capitalize()}: {text[:max_chars]}")
        return "\n".join(lines) or "(no earlier messages)"

    def consecutive_gaps(self, session_id: str) -> int:
        return self._session(session_id)["consecutive_gaps"]

    def get_turn(self, session_id: str, turn_id: str):
        return self._session(session_id)["turns"].get(turn_id)

    def transcript(self, session_id: str) -> List[Dict[str, str]]:
        return [{"role": m["role"], "content": m["content"]} for m in self.get_history(session_id, 50)]

    # ------------------------------------------------------------------ stats

    def get_memory_stats(self) -> Dict[str, Any]:
        total_msgs = sum(len(s["messages"]) for s in self.sessions.values())
        questions = sum(1 for s in self.sessions.values() for m in s["messages"] if isinstance(m, HumanMessage))
        gaps = sum(1 for s in self.sessions.values() for t in s["turns"].values() if t.knowledge_gap)
        escalations = sum(1 for s in self.sessions.values() for t in s["turns"].values() if t.escalated)
        return {"sessions": len(self.sessions), "total_messages": total_msgs, "questions": questions,
                "knowledge_gaps": gaps, "escalations": escalations}
