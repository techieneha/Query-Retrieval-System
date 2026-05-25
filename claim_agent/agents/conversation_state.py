"""
claim_agent/agents/conversation_state.py
LangGraph TypedDict — single object that flows through every graph node.
"""
from __future__ import annotations
from typing import Annotated, Optional, Any
from typing_extensions import TypedDict
import operator


class Message(TypedDict):
    role: str       # "user" | "assistant"
    content: str


class SlotState(TypedDict):
    claim_type:            Optional[str]    # health|vehicle|home|travel|other
    incident_date:         Optional[str]
    incident_description:  Optional[str]
    claimed_amount:        Optional[float]
    hospital_name:         Optional[str]
    vehicle_number:        Optional[str]
    contact_number:        Optional[str]


# Required before submission
REQUIRED_SLOTS   = ["claim_type","incident_date","incident_description","claimed_amount"]

# Per-slot follow-up questions
SLOT_QUESTIONS = {
    "claim_type":           "What type of claim is this?\n→ **health · vehicle · home · travel · other**",
    "incident_date":        "When did the incident happen? *(e.g. 15 March 2024)*",
    "incident_description": "Can you describe what happened in a few sentences?",
    "claimed_amount":       "What is the approximate amount you'd like to claim *(in ₹)*?",
    "hospital_name":        "Which hospital or clinic did you visit?",
    "vehicle_number":       "What is your vehicle registration number?",
    "contact_number":       "What's the best phone number to reach you on?",
}


class ConversationState(TypedDict):
    # ── session ──────────────────────────────────────────────────
    session_id:    str
    file_id:       str
    policy_number: str
    claimant_name: str

    # ── history (append-only) ─────────────────────────────────────
    messages: Annotated[list[Message], operator.add]

    # ── routing ──────────────────────────────────────────────────
    intent: Optional[str]   # query|claim|status|greeting|confirm|cancel|other
    mode:   str             # chat | claim_intake

    # ── query mode ───────────────────────────────────────────────
    rag_answer:     Optional[str]
    rag_sources:    Optional[list[str]]
    rag_confidence: Optional[float]

    # ── claim intake ──────────────────────────────────────────────
    slots:         SlotState
    missing_slots: list[str]
    next_question: Optional[str]

    # ── coverage ─────────────────────────────────────────────────
    coverage_result: Optional[dict]

    # ── submission ───────────────────────────────────────────────
    claim_id:        Optional[str]
    claim_submitted: bool

    # ── response ─────────────────────────────────────────────────
    assistant_response: Optional[str]
    response_type:      str   # text|coverage_card|claim_summary|error

    errors: Annotated[list[str], operator.add]