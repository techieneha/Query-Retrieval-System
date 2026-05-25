"""
claim_agent/agents/graph.py
LangGraph conversation pipeline — builds and compiles the state machine.
"""
from __future__ import annotations
import uuid
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from claim_agent.agents.conversation_state import ConversationState, Message, SlotState
from claim_agent.agents.nodes import (
    intent_detector, query_handler, claim_initiator, slot_extractor,
    question_chooser, coverage_checker, claim_submitter,
    greeting_handler, cancel_handler, status_checker,
)


# ── Routers ──────────────────────────────────────────────────────
def route_intent(state: ConversationState) -> str:
    intent = state.get("intent","other")
    mode   = state.get("mode","chat")

    if mode == "claim_intake":
        if intent == "cancel":              return "cancel"
        if intent == "confirm" and state.get("coverage_result") is not None:
            return "submit"
        return "extract_slots"

    return {
        "query":    "query",
        "claim":    "initiate_claim",
        "status":   "status",
        "greeting": "greeting",
        "confirm":  "greeting",
        "cancel":   "cancel",
        "other":    "query",
    }.get(intent, "query")

def route_slots(state: ConversationState) -> str:
    return "ask_question" if state.get("missing_slots") else "check_coverage"


# ── Build ────────────────────────────────────────────────────────
def build_graph(use_memory: bool = True):
    b = StateGraph(ConversationState)

    b.add_node("detect_intent",   intent_detector)
    b.add_node("query_answer",    query_handler)
    b.add_node("initiate_claim",  claim_initiator)
    b.add_node("extract_slots",   slot_extractor)
    b.add_node("ask_question",    question_chooser)
    b.add_node("check_coverage",  coverage_checker)
    b.add_node("submit_claim",    claim_submitter)
    b.add_node("greet",           greeting_handler)
    b.add_node("cancel",          cancel_handler)
    b.add_node("check_status",    status_checker)

    b.set_entry_point("detect_intent")

    b.add_conditional_edges("detect_intent", route_intent, {
        "query":         "query_answer",
        "initiate_claim":"initiate_claim",
        "extract_slots": "extract_slots",
        "submit":        "submit_claim",
        "status":        "check_status",
        "greeting":      "greet",
        "cancel":        "cancel",
    })

    b.add_conditional_edges("extract_slots", route_slots, {
        "ask_question":  "ask_question",
        "check_coverage":"check_coverage",
    })

    for node in ["query_answer","initiate_claim","ask_question","check_coverage",
                 "submit_claim","greet","cancel","check_status"]:
        b.add_edge(node, END)

    return b.compile(checkpointer=MemorySaver() if use_memory else None)


# ── Singleton ─────────────────────────────────────────────────────
_graph = None
def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Public entry point ───────────────────────────────────────────
def process_message(session_id: str, user_message: str,
                    file_id: str, policy_number: str,
                    claimant_name: str,
                    existing_state: dict | None = None) -> dict:
    """Process one user turn. Returns updated state dict."""
    graph  = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    blank_slots = SlotState(claim_type=None, incident_date=None, incident_description=None,
                            claimed_amount=None, hospital_name=None, vehicle_number=None,
                            contact_number=None)

    if existing_state is None:
        state = ConversationState(
            session_id=session_id, file_id=file_id,
            policy_number=policy_number, claimant_name=claimant_name,
            messages=[Message(role="user", content=user_message)],
            intent=None, mode="chat",
            rag_answer=None, rag_sources=None, rag_confidence=None,
            slots=blank_slots, missing_slots=[], next_question=None,
            coverage_result=None, claim_id=None, claim_submitted=False,
            assistant_response=None, response_type="text", errors=[],
        )
    else:
        existing_state["messages"] = existing_state.get("messages",[]) + \
                                     [Message(role="user", content=user_message)]
        existing_state["assistant_response"] = None
        state = existing_state

    result = graph.invoke(state, config=config)

    # Append assistant reply to history
    reply = result.get("assistant_response") or "I'm not sure how to help with that."
    result["messages"] = result.get("messages",[]) + [Message(role="assistant", content=reply)]
    return result