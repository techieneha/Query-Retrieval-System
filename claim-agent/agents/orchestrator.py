"""
Claim Processing Orchestrator — LangGraph State Machine
Coordinates the full agentic claim processing pipeline:
  DocAgent → CoverageAgent + FraudAgent (parallel) → Decision → Notify

State flows through typed nodes. Each node updates the shared ClaimState.
The loop re-runs if more info is needed (max 3 iterations).
"""
import asyncio
import logging
from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from models.claim import (
    Claim, ClaimStatus, ClaimDocument, ExtractedClaimData,
    CoverageVerdict, FraudSignals, AgentDecision, RiskLevel
)
import agents.doc_agent as _doc_agent_mod
import agents.coverage_agent as _coverage_agent_mod
import agents.fraud_agent as _fraud_agent_mod
from tools.claim_tools import calculate_final_decision, notify_claimant
import json

logger = logging.getLogger(__name__)


# ── SHARED STATE ─────────────────────────────────────────────────────────────

class ClaimState(TypedDict):
    claim: Claim
    iteration: int                        # loop counter (max 3)
    messages: Annotated[list[BaseMessage], add_messages]
    model_name: str


# ── NODE FUNCTIONS ────────────────────────────────────────────────────────────

async def doc_extraction_node(state: ClaimState) -> ClaimState:
    """Run document intelligence agent to extract structured claim data."""
    claim = state["claim"]
    logger.info(f"[Orchestrator] doc_extraction | claim={claim.claim_id}")

    claim.processing_log.append("doc_extraction_node: starting")
    claim.status = ClaimStatus.PROCESSING

    extracted = await _doc_agent_mod.run_doc_agent(claim.documents, model_name=state["model_name"])
    claim.extracted_data = extracted

    if extracted.missing_fields and "agent_error" not in extracted.missing_fields:
        claim.processing_log.append(
            f"doc_extraction_node: missing fields = {extracted.missing_fields}"
        )
    else:
        claim.processing_log.append("doc_extraction_node: complete")

    return {**state, "claim": claim}


async def parallel_analysis_node(state: ClaimState) -> ClaimState:
    """Run coverage reasoning and fraud detection in parallel."""
    claim = state["claim"]
    logger.info(f"[Orchestrator] parallel_analysis | claim={claim.claim_id}")
    claim.processing_log.append("parallel_analysis_node: starting coverage + fraud in parallel")

    extracted = claim.extracted_data or ExtractedClaimData()

    # Run both agents concurrently
    coverage_task = _coverage_agent_mod.run_coverage_agent(claim.file_id, extracted, model_name=state["model_name"])
    fraud_task = _fraud_agent_mod.run_fraud_agent(
        extracted,
        claim.coverage_verdict or CoverageVerdict(is_covered=False, confidence=0.0, reasoning=""),
        model_name=state["model_name"]
    )

    coverage_verdict, fraud_signals = await asyncio.gather(coverage_task, fraud_task)

    claim.coverage_verdict = coverage_verdict
    claim.fraud_signals = fraud_signals
    claim.processing_log.append(
        f"parallel_analysis_node: coverage={coverage_verdict.is_covered} "
        f"conf={coverage_verdict.confidence:.2f} fraud_risk={fraud_signals.risk_level}"
    )

    return {**state, "claim": claim}


async def decision_node(state: ClaimState) -> ClaimState:
    """Compute final routing decision using all agent outputs."""
    claim = state["claim"]
    logger.info(f"[Orchestrator] decision_node | claim={claim.claim_id}")

    extracted = claim.extracted_data or ExtractedClaimData()
    coverage = claim.coverage_verdict or CoverageVerdict(is_covered=False, confidence=0.0, reasoning="")
    fraud = claim.fraud_signals or FraudSignals(risk_level=RiskLevel.MEDIUM, risk_score=0.5, flags=[], reasoning="")

    decision_result = calculate_final_decision.invoke({
        "coverage_confidence": coverage.confidence,
        "fraud_risk_score": fraud.risk_score,
        "claimed_amount": extracted.claimed_amount or 0,
        "missing_fields": json.dumps(extracted.missing_fields),
        "is_covered": coverage.is_covered
    })

    claim.agent_decision = AgentDecision(**decision_result)

    # Map action to claim status
    status_map = {
        "auto_approve": ClaimStatus.AUTO_APPROVED,
        "escalate_review": ClaimStatus.PENDING_REVIEW,
        "request_info": ClaimStatus.NEEDS_INFO,
        "reject": ClaimStatus.REJECTED
    }
    claim.status = status_map.get(decision_result["action"], ClaimStatus.PENDING_REVIEW)
    claim.processing_log.append(
        f"decision_node: action={decision_result['action']} confidence={decision_result['confidence']:.2f}"
    )

    return {**state, "claim": claim}


async def notification_node(state: ClaimState) -> ClaimState:
    """Send claimant notification and finalize claim."""
    claim = state["claim"]
    decision = claim.agent_decision

    if decision:
        notify_claimant.invoke({
            "claim_id": claim.claim_id,
            "action": decision.action,
            "reason": decision.reason,
            "contact_email": ""  # add claimant email from extracted_data in production
        })
        claim.processing_log.append(f"notification_node: notified claimant | action={decision.action}")

    return {**state, "claim": claim}


# ── ROUTING LOGIC ─────────────────────────────────────────────────────────────

def route_after_extraction(state: ClaimState) -> str:
    """After doc extraction: if too many missing fields and iterations remain, loop back."""
    claim = state["claim"]
    extracted = claim.extracted_data

    if not extracted:
        return "request_info_end"

    critical_missing = [f for f in (extracted.missing_fields or [])
                        if f in ("policy_number", "incident_type", "claimed_amount")]

    if critical_missing and state["iteration"] < 3:
        return "request_info_end"

    return "parallel_analysis"


def route_after_decision(state: ClaimState) -> str:
    """After decision: route to notify (all paths end at notify)."""
    return "notify"


# ── GRAPH ASSEMBLY ────────────────────────────────────────────────────────────

def build_claim_graph() -> StateGraph:
    graph = StateGraph(ClaimState)

    graph.add_node("doc_extraction", doc_extraction_node)
    graph.add_node("parallel_analysis", parallel_analysis_node)
    graph.add_node("decision", decision_node)
    graph.add_node("notify", notification_node)

    graph.set_entry_point("doc_extraction")

    graph.add_conditional_edges(
        "doc_extraction",
        route_after_extraction,
        {
            "parallel_analysis": "parallel_analysis",
            "request_info_end": "decision"   # skip analysis, go straight to request_info decision
        }
    )

    graph.add_edge("parallel_analysis", "decision")
    graph.add_edge("decision", "notify")
    graph.add_edge("notify", END)

    return graph.compile()


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────

async def process_claim(claim: Claim, model_name: str = "mistral") -> Claim:
    """
    Main entry point. Run the full agentic claim processing pipeline.

    Args:
        claim: Claim object with file_id and documents populated
        model_name: Ollama model to use (mistral, llama3, phi3, etc.)

    Returns:
        Updated Claim with status, verdict, fraud signals, and decision
    """
    graph = build_claim_graph()

    initial_state: ClaimState = {
        "claim": claim,
        "iteration": 0,
        "messages": [],
        "model_name": model_name
    }

    logger.info(f"[Orchestrator] Starting pipeline | claim={claim.claim_id} model={model_name}")

    final_state = await graph.ainvoke(initial_state)
    result_claim = final_state["claim"]

    logger.info(
        f"[Orchestrator] Complete | claim={result_claim.claim_id} "
        f"status={result_claim.status} decision={result_claim.agent_decision}"
    )

    return result_claim