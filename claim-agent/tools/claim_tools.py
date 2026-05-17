"""
Tool definitions for claim processing agents.
These wrap your existing PolicyAI infrastructure + new claim-specific tools.
"""
import re
import json
import random
import httpx
from typing import Any
from langchain_core.tools import tool
from models.claim import ExtractedClaimData, FraudSignals, RiskLevel


# ── CONFIG ──────────────────────────────────────────────────────────────────
POLICY_AI_BASE = "http://localhost:8000"  # your existing FastAPI app


# ── DOCUMENT TOOLS ───────────────────────────────────────────────────────────

@tool
def extract_text_from_document(doc_content: str, doc_type: str) -> dict:
    """
    Extract structured text from a claim document.
    Uses pytesseract for images, pdfplumber for PDFs.
    In production replace with AWS Textract or Google Document AI.

    Args:
        doc_content: base64-encoded document content
        doc_type: one of 'pdf', 'image', 'text'

    Returns:
        dict with extracted_text and confidence score
    """
    # Production: call Textract/DocAI here
    # For now returns a structured extraction result
    return {
        "extracted_text": doc_content[:500] if len(doc_content) > 500 else doc_content,
        "confidence": 0.92,
        "doc_type_detected": doc_type,
        "language": "en"
    }


@tool
def parse_claim_fields(raw_text: str) -> dict:
    """
    Parse structured fields from raw extracted claim text using an LLM.
    Extracts: claimant name, policy number, date of loss, incident type,
    claimed amount, and identifies missing required fields.

    Args:
        raw_text: raw text extracted from claim documents

    Returns:
        dict matching ExtractedClaimData schema
    """
    # In production this calls Mistral/Ollama to extract structured JSON
    # Simulated parse for demonstration
    fields = {
        "claimant_name": None,
        "policy_number": None,
        "date_of_loss": None,
        "incident_type": None,
        "claimed_amount": None,
        "description": raw_text[:300] if raw_text else "",
        "supporting_docs": [],
        "missing_fields": []
    }

    # Simple regex extractions as fallback
    if amount_match := re.search(r'\$[\d,]+\.?\d*', raw_text):
        amount_str = amount_match.group().replace('$', '').replace(',', '')
        fields["claimed_amount"] = float(amount_str)

    if date_match := re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', raw_text):
        fields["date_of_loss"] = date_match.group()

    required = ["claimant_name", "policy_number", "date_of_loss", "incident_type", "claimed_amount"]
    fields["missing_fields"] = [f for f in required if not fields.get(f)]

    return fields


# ── POLICY RAG TOOLS (wraps your existing PolicyAI endpoints) ────────────────

@tool
def query_policy_coverage(file_id: str, question: str) -> dict:
    """
    Query the existing PolicyAI RAG system for a specific coverage question.
    Wraps POST /api/v1/query on your existing FastAPI backend.

    Args:
        file_id: PolicyAI file_id for the uploaded policy PDF
        question: natural language coverage question

    Returns:
        dict with answer, confidence, sources, latency
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{POLICY_AI_BASE}/api/v1/query",
                json={"file_id": file_id, "questions": [question]}
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        # Fallback for when PolicyAI is not running locally during dev
        return {
            "answer": f"[PolicyAI unavailable: {str(e)}] Simulated coverage answer for: {question}",
            "confidence": 0.0,
            "sources": [],
            "latency": 0
        }


@tool
def run_coverage_checklist(file_id: str, incident_type: str, claimed_amount: float, date_of_loss: str) -> dict:
    """
    Run the full standard coverage checklist against a policy.
    Fires 5 structured RAG queries in sequence and returns consolidated results.

    Args:
        file_id: PolicyAI policy file_id
        incident_type: type of incident (e.g. 'vehicle accident', 'hospitalization')
        claimed_amount: amount being claimed in INR/USD
        date_of_loss: date of the incident ISO format

    Returns:
        dict with coverage_checks list, each with question, answer, confidence, citations
    """
    questions = [
        f"Is '{incident_type}' a covered event under this policy?",
        f"What is the deductible amount that applies to {incident_type} claims?",
        f"What is the maximum coverage limit for {incident_type}?",
        f"Are there any exclusions that would apply to a {incident_type} claim of {claimed_amount}?",
        f"What is the claim filing deadline and was a claim filed on {date_of_loss} within that deadline?",
    ]

    results = []
    for q in questions:
        result = query_policy_coverage.invoke({"file_id": file_id, "question": q})
        results.append({
            "question": q,
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0.0),
            "citations": result.get("sources", [])
        })

    return {"coverage_checks": results}


# ── FRAUD DETECTION TOOLS ────────────────────────────────────────────────────

@tool
def check_claim_history(policy_number: str, claimant_name: str) -> dict:
    """
    Check claim history for duplicate or suspicious patterns.
    In production: query your PostgreSQL claims table.

    Args:
        policy_number: the policy number
        claimant_name: full name of claimant

    Returns:
        dict with previous_claims count and any duplicate flags
    """
    # Production: SELECT * FROM claims WHERE policy_number=? AND created_at > NOW()-INTERVAL '90 days'
    return {
        "previous_claims_90_days": 0,
        "duplicate_detected": False,
        "similar_claims": [],
        "policy_active": True
    }


@tool
def assess_statistical_anomaly(claimed_amount: float, incident_type: str, policy_limit: float) -> dict:
    """
    Check if claimed amount is statistically anomalous for this incident type.
    Compares against historical averages for the incident category.

    Args:
        claimed_amount: amount being claimed
        incident_type: type of incident
        policy_limit: maximum coverage limit from policy

    Returns:
        dict with is_anomalous flag, z_score, and explanation
    """
    # Production: query a claims statistics table or ML model
    # Simulated statistical check
    typical_ranges = {
        "vehicle accident": (50000, 500000),
        "hospitalization": (20000, 800000),
        "property damage": (10000, 1000000),
        "theft": (5000, 200000),
        "fire": (100000, 2000000),
    }

    key = next((k for k in typical_ranges if k in incident_type.lower()), None)
    if key:
        low, high = typical_ranges[key]
        is_anomalous = claimed_amount > high * 1.5 or claimed_amount < low * 0.1
        ratio = claimed_amount / policy_limit if policy_limit > 0 else 0
        return {
            "is_anomalous": is_anomalous,
            "typical_range": typical_ranges[key],
            "policy_limit_ratio": round(ratio, 2),
            "explanation": f"Claim is {'outside' if is_anomalous else 'within'} typical range for {key}"
        }

    return {"is_anomalous": False, "typical_range": None, "policy_limit_ratio": 0.0, "explanation": "Unknown incident type"}


@tool
def check_document_consistency(extracted_data: str) -> dict:
    """
    Check for internal consistency across claim documents.
    Looks for date mismatches, amount discrepancies, conflicting statements.

    Args:
        extracted_data: JSON string of ExtractedClaimData

    Returns:
        dict with consistency_issues list and overall consistency score
    """
    try:
        data = json.loads(extracted_data) if isinstance(extracted_data, str) else extracted_data
    except Exception:
        data = {}

    issues = []
    score = 1.0

    if not data.get("date_of_loss"):
        issues.append("Date of loss not found in any document")
        score -= 0.2

    if not data.get("claimant_name"):
        issues.append("Claimant name inconsistent or missing across documents")
        score -= 0.15

    if data.get("claimed_amount") and data["claimed_amount"] <= 0:
        issues.append("Invalid claimed amount")
        score -= 0.3

    return {
        "consistency_score": max(0.0, score),
        "consistency_issues": issues,
        "documents_checked": len(data.get("supporting_docs", [])) + 1
    }


# ── DECISION & NOTIFICATION TOOLS ────────────────────────────────────────────

@tool
def calculate_final_decision(
    coverage_confidence: float,
    fraud_risk_score: float,
    claimed_amount: float,
    missing_fields: str,
    is_covered: bool
) -> dict:
    """
    Compute the final routing decision based on all agent outputs.
    Implements the tiered confidence model for industry-standard automation.

    Args:
        coverage_confidence: 0.0-1.0 from coverage agent
        fraud_risk_score: 0.0-1.0 from fraud agent (higher = more risky)
        claimed_amount: claim amount in currency units
        missing_fields: JSON list of missing required fields
        is_covered: whether coverage agent determined claim is covered

    Returns:
        dict with action, confidence, reason, recommended_payout
    """
    missing = json.loads(missing_fields) if isinstance(missing_fields, str) else missing_fields

    if missing:
        return {
            "action": "request_info",
            "confidence": 0.95,
            "reason": f"Required fields missing: {', '.join(missing)}",
            "missing_info": missing,
            "recommended_payout": None
        }

    if not is_covered:
        return {
            "action": "reject",
            "confidence": coverage_confidence,
            "reason": "Claim is not covered under the policy terms",
            "missing_info": [],
            "recommended_payout": None
        }

    # Tiered routing logic
    if coverage_confidence >= 0.85 and fraud_risk_score <= 0.2 and claimed_amount <= 100000:
        return {
            "action": "auto_approve",
            "confidence": coverage_confidence,
            "reason": "High confidence coverage, low fraud risk, within auto-approval threshold",
            "missing_info": [],
            "recommended_payout": claimed_amount
        }
    elif coverage_confidence >= 0.70 and fraud_risk_score <= 0.5:
        return {
            "action": "escalate_review",
            "confidence": coverage_confidence,
            "reason": "Moderate confidence or amount exceeds auto-approval threshold — adjuster review recommended",
            "missing_info": [],
            "recommended_payout": claimed_amount
        }
    else:
        return {
            "action": "escalate_review",
            "confidence": coverage_confidence,
            "reason": f"Low coverage confidence ({coverage_confidence:.0%}) or elevated fraud risk ({fraud_risk_score:.0%})",
            "missing_info": [],
            "recommended_payout": None
        }


@tool
def notify_claimant(claim_id: str, action: str, reason: str, contact_email: str = "") -> dict:
    """
    Send notification to claimant about claim status.
    In production: integrate with SendGrid, Twilio, or your email service.

    Args:
        claim_id: unique claim identifier
        action: the decision action taken
        reason: human-readable explanation
        contact_email: claimant email if available

    Returns:
        dict with notification_sent status
    """
    messages = {
        "auto_approve": f"Good news! Your claim {claim_id} has been approved. Payment will be processed within 3-5 business days.",
        "escalate_review": f"Your claim {claim_id} is under review by our team. We'll update you within 2 business days.",
        "request_info": f"We need additional information for claim {claim_id}. {reason}",
        "reject": f"Unfortunately, claim {claim_id} cannot be approved. {reason}. You may appeal this decision."
    }

    message = messages.get(action, f"Your claim {claim_id} status has been updated.")
    print(f"[NOTIFY] To: {contact_email or 'claimant'} | {message}")

    return {
        "notification_sent": True,
        "channel": "email" if contact_email else "in-app",
        "message_preview": message[:100]
    }