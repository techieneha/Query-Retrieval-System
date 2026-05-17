"""
Claim Processing API Routes
Add these to your existing PolicyAI api/main.py:

    from api.claim_routes import router as claim_router
    app.include_router(claim_router)

These endpoints sit alongside your existing /api/v1/upload and /api/v1/query.
"""
import uuid
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse

from models.claim import (
    Claim, ClaimStatus, ClaimDocument, ClaimSubmitRequest,
    ClaimUpdateRequest
)
from agents.orchestrator import process_claim

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/claims", tags=["claims"])

# In-memory store for demo. Replace with PostgreSQL in production:
# from database import SessionLocal, ClaimORM
_claims_store: dict[str, Claim] = {}


# ── SUBMIT CLAIM ──────────────────────────────────────────────────────────────

@router.post("/submit", summary="Submit a new insurance claim")
async def submit_claim(
    background_tasks: BackgroundTasks,
    file_id: str = Form(..., description="PolicyAI file_id of the uploaded policy PDF"),
    claimant_name: str = Form(...),
    policy_number: str = Form(...),
    incident_description: str = Form(...),
    claimed_amount: float = Form(...),
    date_of_loss: str = Form(...),
    model_name: str = Form(default="mistral", description="Ollama model: mistral, llama3, phi3"),
    documents: list[UploadFile] = File(default=[], description="Supporting claim documents")
):
    """
    Submit a new claim against an existing policy (file_id from PolicyAI upload).
    Triggers the agentic processing pipeline in the background.
    """
    # Build ClaimDocument list from uploads
    claim_docs = []
    for upload in documents:
        content = await upload.read()
        doc = ClaimDocument(
            filename=upload.filename,
            doc_type=_detect_doc_type(upload.filename),
            extracted_text=content.decode("utf-8", errors="ignore")[:5000]
        )
        claim_docs.append(doc)

    # If no files, create a text document from the form description
    if not claim_docs:
        claim_docs.append(ClaimDocument(
            filename="claim_form.txt",
            doc_type="claim_form",
            extracted_text=(
                f"Claimant: {claimant_name}\n"
                f"Policy Number: {policy_number}\n"
                f"Date of Loss: {date_of_loss}\n"
                f"Claimed Amount: {claimed_amount}\n"
                f"Incident Description: {incident_description}"
            )
        ))

    claim = Claim(
        file_id=file_id,
        status=ClaimStatus.SUBMITTED,
        documents=claim_docs,
        processing_log=["claim submitted via API"]
    )

    _claims_store[claim.claim_id] = claim

    # Process asynchronously so API returns immediately
    background_tasks.add_task(_run_pipeline, claim.claim_id, model_name)

    return {
        "claim_id": claim.claim_id,
        "status": claim.status,
        "message": "Claim submitted. Processing started. Poll /status for updates.",
        "status_url": f"/api/v1/claims/{claim.claim_id}/status"
    }


async def _run_pipeline(claim_id: str, model_name: str):
    """Background task: run the full agent pipeline."""
    claim = _claims_store.get(claim_id)
    if not claim:
        return
    try:
        updated = await process_claim(claim, model_name=model_name)
        _claims_store[claim_id] = updated
    except Exception as e:
        logger.error(f"Pipeline error for {claim_id}: {e}")
        if claim_id in _claims_store:
            _claims_store[claim_id].status = ClaimStatus.PENDING_REVIEW
            _claims_store[claim_id].processing_log.append(f"pipeline_error: {str(e)}")


def _detect_doc_type(filename: str) -> str:
    ext = filename.lower().split(".")[-1] if "." in filename else "txt"
    return {
        "pdf": "pdf_document",
        "jpg": "photo", "jpeg": "photo", "png": "photo", "heic": "photo",
        "txt": "claim_form", "csv": "data_export",
        "doc": "word_document", "docx": "word_document"
    }.get(ext, "unknown")


# ── STATUS ────────────────────────────────────────────────────────────────────

@router.get("/{claim_id}/status", summary="Get claim processing status")
async def get_claim_status(claim_id: str):
    """Poll this endpoint after submission to get real-time status."""
    claim = _claims_store.get(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    response = {
        "claim_id": claim_id,
        "status": claim.status,
        "created_at": claim.created_at.isoformat(),
        "updated_at": claim.updated_at.isoformat(),
    }

    if claim.agent_decision:
        response["decision"] = {
            "action": claim.agent_decision.action,
            "confidence": claim.agent_decision.confidence,
            "reason": claim.agent_decision.reason,
            "recommended_payout": claim.agent_decision.recommended_payout
        }

    if claim.coverage_verdict:
        response["coverage"] = {
            "is_covered": claim.coverage_verdict.is_covered,
            "confidence": claim.coverage_verdict.confidence,
            "coverage_type": claim.coverage_verdict.coverage_type,
            "deductible": claim.coverage_verdict.deductible,
            "exclusions": claim.coverage_verdict.exclusions_triggered
        }

    if claim.fraud_signals:
        response["fraud_risk"] = {
            "level": claim.fraud_signals.risk_level,
            "score": claim.fraud_signals.risk_score,
            "flags": claim.fraud_signals.flags
        }

    return response


# ── ADJUSTER VIEW (full dossier) ──────────────────────────────────────────────

@router.get("/{claim_id}/dossier", summary="Full adjuster dossier with AI reasoning")
async def get_claim_dossier(claim_id: str):
    """Returns the complete claim with all AI reasoning, citations, and audit log."""
    claim = _claims_store.get(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    return {
        "claim": claim.model_dump(),
        "coverage_reasoning": claim.coverage_verdict.reasoning if claim.coverage_verdict else None,
        "policy_citations": claim.coverage_verdict.policy_citations if claim.coverage_verdict else [],
        "fraud_reasoning": claim.fraud_signals.reasoning if claim.fraud_signals else None,
        "processing_log": claim.processing_log
    }


# ── ADJUSTER OVERRIDE ─────────────────────────────────────────────────────────

@router.patch("/{claim_id}/review", summary="Adjuster override: approve or reject")
async def adjuster_review(claim_id: str, update: ClaimUpdateRequest):
    """
    Adjuster can override AI decision, add notes, and finalize claim.
    Every override is logged immutably in processing_log.
    """
    claim = _claims_store.get(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    if update.override_decision:
        old_status = claim.status
        claim.status = ClaimStatus.APPROVED if update.override_decision == "approve" else ClaimStatus.REJECTED
        claim.processing_log.append(
            f"adjuster_override: {old_status} → {claim.status} "
            f"| notes: {update.adjuster_notes or 'none'}"
        )

    if update.adjuster_notes:
        claim.adjuster_notes = update.adjuster_notes

    _claims_store[claim_id] = claim

    return {
        "claim_id": claim_id,
        "status": claim.status,
        "message": "Claim updated by adjuster"
    }


# ── QUEUE (all pending claims for adjuster dashboard) ─────────────────────────

@router.get("/queue/pending", summary="Adjuster queue: all claims needing review")
async def get_pending_claims():
    """Returns all claims in PENDING_REVIEW or NEEDS_INFO state."""
    pending = [
        {
            "claim_id": c.claim_id,
            "status": c.status,
            "claimant": c.extracted_data.claimant_name if c.extracted_data else "unknown",
            "amount": c.extracted_data.claimed_amount if c.extracted_data else None,
            "fraud_risk": c.fraud_signals.risk_level if c.fraud_signals else None,
            "coverage_confidence": c.coverage_verdict.confidence if c.coverage_verdict else None,
            "created_at": c.created_at.isoformat()
        }
        for c in _claims_store.values()
        if c.status in (ClaimStatus.PENDING_REVIEW, ClaimStatus.NEEDS_INFO)
    ]
    return {"total": len(pending), "claims": pending}