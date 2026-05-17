"""
End-to-end tests for the agentic claim processing pipeline.
Run: pytest tests/test_pipeline.py -v --asyncio-mode=auto
"""
import pytest
import json
from unittest.mock import AsyncMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.claim import (
    Claim, ClaimDocument, ClaimStatus, ExtractedClaimData,
    CoverageVerdict, FraudSignals, RiskLevel
)
from tools.claim_tools import calculate_final_decision, check_document_consistency


# ── Helper: fresh claim per test (avoids state bleed between tests) ───────────

def make_claim():
    return Claim(
        file_id="test-policy-file-id",
        status=ClaimStatus.SUBMITTED,
        documents=[
            ClaimDocument(
                filename="claim.txt",
                doc_type="claim_form",
                extracted_text=(
                    "Claimant: Neha Sharma\n"
                    "Policy Number: POL-2024-0042\n"
                    "Date of Loss: 2024-03-15\n"
                    "Incident Type: vehicle accident\n"
                    "Claimed Amount: $45,000\n"
                    "Description: Rear-end collision on highway."
                )
            )
        ]
    )


# ── Tool unit tests ───────────────────────────────────────────────────────────

class TestDecisionTool:
    def test_auto_approve_high_confidence_low_risk(self):
        result = calculate_final_decision.invoke({
            "coverage_confidence": 0.92,
            "fraud_risk_score": 0.1,
            "claimed_amount": 50000,
            "missing_fields": "[]",
            "is_covered": True
        })
        assert result["action"] == "auto_approve"
        assert result["recommended_payout"] == 50000

    def test_escalate_high_amount(self):
        result = calculate_final_decision.invoke({
            "coverage_confidence": 0.90,
            "fraud_risk_score": 0.1,
            "claimed_amount": 500000,
            "missing_fields": "[]",
            "is_covered": True
        })
        assert result["action"] == "escalate_review"

    def test_escalate_medium_fraud_risk(self):
        result = calculate_final_decision.invoke({
            "coverage_confidence": 0.88,
            "fraud_risk_score": 0.45,
            "claimed_amount": 30000,
            "missing_fields": "[]",
            "is_covered": True
        })
        assert result["action"] == "escalate_review"

    def test_request_info_missing_fields(self):
        result = calculate_final_decision.invoke({
            "coverage_confidence": 0.9,
            "fraud_risk_score": 0.1,
            "claimed_amount": 20000,
            "missing_fields": '["policy_number", "incident_type"]',
            "is_covered": True
        })
        assert result["action"] == "request_info"
        assert "policy_number" in result["missing_info"]

    def test_reject_not_covered(self):
        result = calculate_final_decision.invoke({
            "coverage_confidence": 0.95,
            "fraud_risk_score": 0.05,
            "claimed_amount": 10000,
            "missing_fields": "[]",
            "is_covered": False
        })
        assert result["action"] == "reject"


class TestConsistencyTool:
    def test_flags_missing_date(self):
        data = json.dumps({
            "claimant_name": "Test User",
            "date_of_loss": None,
            "claimed_amount": 50000
        })
        result = check_document_consistency.invoke({"extracted_data": data})
        assert result["consistency_score"] < 1.0
        assert any("date" in issue.lower() for issue in result["consistency_issues"])

    def test_perfect_consistency(self):
        data = json.dumps({
            "claimant_name": "Test User",
            "date_of_loss": "2024-01-15",
            "claimed_amount": 50000,
            "supporting_docs": ["bill.pdf"]
        })
        result = check_document_consistency.invoke({"extracted_data": data})
        assert result["consistency_score"] == 1.0
        assert len(result["consistency_issues"]) == 0


# ── Integration tests: full pipeline with mocked agents ──────────────────────
# Patch target = module attribute used by orchestrator's _doc_agent_mod etc.

class TestPipeline:

    @pytest.mark.asyncio
    async def test_pipeline_auto_approve(self):
        """Clean low-risk claim should be auto-approved."""
        mock_extracted = ExtractedClaimData(
            claimant_name="Neha Sharma", policy_number="POL-2024-0042",
            date_of_loss="2024-03-15", incident_type="vehicle accident",
            claimed_amount=45000, description="Rear-end collision", missing_fields=[]
        )
        mock_coverage = CoverageVerdict(
            is_covered=True, confidence=0.91, coverage_type="vehicle",
            deductible=5000, coverage_limit=500000, exclusions_triggered=[],
            deadline_met=True, reasoning="Covered under Section 4.2",
            policy_citations=[{"page": 4, "text": "Vehicle accidents...", "relevance_score": 0.95}]
        )
        mock_fraud = FraudSignals(
            risk_level=RiskLevel.LOW, risk_score=0.08, flags=[],
            reasoning="No anomalies detected."
        )

        with patch("agents.doc_agent.run_doc_agent", new=AsyncMock(return_value=mock_extracted)), \
             patch("agents.coverage_agent.run_coverage_agent", new=AsyncMock(return_value=mock_coverage)), \
             patch("agents.fraud_agent.run_fraud_agent", new=AsyncMock(return_value=mock_fraud)):
            from agents.orchestrator import process_claim
            result = await process_claim(make_claim(), model_name="mistral")

        assert result.status == ClaimStatus.AUTO_APPROVED
        assert result.agent_decision.action == "auto_approve"
        assert result.agent_decision.recommended_payout == 45000
        assert len(result.processing_log) > 0

    @pytest.mark.asyncio
    async def test_pipeline_escalates_high_fraud(self):
        """High fraud risk escalates even with good coverage."""
        mock_extracted = ExtractedClaimData(
            claimant_name="Test User", policy_number="POL-2024-0001",
            date_of_loss="2024-03-15", incident_type="vehicle accident",
            claimed_amount=45000, missing_fields=[]
        )
        mock_coverage = CoverageVerdict(
            is_covered=True, confidence=0.93, reasoning="Covered",
            exclusions_triggered=[], deadline_met=True, policy_citations=[]
        )
        mock_fraud = FraudSignals(
            risk_level=RiskLevel.HIGH, risk_score=0.78,
            flags=["amount_anomaly", "third_claim_this_month"],
            reasoning="3 claims in 30 days"
        )

        with patch("agents.doc_agent.run_doc_agent", new=AsyncMock(return_value=mock_extracted)), \
             patch("agents.coverage_agent.run_coverage_agent", new=AsyncMock(return_value=mock_coverage)), \
             patch("agents.fraud_agent.run_fraud_agent", new=AsyncMock(return_value=mock_fraud)):
            from agents.orchestrator import process_claim
            result = await process_claim(make_claim(), model_name="mistral")

        assert result.status == ClaimStatus.PENDING_REVIEW
        assert result.agent_decision.action == "escalate_review"

    @pytest.mark.asyncio
    async def test_pipeline_requests_missing_info(self):
        """Missing required fields triggers request_info without running analysis."""
        mock_extracted = ExtractedClaimData(
            claimant_name=None, policy_number=None, date_of_loss=None,
            incident_type=None, claimed_amount=None,
            missing_fields=["policy_number", "incident_type", "claimed_amount"]
        )

        with patch("agents.doc_agent.run_doc_agent", new=AsyncMock(return_value=mock_extracted)):
            from agents.orchestrator import process_claim
            result = await process_claim(make_claim(), model_name="mistral")

        assert result.status == ClaimStatus.NEEDS_INFO
        assert result.agent_decision.action == "request_info"
        assert len(result.agent_decision.missing_info) > 0

    @pytest.mark.asyncio
    async def test_pipeline_rejects_uncovered_claim(self):
        """Claim outside policy coverage is rejected."""
        mock_extracted = ExtractedClaimData(
            claimant_name="Test User", policy_number="POL-001",
            date_of_loss="2024-03-15", incident_type="flood damage",
            claimed_amount=80000, missing_fields=[]
        )
        mock_coverage = CoverageVerdict(
            is_covered=False, confidence=0.97,
            reasoning="Flood damage excluded under Section 8.1",
            exclusions_triggered=["flood_exclusion"], deadline_met=True, policy_citations=[]
        )
        mock_fraud = FraudSignals(
            risk_level=RiskLevel.LOW, risk_score=0.05, flags=[], reasoning="No issues."
        )

        with patch("agents.doc_agent.run_doc_agent", new=AsyncMock(return_value=mock_extracted)), \
             patch("agents.coverage_agent.run_coverage_agent", new=AsyncMock(return_value=mock_coverage)), \
             patch("agents.fraud_agent.run_fraud_agent", new=AsyncMock(return_value=mock_fraud)):
            from agents.orchestrator import process_claim
            result = await process_claim(make_claim(), model_name="mistral")

        assert result.status == ClaimStatus.REJECTED
        assert result.agent_decision.action == "reject"


# ── API endpoint tests ────────────────────────────────────────────────────────

class TestClaimAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.combined_app import app
        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_submit_claim_returns_claim_id(self, client):
        resp = client.post("/api/v1/claims/submit", data={
            "file_id": "test-file-id",
            "claimant_name": "Neha Sharma",
            "policy_number": "POL-001",
            "incident_description": "Vehicle accident on highway",
            "claimed_amount": 50000,
            "date_of_loss": "2024-03-15"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "claim_id" in data
        assert "status_url" in data

    def test_get_pending_queue(self, client):
        resp = client.get("/api/v1/claims/queue/pending")
        assert resp.status_code == 200
        assert "claims" in resp.json()

    def test_submit_then_status(self, client):
        """Submit a claim then verify the status endpoint exists."""
        submit_resp = client.post("/api/v1/claims/submit", data={
            "file_id": "test-file-id",
            "claimant_name": "Neha Sharma",
            "policy_number": "POL-002",
            "incident_description": "Medical expenses",
            "claimed_amount": 20000,
            "date_of_loss": "2024-04-01"
        })
        claim_id = submit_resp.json()["claim_id"]
        status_resp = client.get(f"/api/v1/claims/{claim_id}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["claim_id"] == claim_id

    def test_unknown_claim_returns_404(self, client):
        resp = client.get("/api/v1/claims/nonexistent-id/status")
        assert resp.status_code == 404