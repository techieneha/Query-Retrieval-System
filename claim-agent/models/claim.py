from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum
import uuid


class ClaimStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    NEEDS_INFO = "needs_info"
    AUTO_APPROVED = "auto_approved"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClaimDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    doc_type: str  # "claim_form", "medical_bill", "photo", "police_report", etc.
    extracted_text: str = ""
    ocr_confidence: float = 0.0


class ExtractedClaimData(BaseModel):
    claimant_name: Optional[str] = None
    policy_number: Optional[str] = None
    date_of_loss: Optional[str] = None
    incident_type: Optional[str] = None
    claimed_amount: Optional[float] = None
    description: Optional[str] = None
    supporting_docs: List[str] = []
    missing_fields: List[str] = []


class CoverageVerdict(BaseModel):
    is_covered: bool
    confidence: float  # 0.0 - 1.0
    coverage_type: Optional[str] = None
    deductible: Optional[float] = None
    coverage_limit: Optional[float] = None
    exclusions_triggered: List[str] = []
    deadline_met: bool = True
    reasoning: str = ""
    policy_citations: List[dict] = []  # [{page, text, relevance_score}]


class FraudSignals(BaseModel):
    risk_level: RiskLevel
    risk_score: float  # 0.0 - 1.0
    flags: List[str] = []
    reasoning: str = ""


class AgentDecision(BaseModel):
    action: Literal["auto_approve", "escalate_review", "request_info", "reject"]
    confidence: float
    reason: str
    missing_info: List[str] = []
    recommended_payout: Optional[float] = None


class Claim(BaseModel):
    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_id: str  # PolicyAI's existing file_id for the policy document
    status: ClaimStatus = ClaimStatus.DRAFT
    documents: List[ClaimDocument] = []
    extracted_data: Optional[ExtractedClaimData] = None
    coverage_verdict: Optional[CoverageVerdict] = None
    fraud_signals: Optional[FraudSignals] = None
    agent_decision: Optional[AgentDecision] = None
    adjuster_notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    processing_log: List[str] = []  # agent trace for audit


class ClaimSubmitRequest(BaseModel):
    file_id: str  # existing PolicyAI policy file_id
    claimant_name: str
    policy_number: str
    incident_description: str
    claimed_amount: float
    date_of_loss: str  # ISO format


class ClaimUpdateRequest(BaseModel):
    adjuster_notes: Optional[str] = None
    status: Optional[ClaimStatus] = None
    override_decision: Optional[Literal["approve", "reject"]] = None