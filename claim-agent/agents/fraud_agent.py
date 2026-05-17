"""
Fraud Detection Agent
Runs parallel to coverage reasoning. Checks for statistical anomalies,
duplicate claims, and internal document inconsistencies.
"""
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from tools.claim_tools import check_claim_history, assess_statistical_anomaly, check_document_consistency
from models.claim import FraudSignals, RiskLevel, ExtractedClaimData, CoverageVerdict
import json
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Fraud Detection Agent for insurance claim processing.

Your job is to assess the fraud risk of a claim using three tools:
1. check_claim_history — look for recent duplicate claims or suspicious patterns
2. assess_statistical_anomaly — check if the amount is unusual for this incident type
3. check_document_consistency — look for internal inconsistencies across documents

RISK SCORING GUIDE:
- LOW (0.0-0.3): No flags, amounts in normal range, consistent documents, no claim history issues
- MEDIUM (0.3-0.6): Minor flags, some anomalies, or missing documents
- HIGH (0.6-1.0): Duplicate detected, large anomaly, multiple inconsistencies

Your final answer MUST include JSON:
{
  "risk_level": "low" | "medium" | "high",
  "risk_score": float (0.0-1.0),
  "flags": [list of specific issues found],
  "reasoning": string
}

Be specific about flags. Do not label claims high-risk without clear evidence.
"""


def build_fraud_agent(model_name: str = "mistral"):
    llm = ChatOllama(model=model_name, temperature=0)
    tools = [check_claim_history, assess_statistical_anomaly, check_document_consistency]
    return create_react_agent(llm, tools=tools)


async def run_fraud_agent(
    extracted_data: ExtractedClaimData,
    coverage_verdict: CoverageVerdict,
    model_name: str = "mistral"
) -> FraudSignals:
    """
    Run fraud detection agent.

    Args:
        extracted_data: structured claim data
        coverage_verdict: result from coverage agent (for limit comparison)
        model_name: Ollama model

    Returns:
        FraudSignals with risk_level, risk_score, flags, reasoning
    """
    agent = build_fraud_agent(model_name)

    policy_limit = coverage_verdict.coverage_limit or 0
    extracted_json = extracted_data.model_dump_json()

    user_message = f"""
Assess fraud risk for this claim:

CLAIM DETAILS:
- Claimant: {extracted_data.claimant_name}
- Policy Number: {extracted_data.policy_number}
- Incident Type: {extracted_data.incident_type}
- Claimed Amount: {extracted_data.claimed_amount}
- Date of Loss: {extracted_data.date_of_loss}
- Policy Coverage Limit: {policy_limit}

FULL EXTRACTED DATA:
{extracted_json}

Steps:
1. Check claim history for this policy/claimant
2. Assess if amount is statistically anomalous for '{extracted_data.incident_type}' with limit {policy_limit}
3. Check document consistency
4. Synthesize a final risk verdict

Be objective — false positives hurt legitimate claimants.
"""

    try:
        result = await agent.ainvoke({
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message)
            ]
        })

        last_msg = result["messages"][-1].content
        logger.info(f"[FraudAgent] Risk: {last_msg[:200]}")

        fraud_data = _parse_fraud_result(last_msg)
        return FraudSignals(**fraud_data)

    except Exception as e:
        logger.error(f"[FraudAgent] Failed: {e}")
        return FraudSignals(
            risk_level=RiskLevel.MEDIUM,
            risk_score=0.5,
            flags=["fraud_agent_error"],
            reasoning=f"Fraud agent error: {str(e)} — defaulting to medium risk"
        )


def _parse_fraud_result(text: str) -> dict:
    try:
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return json.loads(text[start:end].strip())
        elif "{" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
    except Exception:
        pass

    return {
        "risk_level": "medium",
        "risk_score": 0.5,
        "flags": ["parse_error"],
        "reasoning": "Could not parse fraud assessment — defaulting to medium risk"
    }