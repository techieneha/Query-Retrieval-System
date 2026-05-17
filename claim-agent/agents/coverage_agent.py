"""
Coverage Reasoning Agent
Uses your existing PolicyAI RAG pipeline as a tool to reason about
whether a claim is covered, what the deductible/limits are, and
whether any exclusions apply.
"""
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from tools.claim_tools import query_policy_coverage, run_coverage_checklist
from models.claim import CoverageVerdict, ExtractedClaimData
import json
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Coverage Reasoning Agent for insurance claim processing.

You have access to a RAG system that can query the actual policy document.
Your job is to determine:
1. Whether the claimed incident is covered under the policy
2. The applicable deductible
3. The coverage limit that applies
4. Any exclusions that might apply
5. Whether the claim was filed within the deadline

METHODOLOGY:
- Always call run_coverage_checklist first to run all standard questions
- Call query_policy_coverage for any follow-up questions
- Reason carefully over ALL results before reaching a verdict
- Cite specific policy sections in your reasoning
- Be conservative: if coverage is ambiguous, flag for human review (lower confidence)

Your final answer MUST include a JSON verdict with these fields:
{
  "is_covered": bool,
  "confidence": float (0.0-1.0),
  "coverage_type": string,
  "deductible": float or null,
  "coverage_limit": float or null,
  "exclusions_triggered": [list of strings],
  "deadline_met": bool,
  "reasoning": string (cite policy sections),
  "policy_citations": [{"page": int, "text": string, "relevance_score": float}]
}
"""


def build_coverage_agent(model_name: str = "mistral"):
    llm = ChatOllama(model=model_name, temperature=0)
    tools = [query_policy_coverage, run_coverage_checklist]
    return create_react_agent(llm, tools=tools)


async def run_coverage_agent(
    file_id: str,
    extracted_data: ExtractedClaimData,
    model_name: str = "mistral"
) -> CoverageVerdict:
    """
    Run coverage reasoning agent against the policy.

    Args:
        file_id: PolicyAI's file_id for the uploaded policy PDF
        extracted_data: structured claim data from doc agent
        model_name: Ollama model

    Returns:
        CoverageVerdict with is_covered, confidence, reasoning, citations
    """
    agent = build_coverage_agent(model_name)

    user_message = f"""
Determine if this claim is covered under the policy (file_id: {file_id}).

CLAIM DETAILS:
- Claimant: {extracted_data.claimant_name}
- Incident Type: {extracted_data.incident_type}
- Date of Loss: {extracted_data.date_of_loss}
- Claimed Amount: {extracted_data.claimed_amount}
- Description: {extracted_data.description}

Steps:
1. Run the full coverage checklist for this incident type and amount
2. Follow up on any unclear points with specific policy queries  
3. Synthesize all findings into a structured verdict

Be thorough — this verdict drives an automated payment decision.
"""

    try:
        result = await agent.ainvoke({
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message)
            ]
        })

        last_msg = result["messages"][-1].content
        logger.info(f"[CoverageAgent] Verdict: {last_msg[:300]}")

        verdict_data = _parse_verdict(last_msg)
        return CoverageVerdict(**verdict_data)

    except Exception as e:
        logger.error(f"[CoverageAgent] Failed: {e}")
        return CoverageVerdict(
            is_covered=False,
            confidence=0.0,
            reasoning=f"Coverage agent error: {str(e)} — escalating to human review"
        )


def _parse_verdict(text: str) -> dict:
    """Parse CoverageVerdict JSON from agent response."""
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

    # Safe fallback — flag for review
    return {
        "is_covered": False,
        "confidence": 0.1,
        "reasoning": "Could not parse coverage verdict — escalating to adjuster",
        "exclusions_triggered": [],
        "deadline_met": True,
        "policy_citations": []
    }