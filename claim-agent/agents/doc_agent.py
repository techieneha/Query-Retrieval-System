"""
Document Intelligence Agent
Handles intake of claim documents, runs OCR, and extracts structured data.
Integrates with your existing PolicyAI file upload infrastructure.
"""
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from tools.claim_tools import extract_text_from_document, parse_claim_fields, check_document_consistency
from models.claim import ExtractedClaimData, ClaimDocument
import json
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Document Intelligence Agent for insurance claim processing.

Your job is to:
1. Extract text from uploaded claim documents (forms, bills, photos, reports)
2. Parse structured fields: claimant name, policy number, date of loss, incident type, claimed amount
3. Identify any missing required fields
4. Check internal consistency across documents

ALWAYS call extract_text_from_document first, then parse_claim_fields, then check_document_consistency.
Return a JSON summary of your findings with all extracted fields and any issues found.
Be precise — insurance decisions depend on accuracy.
"""


def build_doc_agent(model_name: str = "mistral"):
    """
    Build the document intelligence agent using Ollama (local, open source).
    Defaults to Mistral — change to 'llama3', 'phi3', etc. as needed.
    """
    llm = ChatOllama(model=model_name, temperature=0)
    tools = [extract_text_from_document, parse_claim_fields, check_document_consistency]
    return create_react_agent(llm, tools=tools)


async def run_doc_agent(
    documents: list[ClaimDocument],
    model_name: str = "mistral"
) -> ExtractedClaimData:
    """
    Run document intelligence agent on a list of claim documents.

    Args:
        documents: list of ClaimDocument objects (already uploaded)
        model_name: Ollama model to use

    Returns:
        ExtractedClaimData with all parsed fields and missing field flags
    """
    agent = build_doc_agent(model_name)

    # Build context from documents
    doc_summary = "\n\n".join([
        f"Document: {doc.filename} (type: {doc.doc_type})\n"
        f"Content: {doc.extracted_text[:800] if doc.extracted_text else '[binary - needs OCR]'}"
        for doc in documents
    ])

    user_message = f"""
Process these claim documents and extract all required fields:

{doc_summary}

Use your tools to:
1. Extract text from each document
2. Parse structured claim fields
3. Check consistency across documents
4. Report any missing required fields

Return a comprehensive extraction result.
"""

    try:
        result = await agent.ainvoke({
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message)
            ]
        })

        # Parse final agent message
        last_msg = result["messages"][-1].content
        logger.info(f"[DocAgent] Raw output: {last_msg[:300]}")

        # Try to extract JSON from agent output
        extracted = _parse_agent_json(last_msg)
        return ExtractedClaimData(**extracted)

    except Exception as e:
        logger.error(f"[DocAgent] Failed: {e}")
        # Return partial data with error flag
        return ExtractedClaimData(
            missing_fields=["agent_error"],
            description=f"Document agent error: {str(e)}"
        )


def _parse_agent_json(text: str) -> dict:
    """Extract JSON dict from agent response text."""
    try:
        # Look for JSON block
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

    # Fallback: return empty dict, agent will fill missing_fields
    return {"missing_fields": ["parse_error"], "description": text[:300]}