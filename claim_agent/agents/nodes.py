"""
claim_agent/agents/nodes.py
Every LangGraph node for the conversational claims assistant.
Fixed: intent detection, claim intake, dynamic slots, validation.
"""
from __future__ import annotations
import json, os, re, uuid
from datetime import datetime
from loguru import logger
from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential

from claim_agent.agents.conversation_state import (
    ConversationState, SlotState, Message,
    REQUIRED_SLOTS, SLOT_QUESTIONS, CLAIM_TYPE_EXTRA_SLOTS,
)

# ── LLM call ─────────────────────────────────────────────────────
def _mistral(messages: list[dict], temperature: float = 0.0) -> str:
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    resp   = client.chat.complete(
        model=os.getenv("MISTRAL_MODEL", "mistral-tiny"),
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _call(messages, temperature=0.0):
    return _mistral(messages, temperature)

def _last_user(state: ConversationState) -> str:
    for m in reversed(state.get("messages", [])):
        if m["role"] == "user":
            return m["content"]
    return ""

def _strip_json(raw: str) -> str:
    raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    return re.sub(r"\n?```$", "", raw)


# ─────────────────────────────────────────────────────────────────
# 1. INTENT DETECTOR (fixed with regex + few-shot)
# ─────────────────────────────────────────────────────────────────
_INTENT_SYS = """You are an intent classifier for an insurance assistant.

Classify the user's message into EXACTLY one of these categories:
- claim: user wants to FILE a NEW claim (e.g., "I want to file a claim", "I need to submit a claim", "start a claim", "file a claim for my accident")
- query: user asks a factual question about policy, coverage, or process (e.g., "what's my deductible?")
- status: user asks about an existing claim (e.g., "check my claim status", "where is my claim CLM-123")
- greeting: hello, hi, thank you, goodbye
- confirm: yes, correct, proceed, submit
- cancel: no, cancel, stop, nevermind
- other: anything else

Examples:
User: "How do I file a claim?" → query
User: "I want to file a claim" → claim
User: "File a claim for my car accident" → claim
User: "What is my coverage limit?" → query
User: "Start a new claim" → claim
User: "Yes, submit it" → confirm
User: "CLM-ABC123" → status

Reply with ONLY the single word (lowercase). No punctuation, no extra text."""

def intent_detector(state: ConversationState) -> dict:
    msg = _last_user(state).lower().strip()
    if not msg:
        return {"intent": "greeting"}

    # ---- Regex fallback for common claim phrases ----
    claim_patterns = [
        r"file (a|my|an? )?claim",
        r"submit (a|my|an? )?claim",
        r"start (a|my|an? )?claim",
        r"new claim",
        r"make a claim",
        r"i want to claim",
        r"register (a|my )?claim",
    ]
    for pattern in claim_patterns:
        if re.search(pattern, msg):
            logger.info(f"[Intent] Regex matched claim: '{msg[:50]}'")
            return {"intent": "claim"}

    # ---- LLM call ----
    try:
        intent = _call([
            {"role": "system", "content": _INTENT_SYS},
            {"role": "user",   "content": msg},
        ]).lower().strip().strip('"').strip("'")
        if intent not in ("query","claim","status","greeting","confirm","cancel","other"):
            intent = "other"
    except Exception as e:
        logger.warning(f"Intent detection failed: {e}")
        intent = "other"

    logger.info(f"[Intent] '{msg[:50]}' → {intent}")
    return {"intent": intent}


# ─────────────────────────────────────────────────────────────────
# 2. QUERY HANDLER (RAG) – unchanged, works fine
# ─────────────────────────────────────────────────────────────────
_QUERY_SYS = """You are a helpful insurance policy assistant.
Answer using ONLY the provided policy context. Be concise (2–4 sentences).
Cite the relevant clause. If not found, say so clearly.
End with: "Would you like to file a claim or ask anything else?"
"""

def query_handler(state: ConversationState) -> dict:
    msg     = _last_user(state)
    file_id = state.get("file_id","")
    chunks  = []
    confidence = 0.0
    try:
        from rag_pipeline.retriever import PolicyRetriever
        results   = PolicyRetriever().retrieve(msg, file_id, top_k=4)
        chunks    = results
        confidence= results[0]["score"] if results else 0.0
    except Exception as e:
        logger.warning(f"RAG failed: {e}")

    context = "\n\n---\n\n".join(c["text"] for c in chunks) if chunks else "No policy context available."
    try:
        answer = _call([
            {"role":"system","content":_QUERY_SYS},
            {"role":"user",  "content":f"Policy context:\n{context}\n\nQuestion: {msg}"},
        ], temperature=0.1)
    except Exception as e:
        answer = f"I couldn't retrieve that information right now. ({e})"

    return {
        "rag_answer":     answer,
        "rag_sources":    [c["text"][:100] for c in chunks[:2]],
        "rag_confidence": round(confidence, 3),
        "assistant_response": answer,
        "response_type":  "text",
    }


# ─────────────────────────────────────────────────────────────────
# 3. CLAIM INITIATOR (force reset)
# ─────────────────────────────────────────────────────────────────
def claim_initiator(state: ConversationState) -> dict:
    name = state.get("claimant_name","there")
    response = (
        f"Of course, I'll help you file a claim, **{name}**. "
        "Let me collect a few details.\n\n"
        + SLOT_QUESTIONS["claim_type"]
    )
    blank_slots = SlotState(claim_type=None, incident_date=None,
                            incident_description=None, claimed_amount=None,
                            hospital_name=None, vehicle_number=None,
                            contact_number=None)
    return {
        "mode":              "claim_intake",
        "slots":             blank_slots,
        "missing_slots":     list(REQUIRED_SLOTS),
        "coverage_result":   None,
        "claim_id":          None,
        "claim_submitted":   False,
        "assistant_response": response,
        "response_type":     "text",
        "next_question":     None,
        "intent":            None,   # clear stale intent
    }


# ─────────────────────────────────────────────────────────────────
# 4. SLOT EXTRACTOR (improved JSON + validation)
# ─────────────────────────────────────────────────────────────────
_SLOT_SYS = """Extract insurance claim fields from the user message.
Return ONLY valid JSON (no markdown) with these keys:
{
  "claim_type": "health|vehicle|home|travel|other or null",
  "incident_date": "YYYY-MM-DD or null",
  "incident_description": "string or null",
  "claimed_amount": number_or_null,
  "hospital_name": "string or null",
  "vehicle_number": "string or null",
  "contact_number": "string or null"
}
If a value is not present, use null.
Convert dates to YYYY-MM-DD if possible."""

def slot_extractor(state: ConversationState) -> dict:
    msg     = _last_user(state)
    current = dict(state.get("slots") or {})
    try:
        raw  = _call([
            {"role":"system","content":_SLOT_SYS},
            {"role":"user",  "content":msg},
        ], temperature=0.0)
        extracted = json.loads(_strip_json(raw))
        # Validate and convert types
        for k, v in extracted.items():
            if v is None:
                continue
            if k == "claimed_amount":
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
            if k == "incident_date" and isinstance(v, str):
                # simple date validation (can be enhanced)
                if not re.match(r"\d{4}-\d{2}-\d{2}", v):
                    continue
            if v is not None and k in current:
                current[k] = v
    except Exception as e:
        logger.warning(f"Slot extraction failed: {e}")

    # Compute missing slots (required + claim-type-specific)
    missing = [s for s in REQUIRED_SLOTS if not current.get(s)]
    ctype = current.get("claim_type")
    if ctype and ctype in CLAIM_TYPE_EXTRA_SLOTS:
        for extra in CLAIM_TYPE_EXTRA_SLOTS[ctype]:
            if not current.get(extra) and extra not in missing:
                missing.append(extra)

    return {
        "slots":         SlotState(**{k: current.get(k) for k in SlotState.__annotations__}),
        "missing_slots": missing,
    }


# ─────────────────────────────────────────────────────────────────
# 5. QUESTION CHOOSER (dynamic, with progress)
# ─────────────────────────────────────────────────────────────────
def question_chooser(state: ConversationState) -> dict:
    missing = state.get("missing_slots", [])
    if not missing:
        # All required slots filled → will go to coverage checker
        return {
            "next_question":      None,
            "assistant_response": "Got it — checking your coverage now… 🔍",
            "response_type":      "text",
        }
    # Show progress
    filled = len(REQUIRED_SLOTS) - len([s for s in missing if s in REQUIRED_SLOTS])
    total  = len(REQUIRED_SLOTS)
    progress = f"*({filled}/{total} details collected)*\n\n"
    # Use dynamic question mapping
    q = SLOT_QUESTIONS.get(missing[0], f"Can you tell me about {missing[0].replace('_',' ')}?")
    return {
        "next_question":      missing[0],
        "assistant_response": progress + q,
        "response_type":      "text",
    }


# ─────────────────────────────────────────────────────────────────
# 6. COVERAGE CHECKER (improved error handling)
# ─────────────────────────────────────────────────────────────────
_COV_SYS = """You are an insurance coverage analyst.
Return ONLY valid JSON (no markdown):
{
  "is_covered": true|false,
  "coverage_limit": number_in_INR,
  "deductible": number_in_INR,
  "key_clause": "one-sentence clause summary",
  "exclusions": ["list","of","exclusions"],
  "confidence": 0.0_to_1.0
}"""

def coverage_checker(state: ConversationState) -> dict:
    slots   = state.get("slots") or {}
    file_id = state.get("file_id","")
    query   = f"{slots.get('claim_type','')} {slots.get('incident_description','')} {slots.get('claimed_amount','')}"

    chunks = []
    try:
        from rag_pipeline.retriever import PolicyRetriever
        chunks = PolicyRetriever().retrieve(query, file_id, top_k=5)
    except Exception as e:
        logger.warning(f"Coverage RAG failed: {e}")

    context = "\n\n---\n\n".join(c["text"] for c in chunks) if chunks else "No policy context."
    try:
        raw    = _call([
            {"role":"system","content":_COV_SYS},
            {"role":"user",  "content": f"Claim: {json.dumps(dict(slots), default=str)}\n\nPolicy:\n{context}"},
        ], temperature=0.0)
        result = json.loads(_strip_json(raw))
    except Exception as e:
        logger.error(f"Coverage LLM failed: {e}")
        result = {"is_covered": None, "coverage_limit":0, "deductible":0,
                  "key_clause":"Unable to determine","exclusions":[],"confidence":0.0}

    if result.get("is_covered") is True:
        claimed   = float(slots.get("claimed_amount") or 0)
        limit     = float(result.get("coverage_limit") or 0)
        deductible= float(result.get("deductible") or 0)
        estimated = max(0, min(claimed, limit) - deductible)
        response = (
            f"✅ **Your claim is covered!**\n\n"
            f"| Detail | Amount |\n|---|---|\n"
            f"| Claimed | ₹{claimed:,.0f} |\n"
            f"| Coverage limit | ₹{limit:,.0f} |\n"
            f"| Deductible | ₹{deductible:,.0f} |\n"
            f"| **Estimated payout** | **₹{estimated:,.0f}** |\n\n"
            f"*{result.get('key_clause','')}*\n\n"
            "**Shall I submit this claim?** Reply **yes** to confirm."
        )
        rtype = "coverage_card"
    elif result.get("is_covered") is False:
        excl = ", ".join(result.get("exclusions",[])) or "policy exclusions"
        response = (
            f"⚠️ This claim may **not be covered** due to: {excl}.\n\n"
            f"*{result.get('key_clause','')}*\n\n"
            "Would you still like me to submit it for manual adjuster review? Reply **yes** or **no**."
        )
        rtype = "text"
    else:
        response = "I wasn't able to confirm coverage. Would you like me to submit it for adjuster review? Reply **yes** or **no**."
        rtype = "text"

    return {"coverage_result": result, "assistant_response": response, "response_type": rtype}


# ─────────────────────────────────────────────────────────────────
# 7. CLAIM SUBMITTER (adds summary before final)
# ─────────────────────────────────────────────────────────────────
def claim_submitter(state: ConversationState) -> dict:
    import redis as _redis
    claim_id = "CLM-" + str(uuid.uuid4())[:6].upper()
    slots    = state.get("slots") or {}
    coverage = state.get("coverage_result") or {}

    record = {
        "claim_id":     claim_id,
        "session_id":   state.get("session_id"),
        "policy_number":state.get("policy_number"),
        "claimant_name":state.get("claimant_name"),
        "submitted_at": datetime.now().isoformat(),
        "slots":        dict(slots),
        "coverage":     coverage,
        "status":       "submitted",
    }
    try:
        r = _redis.Redis(host=os.getenv("REDIS_HOST","localhost"),
                         port=int(os.getenv("REDIS_PORT",6379)),
                         decode_responses=True)
        r.setex(f"claim:{claim_id}", 86400*30, json.dumps(record, default=str))
        logger.success(f"Claim {claim_id} saved")
    except Exception as e:
        logger.warning(f"Redis save failed: {e}")

    # analytics
    try:
        from rag_pipeline.analytics import Analytics
        Analytics().log_claim(claim_id, state.get("policy_number",""))
    except Exception: pass

    claimed   = float(slots.get("claimed_amount") or 0)
    limit     = float(coverage.get("coverage_limit") or claimed)
    deductible= float(coverage.get("deductible") or 0)
    estimated = max(0, min(claimed, limit) - deductible)

    response = (
        f"🎉 **Claim submitted successfully!**\n\n"
        f"**Claim ID:** `{claim_id}`\n"
        f"**Estimated payout:** ₹{estimated:,.0f}\n\n"
        "You'll receive a confirmation within **24 hours**. "
        "An adjuster will review within **3–5 business days**.\n\n"
        "Is there anything else I can help you with?"
    )
    return {
        "claim_id":          claim_id,
        "claim_submitted":   True,
        "mode":              "chat",
        "assistant_response":response,
        "response_type":     "claim_summary",
    }


# ─────────────────────────────────────────────────────────────────
# 8. GREETING (unchanged)
# ─────────────────────────────────────────────────────────────────
def greeting_handler(state: ConversationState) -> dict:
    name = state.get("claimant_name","")
    msg  = _last_user(state).lower()
    if any(w in msg for w in ["thank","bye","goodbye"]):
        resp = f"You're welcome{', ' + name if name else ''}! Take care. 😊"
    else:
        resp = (
            f"Hello{', ' + name if name else ''}! 👋 I'm your PolicyAI assistant.\n\n"
            "I can help you:\n"
            "• **Answer questions** about your policy coverage\n"
            "• **File a claim** step-by-step through conversation\n"
            "• **Check claim status** with your claim ID\n\n"
            "What would you like to do today?"
        )
    return {"assistant_response": resp, "response_type": "text"}


# ─────────────────────────────────────────────────────────────────
# 9. CANCEL (unchanged)
# ─────────────────────────────────────────────────────────────────
def cancel_handler(state: ConversationState) -> dict:
    blank = SlotState(claim_type=None, incident_date=None, incident_description=None,
                      claimed_amount=None, hospital_name=None, vehicle_number=None,
                      contact_number=None)
    return {
        "mode":              "chat",
        "slots":             blank,
        "missing_slots":     [],
        "claim_submitted":   False,
        "coverage_result":   None,
        "assistant_response":"No problem — claim cancelled. Ask me anything about your policy or start a new claim anytime.",
        "response_type":     "text",
    }


# ─────────────────────────────────────────────────────────────────
# 10. STATUS CHECK (unchanged)
# ─────────────────────────────────────────────────────────────────
def status_checker(state: ConversationState) -> dict:
    msg   = _last_user(state)
    match = re.search(r"CLM-[A-Z0-9]{6}", msg.upper())
    if not match:
        return {
            "assistant_response":"Please share your **Claim ID** (format: `CLM-XXXXXX`) and I'll look it up.",
            "response_type":     "text",
        }
    claim_id = match.group(0)
    try:
        import redis as _redis
        r   = _redis.Redis(host=os.getenv("REDIS_HOST","localhost"),
                           port=int(os.getenv("REDIS_PORT",6379)),
                           decode_responses=True)
        raw = r.get(f"claim:{claim_id}")
        if raw:
            rec  = json.loads(raw)
            resp = (
                f"**Claim `{claim_id}`**\n\n"
                f"| | |\n|---|---|\n"
                f"| Status | {rec.get('status','unknown').title()} |\n"
                f"| Submitted | {rec.get('submitted_at','')[:10]} |\n"
                f"| Type | {(rec.get('slots') or {}).get('claim_type','—').title()} |\n\n"
                "An adjuster will contact you within 3–5 business days."
            )
        else:
            resp = f"Claim `{claim_id}` not found. Please double-check the ID."
    except Exception:
        resp = f"Unable to look up claims right now. Please contact support with ID `{claim_id}`."
    return {"assistant_response": resp, "response_type": "text"}