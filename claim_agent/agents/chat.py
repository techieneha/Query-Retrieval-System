"""
claim_agent/agents/chat.py
FastAPI router — mounted on the main app at /api/v1/chat
"""
from __future__ import annotations
import asyncio, json, os, uuid
from datetime import datetime
from typing import Optional

import redis as _redis
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger

from claim_agent.agents.graph import process_message

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

SESSION_TTL = 60 * 60 * 4   # 4 hours


# ── Redis helpers ─────────────────────────────────────────────────
def _r() -> Optional[_redis.Redis]:
    try:
        r = _redis.Redis(host=os.getenv("REDIS_HOST","localhost"),
                         port=int(os.getenv("REDIS_PORT",6379)),
                         decode_responses=True)
        r.ping(); return r
    except Exception:
        return None

def _save(sid: str, state: dict):
    r = _r()
    if r: r.setex(f"chat:{sid}", SESSION_TTL, json.dumps(state, default=str))

def _load(sid: str) -> dict | None:
    r = _r()
    if not r: return None
    raw = r.get(f"chat:{sid}")
    return json.loads(raw) if raw else None


# ── Models ────────────────────────────────────────────────────────
class StartSessionReq(BaseModel):
    file_id:       str
    policy_number: str
    claimant_name: str = ""

class MsgReq(BaseModel):
    session_id: str
    message:    str

class MsgResp(BaseModel):
    session_id:    str
    response:      str
    response_type: str
    claim_id:      Optional[str] = None
    mode:          str
    metadata:      dict = {}


# ── Endpoints ─────────────────────────────────────────────────────
@router.post("/session")
async def start_session(req: StartSessionReq):
    sid  = str(uuid.uuid4())
    name = req.claimant_name or "there"
    greeting = (
        f"Hello, **{name}**! 👋 I'm your PolicyAI claims assistant.\n\n"
        "I can help you:\n"
        "• **Answer questions** about your insurance coverage\n"
        "• **File a claim** step-by-step through conversation\n"
        "• **Check claim status** with your claim ID\n\n"
        "What would you like to do today?"
    )
    _save(sid, {
        "session_id":    sid,
        "file_id":       req.file_id,
        "policy_number": req.policy_number,
        "claimant_name": req.claimant_name,
        "messages":      [{"role":"assistant","content":greeting}],
        "intent":None, "mode":"chat",
        "rag_answer":None,"rag_sources":None,"rag_confidence":None,
        "slots":{"claim_type":None,"incident_date":None,"incident_description":None,
                 "claimed_amount":None,"hospital_name":None,"vehicle_number":None,"contact_number":None},
        "missing_slots":[], "next_question":None,
        "coverage_result":None, "claim_id":None, "claim_submitted":False,
        "assistant_response":greeting, "response_type":"text", "errors":[],
    })
    return {"session_id": sid, "greeting": greeting}


@router.post("/message", response_model=MsgResp)
async def send_message(req: MsgReq):
    state = _load(req.session_id)
    if not state:
        raise HTTPException(404, f"Session {req.session_id} not found")
    try:
        updated = process_message(
            session_id=req.session_id, user_message=req.message,
            file_id=state.get("file_id",""),
            policy_number=state.get("policy_number",""),
            claimant_name=state.get("claimant_name",""),
            existing_state=state,
        )
        _save(req.session_id, updated)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(500, str(e))

    return MsgResp(
        session_id=req.session_id,
        response=updated.get("assistant_response",""),
        response_type=updated.get("response_type","text"),
        claim_id=updated.get("claim_id"),
        mode=updated.get("mode","chat"),
        metadata={"rag_confidence": updated.get("rag_confidence")},
    )


@router.get("/{session_id}")
def get_history(session_id: str):
    state = _load(session_id)
    if not state: raise HTTPException(404,"Session not found")
    return {
        "session_id":    session_id,
        "messages":      state.get("messages",[]),
        "mode":          state.get("mode","chat"),
        "claim_id":      state.get("claim_id"),
        "claim_submitted":state.get("claim_submitted",False),
    }


@router.get("/{session_id}/stream")
async def stream_message(session_id: str, message: str):
    """SSE streaming — simulates token-by-token output."""
    state = _load(session_id)
    if not state: raise HTTPException(404,"Session not found")

    async def gen():
        try:
            updated = process_message(
                session_id=session_id, user_message=message,
                file_id=state.get("file_id",""),
                policy_number=state.get("policy_number",""),
                claimant_name=state.get("claimant_name",""),
                existing_state=state,
            )
            _save(session_id, updated)

            full  = updated.get("assistant_response","")
            rtype = updated.get("response_type","text")
            words = full.split(" ")

            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words)-1 else "")
                yield f"data: {json.dumps({'token':chunk,'done':False,'type':rtype})}\n\n"
                await asyncio.sleep(0.018)

            yield f"data: {json.dumps({'token':'','done':True,'type':rtype,'claim_id':updated.get('claim_id'),'mode':updated.get('mode','chat')})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error':str(e),'done':True,'type':'error'})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})