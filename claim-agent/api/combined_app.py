"""
Standalone combined FastAPI app.
Mounts both PolicyAI and claim routes together.

Run:  uvicorn api.combined_app:app --reload --port 8000

To integrate into your existing PolicyAI main.py instead, add just 2 lines:
    from api.claim_routes import router as claim_router
    app.include_router(claim_router)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from api.claim_routes import router as claim_router
import os

app = FastAPI(
    title="PolicyAI + Claim Processing",
    description="Insurance document Q&A + Agentic claim automation",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(claim_router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "features": ["policy_qa", "claim_processing"]
    }


@app.get("/", response_class=FileResponse)
def serve_dashboard():
    """Serve the claim processing dashboard."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "..", "dashboard.html")
    return FileResponse(os.path.abspath(dashboard_path))