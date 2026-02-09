"""FastAPI entrypoint for the research analysis system."""

from fastapi import Body, FastAPI, HTTPException
from dotenv import load_dotenv

from orchestrator import ResearchAnalysisOrchestrator

from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="Erevna Research Analysis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Erevna."}

@app.post("/analyze")
def analyze_research(payload: dict = Body(...)) -> dict:
    try:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Invalid JSON payload")

        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(
                status_code=400, detail="Field 'text' must be a non-empty string"
            )

        parallel = payload.get("parallel", True)
        if not isinstance(parallel, bool):
            raise HTTPException(
                status_code=400, detail="Field 'parallel' must be a boolean"
            )

        orchestrator = ResearchAnalysisOrchestrator()
        return orchestrator.analyze(text, parallel=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
