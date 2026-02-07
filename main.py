"""FastAPI entrypoint for the research analysis system."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from orchestrator import ResearchAnalysisOrchestrator
from schemas import AnalysisOutput

load_dotenv()

app = FastAPI(title="Erevna Research Analysis API")


class AnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1)
    parallel: bool = True


@app.post("/analyze", response_model=AnalysisOutput)
def analyze_research(request: AnalysisRequest) -> AnalysisOutput:
    try:
        orchestrator = ResearchAnalysisOrchestrator()
        return orchestrator.analyze(request.text, parallel=request.parallel)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
