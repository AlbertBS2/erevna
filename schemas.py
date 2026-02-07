"""Typed schemas for agent outputs using Pydantic."""

from typing import List
from pydantic import BaseModel, Field


class ResearchStructureOutput(BaseModel):
    """Output schema for Research Structure Agent."""
    
    research_question: str = Field(
        default="",
        description="The main research question being investigated"
    )
    hypothesis: str = Field(
        default="",
        description="The hypothesis being tested"
    )
    method: str = Field(
        default="",
        description="The research methodology used"
    )
    variables: List[str] = Field(
        default_factory=list,
        description="Key variables in the research"
    )
    dataset: str = Field(
        default="",
        description="Dataset used or described"
    )
    evaluation: str = Field(
        default="",
        description="Evaluation metrics or approach"
    )
    missing_elements: List[str] = Field(
        default_factory=list,
        description="Components that are missing or unclear"
    )


class ValidityThreatOutput(BaseModel):
    """Output schema for Validity Threat Agent."""
    
    internal_validity: List[str] = Field(
        default_factory=list,
        description="Internal validity threats and concerns"
    )
    external_validity: List[str] = Field(
        default_factory=list,
        description="External validity threats and concerns"
    )
    construct_validity: List[str] = Field(
        default_factory=list,
        description="Construct validity threats and concerns"
    )
    conclusion_validity: List[str] = Field(
        default_factory=list,
        description="Conclusion validity threats and concerns"
    )
    mitigation_suggestions: List[str] = Field(
        default_factory=list,
        description="Actionable suggestions to mitigate threats"
    )


class LiteratureScoutOutput(BaseModel):
    """Output schema for Literature Scout Agent."""
    
    inferred_topics: List[str] = Field(
        default_factory=list,
        description="Inferred research areas and topics"
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="Suggested search keywords and queries"
    )
    example_related_work: List[str] = Field(
        default_factory=list,
        description="Example related paper titles"
    )


class AnalysisOutput(BaseModel):
    """Complete output from all agents."""
    
    research_structure: ResearchStructureOutput
    validity_threats: ValidityThreatOutput
    literature_scout: LiteratureScoutOutput
