"""Orchestrator for coordinating all research analysis agents."""

import json
import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from research_structure_agent import ResearchStructureAgent
from validity_threat_agent import ValidityThreatAgent
from literature_scout_agent import LiteratureScoutAgent
from schemas import (
    AnalysisOutput,
    ResearchStructureOutput,
    ValidityThreatOutput,
    LiteratureScoutOutput,
)


class ResearchAnalysisOrchestrator:
    """
    Orchestrates the three agents to perform comprehensive research analysis.
    
    Agents can run in parallel for efficiency.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        """
        Initialize the orchestrator with all agents.
        
        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var
            model: Gemini model to use
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not found. "
                    "Set it as environment variable or pass as parameter."
                )
        
        # Initialize all agents
        self.research_agent = ResearchStructureAgent(api_key, model)
        self.validity_agent = ValidityThreatAgent(api_key, model)
        self.literature_agent = LiteratureScoutAgent(api_key, model)
    
    def analyze(self, text: str, parallel: bool = True) -> AnalysisOutput:
        """
        Run all agents on the research text.

        Args:
            text: Research text to analyze
            parallel: If True, run downstream agents in parallel.

        Returns:
            AnalysisOutput containing results from all three agents
        """
        research_json = self.research_agent.get_response(text)
        research_structure = self._parse_research_structure(research_json)

        if parallel:
            validity_threats, literature_scout = self._analyze_downstream_parallel(
                research_json
            )
        else:
            validity_threats, literature_scout = self._analyze_downstream_sequential(
                research_json
            )

        return AnalysisOutput(
            research_structure=research_structure,
            validity_threats=validity_threats,
            literature_scout=literature_scout,
        )
    
    def _analyze_downstream_parallel(
        self, research_json: str
    ) -> tuple[ValidityThreatOutput, LiteratureScoutOutput]:
        """Run downstream agents in parallel using the research JSON."""
        results: dict[str, object] = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_agent = {
                executor.submit(
                    self.validity_agent.get_response, research_json
                ): "validity_threats",
                executor.submit(
                    self.literature_agent.get_response, research_json
                ): "literature_scout",
            }

            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    results[agent_name] = future.result()
                except Exception as e:
                    print(f"Error in {agent_name}: {e}")
                    if agent_name == "validity_threats":
                        results[agent_name] = ValidityThreatOutput(
                            mitigation_suggestions=[f"Error: {str(e)}"]
                        )
                    else:
                        results[agent_name] = LiteratureScoutOutput(
                            inferred_topics=[f"Error: {str(e)}"]
                        )

        validity_output = results.get("validity_threats")
        literature_output = results.get("literature_scout")

        if isinstance(validity_output, str):
            validity_output = self._parse_validity_threats(validity_output)
        if isinstance(literature_output, str):
            literature_output = self._parse_literature_scout(literature_output)

        return (
            validity_output or ValidityThreatOutput(),
            literature_output or LiteratureScoutOutput(),
        )
    
    def _analyze_downstream_sequential(
        self, research_json: str
    ) -> tuple[ValidityThreatOutput, LiteratureScoutOutput]:
        """Run downstream agents sequentially using the research JSON."""
        validity_response = self.validity_agent.get_response(research_json)
        literature_response = self.literature_agent.get_response(research_json)

        validity_threats = self._parse_validity_threats(validity_response)
        literature_scout = self._parse_literature_scout(literature_response)

        return validity_threats, literature_scout

    def _parse_research_structure(self, response: str) -> ResearchStructureOutput:
        try:
            parsed = json.loads(response)
            return ResearchStructureOutput(**parsed)
        except Exception as e:
            return ResearchStructureOutput(
                missing_elements=[f"Failed to parse research structure: {str(e)}"]
            )

    def _parse_validity_threats(self, response: str) -> ValidityThreatOutput:
        try:
            parsed = json.loads(response)
            return ValidityThreatOutput(**parsed)
        except Exception as e:
            return ValidityThreatOutput(
                mitigation_suggestions=[f"Failed to parse validity threats: {str(e)}"]
            )

    def _parse_literature_scout(self, response: str) -> LiteratureScoutOutput:
        try:
            parsed = json.loads(response)
            return LiteratureScoutOutput(**parsed)
        except Exception as e:
            return LiteratureScoutOutput(
                inferred_topics=[f"Failed to parse literature scout: {str(e)}"]
            )
    
    def analyze_and_print(self, text: str, parallel: bool = True) -> AnalysisOutput:
        """
        Analyze research text and print formatted results.
        
        Args:
            text: Research text to analyze
            parallel: If True, run agents in parallel
            
        Returns:
            AnalysisOutput containing all results
        """
        print("🔍 Starting Research Analysis...\n")
        
        results = self.analyze(text, parallel=parallel)
        
        # Print Research Structure
        print("=" * 80)
        print("📋 RESEARCH STRUCTURE")
        print("=" * 80)
        print(f"Research Question: {results.research_structure.research_question}")
        print(f"Hypothesis: {results.research_structure.hypothesis}")
        print(f"Method: {results.research_structure.method}")
        print(f"Variables: {', '.join(results.research_structure.variables) if results.research_structure.variables else 'None specified'}")
        print(f"Dataset: {results.research_structure.dataset}")
        print(f"Evaluation: {results.research_structure.evaluation}")
        if results.research_structure.missing_elements:
            print("\n⚠️  Missing Elements:")
            for elem in results.research_structure.missing_elements:
                print(f"  • {elem}")
        
        # Print Validity Threats
        print("\n" + "=" * 80)
        print("⚠️  VALIDITY THREATS")
        print("=" * 80)
        
        if results.validity_threats.internal_validity:
            print("\n🔴 Internal Validity:")
            for threat in results.validity_threats.internal_validity:
                print(f"  • {threat}")
        
        if results.validity_threats.external_validity:
            print("\n🟠 External Validity:")
            for threat in results.validity_threats.external_validity:
                print(f"  • {threat}")
        
        if results.validity_threats.construct_validity:
            print("\n🟡 Construct Validity:")
            for threat in results.validity_threats.construct_validity:
                print(f"  • {threat}")
        
        if results.validity_threats.conclusion_validity:
            print("\n🟢 Conclusion Validity:")
            for threat in results.validity_threats.conclusion_validity:
                print(f"  • {threat}")
        
        if results.validity_threats.mitigation_suggestions:
            print("\n💡 Mitigation Suggestions:")
            for suggestion in results.validity_threats.mitigation_suggestions:
                print(f"  • {suggestion}")
        
        # Print Literature Scout
        print("\n" + "=" * 80)
        print("📚 LITERATURE SCOUT")
        print("=" * 80)
        
        if results.literature_scout.inferred_topics:
            print("\n🎯 Inferred Topics:")
            for topic in results.literature_scout.inferred_topics:
                print(f"  • {topic}")
        
        if results.literature_scout.search_queries:
            print("\n🔎 Search Queries:")
            for query in results.literature_scout.search_queries:
                print(f"  • {query}")
        
        if results.literature_scout.example_related_work:
            print("\n📄 Example Related Work:")
            for work in results.literature_scout.example_related_work:
                print(f"  • {work}")
        
        print("\n" + "=" * 80)
        print("✅ Analysis Complete!")
        print("=" * 80)
        
        return results
