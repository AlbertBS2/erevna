"""Orchestrator for coordinating all research analysis agents."""

import json
import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from research_structure_agent import ResearchStructureAgent
from validity_threat_agent import ValidityThreatAgent
from literature_scout_agent import LiteratureScoutAgent


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
    
    def analyze(self, text: str, parallel: bool = False) -> dict:
        """
        Run all agents on the research text.

        Args:
            text: Research text to analyze
            parallel: If True, run downstream agents in parallel.

        Returns:
            Dictionary containing results from all three agents
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

        return {
            "research_structure": research_structure,
            "validity_threats": validity_threats,
            "literature_scout": literature_scout,
        }
    
    def _analyze_downstream_parallel(
        self, research_json: str
    ) -> tuple[dict, dict]:
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
                        results[agent_name] = self._default_validity_threats(
                            error=f"Error: {str(e)}"
                        )
                    else:
                        results[agent_name] = self._default_literature_scout(
                            error=f"Error: {str(e)}"
                        )

        validity_output = results.get("validity_threats")
        literature_output = results.get("literature_scout")

        if isinstance(validity_output, str):
            validity_output = self._parse_validity_threats(validity_output)
        if isinstance(literature_output, str):
            literature_output = self._parse_literature_scout(literature_output)

        return (
            validity_output or self._default_validity_threats(),
            literature_output or self._default_literature_scout(),
        )
    
    def _analyze_downstream_sequential(
        self, research_json: str
    ) -> tuple[dict, dict]:
        """Run downstream agents sequentially using the research JSON."""
        validity_response = self.validity_agent.get_response(research_json)
        literature_response = self.literature_agent.get_response(research_json)

        validity_threats = self._parse_validity_threats(validity_response)
        literature_scout = self._parse_literature_scout(literature_response)

        return validity_threats, literature_scout

    def _parse_research_structure(self, response: str) -> dict:
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return self._merge_defaults(self._default_research_structure(), parsed)
            return self._default_research_structure(
                error="Failed to parse research structure: JSON is not an object"
            )
        except Exception as e:
            return self._default_research_structure(
                error=f"Failed to parse research structure: {str(e)}"
            )

    def _parse_validity_threats(self, response: str) -> dict:
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return self._merge_defaults(self._default_validity_threats(), parsed)
            return self._default_validity_threats(
                error="Failed to parse validity threats: JSON is not an object"
            )
        except Exception as e:
            return self._default_validity_threats(
                error=f"Failed to parse validity threats: {str(e)}"
            )

    def _parse_literature_scout(self, response: str) -> dict:
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return self._merge_defaults(self._default_literature_scout(), parsed)
            return self._default_literature_scout(
                error="Failed to parse literature scout: JSON is not an object"
            )
        except Exception as e:
            return self._default_literature_scout(
                error=f"Failed to parse literature scout: {str(e)}"
            )
    
    def analyze_and_print(self, text: str, parallel: bool = True) -> dict:
        """
        Analyze research text and print formatted results.
        
        Args:
            text: Research text to analyze
            parallel: If True, run agents in parallel
            
        Returns:
            Dictionary containing all results
        """
        print("🔍 Starting Research Analysis...\n")
        
        results = self.analyze(text, parallel=parallel)
        
        # Print Research Structure
        print("=" * 80)
        print("📋 RESEARCH STRUCTURE")
        print("=" * 80)
        research_structure = results["research_structure"]
        print(f"Research Question: {research_structure.get('research_question')}")
        print(f"Hypothesis: {research_structure.get('hypothesis')}")
        print(f"Method: {research_structure.get('method')}")
        variables = research_structure.get("variables") or []
        print(
            f"Variables: {', '.join(variables) if variables else 'None specified'}"
        )
        print(f"Dataset: {research_structure.get('dataset')}")
        print(f"Evaluation: {research_structure.get('evaluation')}")
        if research_structure.get("missing_elements"):
            print("\n⚠️  Missing Elements:")
            for elem in research_structure.get("missing_elements", []):
                print(f"  • {elem}")
        
        # Print Validity Threats
        print("\n" + "=" * 80)
        print("⚠️  VALIDITY THREATS")
        print("=" * 80)
        
        validity_threats = results["validity_threats"]
        if validity_threats.get("internal_validity"):
            print("\n🔴 Internal Validity:")
            for threat in validity_threats.get("internal_validity", []):
                print(f"  • {threat}")
        
        if validity_threats.get("external_validity"):
            print("\n🟠 External Validity:")
            for threat in validity_threats.get("external_validity", []):
                print(f"  • {threat}")
        
        if validity_threats.get("construct_validity"):
            print("\n🟡 Construct Validity:")
            for threat in validity_threats.get("construct_validity", []):
                print(f"  • {threat}")
        
        if validity_threats.get("conclusion_validity"):
            print("\n🟢 Conclusion Validity:")
            for threat in validity_threats.get("conclusion_validity", []):
                print(f"  • {threat}")
        
        if validity_threats.get("mitigation_suggestions"):
            print("\n💡 Mitigation Suggestions:")
            for suggestion in validity_threats.get("mitigation_suggestions", []):
                print(f"  • {suggestion}")
        
        # Print Literature Scout
        print("\n" + "=" * 80)
        print("📚 LITERATURE SCOUT")
        print("=" * 80)
        
        literature_scout = results["literature_scout"]
        if literature_scout.get("inferred_topics"):
            print("\n🎯 Inferred Topics:")
            for topic in literature_scout.get("inferred_topics", []):
                print(f"  • {topic}")
        
        if literature_scout.get("search_queries"):
            print("\n🔎 Search Queries:")
            for query in literature_scout.get("search_queries", []):
                print(f"  • {query}")
        
        if literature_scout.get("example_related_work"):
            print("\n📄 Example Related Work:")
            for work in literature_scout.get("example_related_work", []):
                print(f"  • {work}")
        
        print("\n" + "=" * 80)
        print("✅ Analysis Complete!")
        print("=" * 80)
        
        return results

    def _default_research_structure(self, error: Optional[str] = None) -> dict:
        base = {
            "research_question": "Not specified",
            "hypothesis": "Not specified",
            "method": "Not specified",
            "variables": [],
            "dataset": "Not specified",
            "evaluation": "Not specified",
            "missing_elements": [],
        }
        if error:
            base["missing_elements"].append(error)
        return base

    def _default_validity_threats(self, error: Optional[str] = None) -> dict:
        base = {
            "internal_validity": [],
            "external_validity": [],
            "construct_validity": [],
            "conclusion_validity": [],
            "mitigation_suggestions": [],
        }
        if error:
            base["mitigation_suggestions"].append(error)
        return base

    def _default_literature_scout(self, error: Optional[str] = None) -> dict:
        base = {
            "inferred_topics": [],
            "search_queries": [],
            "example_related_work": [],
        }
        if error:
            base["inferred_topics"].append(error)
        return base

    def _merge_defaults(self, defaults: dict, parsed: dict) -> dict:
        merged = defaults.copy()
        for key, value in parsed.items():
            if key in merged:
                merged[key] = value
        return merged
