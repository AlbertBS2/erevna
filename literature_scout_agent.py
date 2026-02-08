"""Literature Scout Agent for discovering related research."""

from google import genai
from google.genai import types


class LiteratureScoutAgent:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def get_response(self, research_text: str) -> str:
        """
        Generate a response from the LLM.

        Args:
            research_text (str): The input.

        Returns:
            str: Generated response from the LLM.
        """

        # System instructions
        sys_instructions = f"""You are an expert research librarian and domain expert. Based on the given information on the Ongoing Research Structured Summary:

**Suggest example paper titles**: Generate realistic example titles of papers (with their authors) that would be related to this research (these can be synthetic but should be plausible and informative)

You MUST respond with ONLY a valid JSON object in this exact format:
{{
  "example_related_work": ["Paper 1", "Paper 2", "Paper 3", ..., "Paper n"]
}}

Generate at least 3-5 items if the given Ongoing Research Structured Summary is sufficiently detailed. If not, generate an empty list. Do not include any text before or after the JSON. Avoid any markdown formatting.
        """

        query = f"""Ongoing Research Structured Summary:

{research_text}
        """

        # Generate a response from the llm
        response = self.client.models.generate_content(
            model=self.model,
            contents=query,
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=sys_instructions
            )
        )
        response = response.text

        return response
