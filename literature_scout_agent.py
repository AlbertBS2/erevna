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
        sys_instructions = f"""You are an expert research librarian and domain expert. Based on the research text:

1. **Infer research areas and topics**: Identify the main research domains, subfields, and cross-disciplinary areas

2. **Suggest search queries**: Create effective search keywords and queries for finding related literature (include both broad and specific queries)

3. **Suggest example paper titles**: Generate realistic example titles of papers that would be related to this research (these can be synthetic but should be plausible and informative)

You MUST respond with ONLY a valid JSON object in this exact format:
{{
  "inferred_topics": ["topic 1", "topic 2", "topic 3"],
  "search_queries": ["query 1", "query 2", "query 3"],
  "example_related_work": ["Example Title 1", "Example Title 2", "Example Title 3"]
}}

Generate at least 3-5 items for each category. Do not include any text before or after the JSON. Avoid any markdown formatting.
        """

        # Generate a response from the llm
        response = self.client.models.generate_content(
            model=self.model,
            contents=research_text,
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=sys_instructions
            )
        )
        response = response.text

        return response
