from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

class ValidityThreatAgent:
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
        sys_instructions = f"""

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
    

# if __name__ == "__main__":
#     # Example usage
#     load_dotenv()
#     GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
#     validity_threat_agent = ValidityThreatAgent(api_key=GOOGLE_API_KEY)
#     user_input = "Example research text to analyze for validity threats."
#     response = validity_threat_agent.get_response(user_input)
#     print(response)
