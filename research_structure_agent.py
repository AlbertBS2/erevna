"""Research Structure Agent for extracting research components."""

import json
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from schemas import ResearchStructureOutput
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os


class ResearchStructureAgent:
    """Agent to extract research structure and identify missing components."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """
        Initialize the Research Structure Agent.
        
        Args:
            api_key: Google API key for Gemini
            model: Gemini model to use
        """
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
        sys_instructions = f"""You are an expert research analyst. Analyze the given text and extract:
- Research question
- Hypothesis
- Research methodology
- Variables (independent, dependent, control)
- Dataset information
- Evaluation metrics

Identify any missing components that would make the research more complete.

You MUST respond with ONLY a valid JSON object in this exact format:
{{
  "research_question": "extracted or 'Not specified'",
  "hypothesis": "extracted or 'Not specified'",
  "method": "extracted methodology or 'Not specified'",
  "variables": ["list", "of", "variables"],
  "dataset": "dataset description or 'Not specified'",
  "evaluation": "evaluation metrics or 'Not specified'"
}}

Do not include any text before or after the JSON. Avoid any markdown formatting.
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
#     research_structure_agent = ResearchStructureAgent(api_key=GOOGLE_API_KEY)
#     user_input = """
#     y
# November 26, 2025
# 1 REMOVED
# REMOVED
# Contents
# 1 Title 3
# 2 Abstract 3
# 3 Background 3
# 4 Description of Tasks 4
# 5 Methods 4
# 6 Relevant Courses 5
# 7 Delimitations 5
# 8 Time Plan 5
# 2 REMOVED
# REMOVED
# REMOVED
# 1 Title
# Efficient Knowledge Transfer in Federated Learning for Heterogeneous Au-
# tonomous Driving Systems
# 2 Abstract
# The project investigates knowledge distillation as a method to support knowl-
# edge transfer between diverse models and hardware setups in autonomous
# vehicles, exploring how knowledge can be efficiently transferred between AI
# models with different architectures. The work will develop and evaluate
# techniques for updating AI models when vehicle sensors or hardware change,
# without the need for full retraining or collecting extensive new datasets. The
# effectiveness of the proposed approach will be analyzed through experiments
# using multi-modal sensor data, including vehicle control signals, geographi-
# cal positions, and lidar, radar, and camera measurements. The project will
# develop and experiment with a federated learning framework that incorpo-
# rates knowledge distillation to maintain model performance and adaptability
# in real-time, large-scale deployments.
# 3 Background
# Autonomous vehicles rely on AI models and vast amounts of multi-modal
# sensor data—including vehicle control signals, GPS, lidar, radar, and cam-
# era inputs—to perceive and navigate their environment. When vehicles are
# updated and new models are developed, sensors and hardware often change,
# which in turn also affects the AI models used. One approach would be to cre-
# ate a new AI model from scratch and collect new data each time the vehicle
# platform is updated. A more efficient solution would be to transfer knowledge
# between models with varying architectures. In this Master’s thesis project,
# we aim to investigate how knowledge distillation mechanisms can facilitate
# knowledge transfer between diverse models and hardware setups, ensuring
# that learning can continue even when architectures change. The work will
# use the Zenseact Open Dataset [1] and also explore knowledge distillation in
# a federated learning context.
# 3 REMOVED
# REMOVED
# REMOVED
# 4 Description of Tasks
# The project will involve the design, development, training, implementation,
# and evaluation of machine learning models, knowledge distillation methods,
# and federated learning mechanisms that enable heterogeneous autonomous
# driving systems to exchange information and adapt across differing architec-
# tures and hardware setups. Work includes:
# • Developing and implementing knowledge distillation pipelines for trans-
# ferring information between heterogeneous AI models in both simula-
# tions and real-world driving scenarios.
# • Training and evaluating models under sensor and hardware shifts, as-
# sessing robustness, efficiency, and transfer quality on the Zenseact Open
# Dataset.
# • Developing and experimenting with a federated learning framework
# that integrates distillation to enable model updates without central-
# ized data collection.
# • Benchmarking against developed baseline models.
# 5 Methods
# • Techniques: Knwoledge distillation mechanisms, federated machine
# learning, deep learning, multimodal models.
# • Data: Zenseact Open Dataset [1] (lidar, radar, camera, GPS, vehicle-
# control logs), real-world driving scenarios.
# • Evaluation: Autonomous driving models task-specific metrics, ro-
# bustness across sensor and hardware shifts, inference efficiency.
# • Tools: PyTorch, NumPy, Scikit-Learn, federated learning frameworks,
# simulation environments, GPU-accelerated training, version control (Git).
# • Literature: Knowledge distillation, heterogeneous model adaptation,
# multimodal learning, federated learning in autonomous systems.
# 4 REMOVED
# REMOVED
# REMOVED
# 6 Relevant Courses
# • 1RT720 - Deep Learning
# • 1RT700 - Statistical Machine Learning
# • 1TD169 - Data Engineering I
# • 1TD076 - Data Engineering II
# • 1MS041 - Introduction to Data Science
# 7 Delimitations
# The project is focused on the design, development, training, implementation,
# and evaluation of machine learning models, knowledge distillation methods,
# and federated learning mechanisms. Hardware level programming, sensor
# calibration and synchronization, and low-level engineering are out of scope.
# 8 Time Plan
# • Literature review on heterogeneous model adaptation, multimodal learn-
# ing, knowledge distillation, and federated learning. (weeks 1-2)
# • Introduction to the federated learning framework. Dataset familiariza-
# tion, preprocessing pipelines, initial baseline models. (weeks 3-4)
# • Design and implementation of knowledge distillation frameworks. Train-
# ing of models. Initial experiments and evaluations. (weeks 5-8)
# • Development of the federated-learning framework. Integration of distil-
# lation into the federated pipeline. Federated experiments on distributed
# nodes. (weeks 9-12)
# • Robustness studies under sensor and hardware shifts. Refinement of
# models, hyperparameter tuning, and multimodal experiments. (weeks
# 13-16)
# • Simulation-based validation. Performance analysis and comparison to
# baselines. (weeks 17-18)
# 5 REMOVED
# REMOVED
# REMOVED
# • Report writing, refinement, preparation for presentation. (weeks 19-20)
# Report writing will also be done in parallel to the detailed tasks throughout
# most of the weeks. Frequent meetings with supervisor throughout.
# """

#     response = research_structure_agent.get_response(user_input)

    # # Save response to json file
    # json_response = json.loads(response)
    # with open("research_structure_output.json", "w") as f:
    #     f.write(json.dumps(json_response, indent=4))





        
    
    # def analyze(self, text: str) -> ResearchStructureOutput:
    #     """
    #     Analyze text and extract research structure.
        
    #     Args:
    #         text: Research text to analyze
            
    #     Returns:
    #         ResearchStructureOutput with extracted components
    #     """
    #     try:
    #         response = self.chain.invoke({"text": text})
    #         result = json.loads(response.content)
    #         return ResearchStructureOutput(**result)
    #     except json.JSONDecodeError as e:
    #         # Fallback if JSON parsing fails
    #         return ResearchStructureOutput(
    #             missing_elements=[f"Failed to parse response: {str(e)}"]
    #         )
    #     except Exception as e:
    #         return ResearchStructureOutput(
    #             missing_elements=[f"Error in analysis: {str(e)}"]
    #         )
