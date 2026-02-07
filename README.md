# Erevna - Multi-Agent Research Analysis System

A modular multi-agentic architecture that uses LangChain and Google's Gemini API to analyze research proposals and papers.

## Features

### Three Specialized Agents

1. **Research Structure Agent** - Extracts and identifies:
   - Research question
   - Hypothesis
   - Variables (independent, dependent, control)
   - Dataset information
   - Methodology
   - Evaluation metrics
   - Missing components

2. **Validity Threat Agent** - Analyzes research validity:
   - Internal validity threats
   - External validity threats
   - Construct validity threats
   - Conclusion validity threats
   - Actionable mitigation suggestions

3. **Literature Scout Agent** - Discovers related research:
   - Inferred research topics and areas
   - Search keywords and queries
   - Example related paper titles

## Architecture

- **Modular agent classes**: Each agent is independent and reusable
- **Orchestrator graph**: Coordinates agents with parallel or sequential execution
- **Environment-based config**: API keys from environment variables
- **Minimal dependencies**: Only essential packages (LangChain, Gemini)

## Installation

1. Clone or navigate to the project directory:
```bash
cd /home/abs/projects/erevna
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key:
```bash
cp .env.example .env
# Edit .env and add your Google API key
```

Get your API key from: https://makersuite.google.com/app/apikey

## Usage

### Run the API Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Call the API

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Your research proposal or paper text here...", "parallel": true}'
```

### Python Client Example

```python
import requests

payload = {
    "text": "Your research proposal or paper text here...",
    "parallel": True
}

response = requests.post("http://localhost:8000/analyze", json=payload, timeout=300)
response.raise_for_status()
data = response.json()
print(data["research_structure"]["research_question"])
```

## Project Structure

```
erevna/
├── main.py                      # Example usage and entry point
├── orchestrator.py              # Orchestrates all agents
├── research_structure_agent.py  # Agent 1: Extract research components
├── validity_threat_agent.py     # Agent 2: Analyze validity threats
├── literature_scout_agent.py    # Agent 3: Suggest related work
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
└── README.md                    # This file
```

## Output Format

All agents return JSON-compatible dictionaries:

```python
{
  "research_structure": {
    "research_question": str,
    "hypothesis": str,
    "method": str,
    "variables": List[str],
    "dataset": str,
    "evaluation": str,
    "missing_elements": List[str]
  },
  "validity_threats": {
    "internal_validity": List[str],
    "external_validity": List[str],
    "construct_validity": List[str],
    "conclusion_validity": List[str],
    "mitigation_suggestions": List[str]
  },
  "literature_scout": {
    "inferred_topics": List[str],
    "search_queries": List[str],
    "example_related_work": List[str]
  }
}
```

## Configuration

The system uses environment variables for configuration:

- `GOOGLE_API_KEY`: Your Google API key for Gemini (required)

## License

MIT License
