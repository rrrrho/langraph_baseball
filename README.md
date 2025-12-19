# Langraph Baseball Data Analysis

This project is a modular baseball data analysis application built with LangChain, LangGraph, and LLM models (OpenAI, Ollama, IBM WatsonX, etc). It allows you to query and analyze baseball datasets (player, team, and game statistics) using intelligent agents and customizable workflows.

## Project Structure

```
langraph_baseball/
│   requirements.txt
│   __init__.py
│   README.md
├── controller/
│   └── controller.py
├── model/
│   └── state.py
├── service/
│   ├── agents.py
│   └── nodes.py
└── view/
    └── view.py
```

- **controller/controller.py**: High-level controller to orchestrate questions and answers.
- **model/state.py**: Defines the structure of the state that flows between nodes.
- **service/agents.py**: Defines functions to create pandas agents for each dataset.
- **service/nodes.py**: Defines the workflow nodes, each querying a specific agent.
- **view/view.py**: User interface (CLI or API) to ask questions and display answers.

## Datasets Used
- Player batting statistics
- Player pitching statistics
- Team statistics
- Results of regular season, playoff, and World Series games

All datasets are loaded from public URLs in CSV format.

## Installation

1. **Clone the repository**
2. **Create a virtual environment with Python 3.11+**
3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

1. Configure your preferred LLM model (OpenAI, Ollama, IBM, etc) in `view/view.py`.

2. Run the main view:
   ```powershell
   python view/view.py
   ```

3. Ask questions about baseball statistics, for example:
   - "Which player has the highest batting average?"
   - "How many games did the Yankees win in 2022?"

## Architecture & Flow

- The user asks a question.
- The workflow routes the question to the appropriate node/agent based on the query type.
- The agent executes code on the corresponding DataFrame and returns the answer.
- The result is displayed to the user.

## Customization
- You can easily change the LLM model.
- You can add new nodes/agents for other datasets.
- The workflow is extensible and decoupled.

## Requirements
- Python 3.11+
- Internet access to download the datasets

## Credits
- Built with LangChain, LangGraph, and pandas.
- Course: https://cognitiveclass.ai/courses/build-a-baseball-data-analysis-agent-w-langgraph

