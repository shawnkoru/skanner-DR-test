# Architecture Specification: Project Horizon CLI

**Version:** 1.0
**Last Updated:** 2025-09-05

## 1. Overview

Project Horizon CLI is a terminal-based application written in Python that automates the strategic foresight process of horizon scanning. It uses a multi-agent LLM architecture to perform deep research, map a topic domain, and scan the web for weak signals of change. The system is designed to be modular and orchestrated from a central command-line interface.

---

## 2. System Architecture & Data Flow

The architecture follows a sequential, multi-stage pipeline orchestrated by the main CLI function.

```plaintext
[User] -> CLI (Typer)
  |
  `-> 1. Orchestrator: Initiates scan with user topic.
  |
  `-> 2. LLM Service: Generates Deep Research (DR).
  |      |
  |      `-> [File System]: Saves `dr_{timestamp}.md`
  |
  `-> 3. LLM Service: Parses DR text into structured entities (JSON).
  |
  `-> 4. Orchestrator: Dispatches entities to 6 STEEPV Agents.
  |
  `-> 5. STEEPV Agents (Parallel Execution):
  |      |
  |      `-> a. LLM Service: Generate Domain Map (Core, Adjacent, Peripheral).
  |      |
  |      `-> b. Web Search Service: Fetch search results for topics.
  |      |
  |      `-> c. LLM Service: Summarize & evaluate relevance of results.
  |      |
  |      `-> d. Return list of Signal objects.
  |
  `-> 6. Orchestrator: Aggregates signals from all agents.
  |
  `-> 7. [File System]: Saves `scan_results_{timestamp}.json`
  |
[User] <- Final confirmation message.
```

---

## 3. Core Components

### 3.1. CLI (`main.py`)
* **Framework**: Python `Typer`.
* **Responsibilities**:
    * Parses user commands and arguments (`--topic`).
    * Orchestrates the entire workflow from start to finish.
    * Manages user-facing I/O, including progress indicators (`rich`) and final messages.
    * Handles file system operations for saving output (`.md` and `.json`).

### 3.2. LLM Service (`llm_service.py`)
* **Description**: A stateless wrapper for all interactions with the chosen Large Language Model API.
* **Responsibilities**:
    * Manages API key and client instantiation.
    * Contains distinct functions for different tasks: `generate_deep_research`, `parse_research`, `generate_domain_map`, `evaluate_signal_relevance`.
    * Formats prompts and ensures the LLM returns structured data (JSON) where required.
    * Implements error handling and retry logic for API calls.

### 3.3. STEEPV Agents (`agents/`)
* **Structure**: A base class `STEEPV_Agent` (`base_agent.py`) defines the common interface. Six subclasses (`social_agent.py`, `tech_agent.py`, etc.) implement the specific context.
* **State**: Each agent instance holds its category (e.g., "Social"), its initial topics, and its generated domain map.
* **Core Methods**:
    * `generate_domain_map()`: Calls the LLM Service to create its Core/Adjacent/Peripheral topic map.
    * `scan_for_signals()`: Uses its domain map to query the `WebSearchService` and orchestrate the analysis of results.

### 3.4. Web Search Service (`web_search_service.py`)
* **Description**: A stateless wrapper for the chosen web search API (e.g., Serper, Tavily).
* **Responsibilities**:
    * Manages the API key and search parameters.
    * Provides a simple function like `search(query: str)` that returns a clean list of URLs and snippets.

---

## 4. Data Models

Key data structures passed between components.

**Extracted Entities (Parser -> Dispatcher):**
```json
{
  "topics": ["AI in hiring", "gig economy platforms"],
  "concepts": ["algorithmic bias", "digital nomadism"],
  "entities": ["OpenAI", "Upwork"]
}
```

**Domain Map (Agent's internal state):**
```json
{
  "research_area": "Future of Work",
  "steevp_category": "Technological",
  "topics": {
    "Core": {"Technological": ["AI in hiring"]},
    "Adjacent": {"Technological": ["VR meeting platforms"]},
    "Peripheral": {"Technological": ["personal data sovereignty"]}
  }
}
```

**Signal (Agent's output):**
```json
{
  "title": "Title of Article/Source",
  "description": "LLM-generated summary of the content.",
  "relevance": "LLM-generated explanation of its connection to the research area.",
  "sourceURL": "[https://example.com/source](https://example.com/source)"
}
```

---

## 5. Technology Stack

* **Language**: Python 3.10+
* **CLI Framework**: Typer
* **Terminal UI**: Rich
* **Configuration**: python-dotenv
* **LLM API Client**: To be determined (e.g., `google-generativeai`)
* **Web Search API Client**: To be determined (e.g., `requests` for a REST API)

---

## 6. Configuration

* All API keys and configurable settings (e.g., LLM model name) must be stored in a `.env` file at the project root.
* A `config.py` module will load these variables into the application scope using `dotenv.load_dotenv()`.
