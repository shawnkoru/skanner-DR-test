# Project Horizon

Project Horizon is a command-line interface (CLI) tool for conducting horizon scans on a given topic. It uses a multi-agent system powered by Large Language Models (LLMs) to perform deep research, identify signals of change, and generate a comprehensive report based on the STEEPV (Social, Technological, Economic, Environmental, Political, and Values) framework.

## Features

-   **Deep Research**: Kicks off the process with an in-depth research phase on the user-specified topic.
-   **STEEPV Agent System**: Deploys six specialized agents, one for each STEEPV category, to analyze the research from different perspectives.
-   **Signal Detection**: Each agent scans for emerging signals, trends, and drivers of change within its domain.
-   **Automated Reporting**: Generates two output files:
    -   A Markdown file (`.md`) containing the initial deep research.
    -   A JSON file (`.json`) containing the structured results of the horizon scan, categorized by STEEPV domain.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd horizon-cli
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API keys:**
    Create a `.env` file in the `horizon-cli` directory and add your API keys. The application requires keys for an OpenAI-compatible LLM and the Parallel.ai search service.

    ```
    OPENAI_API_KEY="your-openai-api-key"
    OPENAI_MODEL="o4-mini-deep-research"
    PARALLEL_AI_API_KEY="your-parallel-ai-api-key"
    ```

## Usage

To run a horizon scan, use the `main.py` script with the `--topic` option.

```bash
python main.py --topic "The Future of Artificial Intelligence"
```

### Options

-   `--topic TEXT`: **(Required)** The topic you want to research.
-   `--output-dir DIRECTORY`: The directory where the output files will be saved. Defaults to the current directory.

### Example

```bash
python main.py --topic "Quantum Computing" --output-dir ./scan_results
```

This command will:
1.  Perform a deep research scan on "Quantum Computing".
2.  Create a directory named `scan_results` if it doesn't exist.
3.  Save `dr_YYYY-MM-DD_HHMMSS.md` with the research text in `scan_results`.
4.  Save `horizon_scan_results_YYYY-MM-DD_HHMMSS.json` with the agent findings in `scan_results`.
