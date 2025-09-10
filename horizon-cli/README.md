# Project Horizon

Project Horizon is a command-line interface (CLI) tool for conducting horizon scans on a given topic. It uses a multi-agent system powered by Large Language Models (LLMs) to perform deep research, identify signals of change, and generate a comprehensive report based on the STEEPV (Social, Technological, Economic, Environmental, Political, and Values) framework.

## Features

-   **Deep Research**: Kicks off the process with an in-depth research phase on the user-specified topic.
-   **STEEPV Agent System**: Deploys six specialized agents, one for each STEEPV category, to analyze the research from different perspectives.
-   **Signal Detection**: Each agent scans for emerging signals, trends, and drivers of change within its domain (optional offline mode).
-   **Automated Reporting**: Generates two output files:
    -   A Markdown file (`.md`) containing the initial deep research.
    -   A JSON file (`.json`) containing the structured results of the horizon scan, categorized by STEEPV domain.

> NOTE: Web search now targets Parallel's `v1beta/search` endpoint (fields: `objective`, `search_queries`, etc.) with a temporary legacy fallback.

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
    Create a `.env` file in the `horizon-cli` directory and add your API keys (never commit real keys to git).

    ```env
    OPENAI_API_KEY="your-openai-api-key"
    OPENAI_MODEL="o4-mini-deep-research"
    PARALLEL_AI_API_KEY="your-parallel-ai-api-key"  # Optional if skipping web search
    ```

## Usage

To run a horizon scan, use the `main.py` script with the `--topic` option.

```bash
python main.py --topic "The Future of Artificial Intelligence"
```

### Key Options

- `--topic TEXT` (required): Topic to research (still required when reusing a file).
- `--dr-file PATH`: Reuse an existing deep research markdown file (skips new LLM generation).
- `--skip-web-search`: Skip external web searching (offline / no Parallel key). Signals list will be empty unless you later enrich.
- `--no-scenario-scoring`: Skip scenario extraction & scoring.
- `--cache-dir DIRECTORY`: Enable caching of deep research + parsed JSON.
- `--refresh-cache`: Force regeneration ignoring cached artifacts.
- `--poll-interval / --max-cycles`: Control LLM polling cadence & timeout.
- `--log-json`: Emit structured JSON logs instead of plain text.

### Examples

Fresh run (full pipeline):
```bash
python main.py --topic "Quantum Computing" --output-dir ./scan_results
```

Reuse an existing deep research file and skip search (offline signal structure only):
```bash
python main.py --topic "AI Animal Communication" --dr-file ./dr_2025-09-09_164653.md --skip-web-search --no-scenario-scoring
```

With caching and JSON logs:
```bash
python main.py --topic "Generative Biology" --cache-dir ./.cache --log-json --poll-interval 5 --max-cycles 90
```

Each run produces (under `results/` inside the specified `--output-dir`):
1. `dr_*.md` deep research markdown (unless `--dr-file` used).
2. `parsed_research_*.json` parsed research structure.
3. `horizon_scan_results_*.json` ordered STEEPV signals + summary (+ scenarios unless skipped).

If `--skip-web-search` is set, agents still build domain maps but return empty signals; you can later inject mock results or run again without the flag to populate signals.
