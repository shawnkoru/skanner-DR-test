import typer
from rich.console import Console
from datetime import datetime
import llm_service
import logger_service
import cache_service
import json
from agents.social_agent import SocialAgent
from agents.tech_agent import TechAgent
from agents.economic_agent import EconomicAgent
from agents.environmental_agent import EnvironmentalAgent
from agents.political_agent import PoliticalAgent
from agents.values_agent import ValuesAgent
import scenario_service
from collections import OrderedDict

console = Console()
app = typer.Typer()

from pathlib import Path

@app.command()
def horizon_scan(
    topic: str = typer.Option(..., "--topic", help="The topic to research (still required when using --dr-file)."),
    output_dir: Path = typer.Option(".", "--output-dir", help="The directory to save output files.", file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    poll_interval: int = typer.Option(8, "--poll-interval", min=1, help="Polling interval (seconds) for Deep Research status."),
    max_cycles: int = typer.Option(120, "--max-cycles", min=1, help="Maximum poll cycles before timing out."),
    debug_dr: bool = typer.Option(False, "--debug-dr", help="Enable Deep Research debug dump."),
    cache_dir: Path = typer.Option(None, "--cache-dir", help="Directory for caching per-topic research (reuse to avoid re-calling LLM).", file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass reading/writing cache even if cache-dir provided."),
    refresh_cache: bool = typer.Option(False, "--refresh-cache", help="Force refresh ignoring existing cached artifacts but overwrite after run."),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level (DEBUG, INFO, WARNING, ERROR)."),
    log_json: bool = typer.Option(False, "--log-json", help="Emit logs in JSON format to stderr."),
    no_scenario_scoring: bool = typer.Option(False, "--no-scenario-scoring", help="Skip scenario extraction and scoring phase."),
    dr_file: Path = typer.Option(None, "--dr-file", exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True, help="Use an existing deep research markdown file instead of generating new content."),
    skip_web_search: bool = typer.Option(False, "--skip-web-search", help="Skip external web search (useful offline or without API key)."),
):
    """
    Run a horizon scan for a given topic.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Initialize logging
    logger_service.init_logger(log_json=log_json, log_level=log_level)
    logger_service.log_event("start", topic=topic, message=f"Starting deep research for topic: {topic}")
    console.print(f"🚀 Starting deep research for topic: [bold cyan]{topic}[/bold cyan]")
    # Apply runtime overrides for LLM polling behavior
    llm_service.configure_timings(poll_interval=poll_interval, max_cycles=max_cycles, debug=debug_dr)

    dr_text = None
    parsed_research = None

    if dr_file is not None:
        # Bypass cache logic; user explicitly supplied a file
        try:
            dr_text = dr_file.read_text(encoding="utf-8")
            console.print(f"📄 Using existing deep research file: [bold]{dr_file}[/bold]")
            logger_service.log_event("deep_research_loaded_file", topic=topic, file=str(dr_file))
        except Exception as e:
            console.print(f"[red]Failed to read provided deep research file: {e}[/red]")
            raise typer.Exit(code=1)
    else:
        if cache_dir and not no_cache:
            dr_cached, parsed_cached = cache_service.load(cache_dir, topic)
            if dr_cached and parsed_cached and not refresh_cache:
                console.print("♻️  Using cached deep research and parsed JSON.")
                logger_service.log_event("cache_hit", topic=topic, parsed=True)
                dr_text, parsed_research = dr_cached, parsed_cached
            elif dr_cached and not refresh_cache:
                console.print("♻️  Using cached deep research; re-parsing JSON.")
                logger_service.log_event("cache_partial_hit", topic=topic, parsed=False)
                dr_text = dr_cached
            else:
                console.print("🔄 Cache miss or refresh requested; generating new deep research.")
                logger_service.log_event("cache_miss", topic=topic, refresh=refresh_cache)

    if dr_text is None:
        with console.status("[bold green]Generating deep research...[/bold green]"):
            dr_text = llm_service.generate_deep_research(topic)
        logger_service.log_event("deep_research_complete", topic=topic, length=len(dr_text))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    dr_filename = results_dir / f"dr_{timestamp}.md"

    if dr_file is None:  # Only write if we just generated it
        with open(dr_filename, "w") as f:
            f.write(dr_text)
        console.print(f"✅ Deep Research complete. Saved to [bold]{dr_filename}[/bold]")
    else:
        # Reuse provided file path for downstream messaging / leave timestamped artifacts separate
        dr_filename = dr_file

    if parsed_research is None:
        parsed_research = llm_service.parse_research(dr_text)
        logger_service.log_event("parse_complete", topic=topic, topics=len(parsed_research.get("topics", [])))
        console.print("✅ Research parsed successfully.")
    else:
        console.print("✅ Parsed research loaded from cache.")
        logger_service.log_event("parse_loaded_from_cache", topic=topic)

    if dr_file is None and cache_dir and not no_cache:
        cache_service.save(cache_dir, topic, dr_text, parsed_research)
        logger_service.log_event("cache_saved", topic=topic)

    # Persist parsed research as its own artifact for reuse/debugging
    parsed_filename = results_dir / f"parsed_research_{timestamp}.json"
    try:
        with open(parsed_filename, "w") as pf:
            json.dump(parsed_research, pf, indent=2)
        console.print(f"💾 Parsed research saved to [bold]{parsed_filename}[/bold]")
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to save parsed research JSON: {e}[/yellow]")
    
    agents = [
        SocialAgent(parsed_research.get("topics", [])),
        TechAgent(parsed_research.get("topics", [])),
        EconomicAgent(parsed_research.get("topics", [])),
        EnvironmentalAgent(parsed_research.get("topics", [])),
        PoliticalAgent(parsed_research.get("topics", [])),
        ValuesAgent(parsed_research.get("topics", []))
    ]

    all_signals = {}
    with console.status("[bold green]Agents are scanning...[/bold green]") as status:
        for agent in agents:
            agent_name = agent.__class__.__name__.replace("Agent", "")
            status.update(f"[bold green]Agent {agent_name} is generating domain map...[/bold green]")
            logger_service.log_event("agent_domain_map", agent=agent_name)
            agent.generate_domain_map()
            status.update(f"[bold green]Agent {agent_name} is scanning for signals...[/bold green]")
            logger_service.log_event("agent_scanning", agent=agent_name)
            if skip_web_search:
                signals = []
                logger_service.log_event("agent_skipped_search", agent=agent_name)
            else:
                signals = agent.scan_for_signals()
            all_signals[agent_name] = signals
            logger_service.log_event("agent_complete", agent=agent_name, signals=len(signals))

    scenario_scores = []
    if not no_scenario_scoring:
        scenarios = scenario_service.extract_scenarios(dr_text)
        if scenarios:
            logger_service.log_event("scenarios_extracted", count=len(scenarios))
            scenario_scores = scenario_service.score_scenarios(scenarios)
            logger_service.log_event("scenarios_scored", count=len(scenario_scores))
        else:
            logger_service.log_event("scenarios_none")

    # Enforce STEEPV ordering in output
    steepv_order = ["Social", "Tech", "Economic", "Environmental", "Political", "Values"]
    ordered_signals = OrderedDict((k, all_signals.get(k, [])) for k in steepv_order)
    report = {
        "topic": topic,
        "timestamp": timestamp,
        "summary": {
            "total_signals": sum(len(v) for v in ordered_signals.values()),
            "domains_with_signals": sum(1 for v in ordered_signals.values() if v)
        },
        "signals": ordered_signals,
        "scenario_scores": scenario_scores
    }
    report_filename = results_dir / f"horizon_scan_results_{timestamp}.json"
    with open(report_filename, "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"✅ Horizon scan complete. Report saved to [bold]{report_filename}[/bold]")
    logger_service.log_event("scan_complete", topic=topic, report=str(report_filename), scenarios=len(scenario_scores))



if __name__ == "__main__":
    app()
