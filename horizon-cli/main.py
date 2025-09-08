import typer
from rich.console import Console
from datetime import datetime
import llm_service
import json
from agents.social_agent import SocialAgent
from agents.tech_agent import TechAgent
from agents.economic_agent import EconomicAgent
from agents.environmental_agent import EnvironmentalAgent
from agents.political_agent import PoliticalAgent
from agents.values_agent import ValuesAgent

console = Console()
app = typer.Typer()

from pathlib import Path

@app.command()
def horizon_scan(
    topic: str = typer.Option(..., "--topic", help="The topic to research."),
    output_dir: Path = typer.Option(".", "--output-dir", help="The directory to save output files.", file_okay=False, dir_okay=True, writable=True, resolve_path=True)
):
    """
    Run a horizon scan for a given topic.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"🚀 Starting deep research for topic: [bold cyan]{topic}[/bold cyan]")

    with console.status("[bold green]Generating deep research...[/bold green]") as status:
        dr_text = llm_service.generate_deep_research(topic)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dr_filename = output_dir / f"dr_{timestamp}.md"

    with open(dr_filename, "w") as f:
        f.write(dr_text)

    console.print(f"✅ Deep Research complete. Saved to [bold]{dr_filename}[/bold]")

    parsed_research = llm_service.parse_research(dr_text)
    console.print("✅ Research parsed successfully.")
    
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
            agent.generate_domain_map()
            status.update(f"[bold green]Agent {agent_name} is scanning for signals...[/bold green]")
            signals = agent.scan_for_signals()
            all_signals[agent_name] = signals

    report_filename = output_dir / f"horizon_scan_results_{timestamp}.json"
    with open(report_filename, "w") as f:
        json.dump(all_signals, f, indent=2)

    console.print(f"✅ Horizon scan complete. Report saved to [bold]{report_filename}[/bold]")


if __name__ == "__main__":
    app()
