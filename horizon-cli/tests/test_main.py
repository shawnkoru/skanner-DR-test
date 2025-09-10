import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

# Add the project root to the path to allow imports from the app
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app
import cache_service

runner = CliRunner()

# Since we are testing the orchestrator, we mock all the components it orchestrates.
@pytest.fixture
def mock_llm_service():
    with patch('main.llm_service') as mock_llm:
        mock_llm.generate_deep_research.return_value = "Deep research text"
        mock_llm.parse_research.return_value = {"topics": ["parsed topic"]}
        yield mock_llm

@pytest.fixture
def mock_agents():
    # Mock all 6 agent classes
    with patch('main.SocialAgent') as mock_social, \
         patch('main.TechAgent') as mock_tech, \
         patch('main.EconomicAgent') as mock_economic, \
         patch('main.EnvironmentalAgent') as mock_environmental, \
         patch('main.PoliticalAgent') as mock_political, \
         patch('main.ValuesAgent') as mock_values:
        
        # Make all agent instances behave the same way
        mock_agent_instance = MagicMock()
        mock_agent_instance.generate_domain_map.return_value = None
        mock_agent_instance.scan_for_signals.return_value = [{"title": "A Signal"}]

        mock_social.return_value = mock_agent_instance
        mock_tech.return_value = mock_agent_instance
        mock_economic.return_value = mock_agent_instance
        mock_environmental.return_value = mock_agent_instance
        mock_political.return_value = mock_agent_instance
        mock_values.return_value = mock_agent_instance
        
        yield {
            "Social": mock_social,
            "Tech": mock_tech,
            "Economic": mock_economic,
            "Environmental": mock_environmental,
            "Political": mock_political,
            "Values": mock_values,
            "instance": mock_agent_instance
        }

def test_horizon_scan_end_to_end_flow(mock_llm_service, mock_agents, tmp_path):
    # Arrange
    topic = "test-topic"
    output_dir = tmp_path / "test_output"

    # Act
    result = runner.invoke(app, ["--topic", topic, "--output-dir", str(output_dir)])

    # Assert
    # Check for successful execution
    assert result.exit_code == 0
    assert f"🚀 Starting deep research for topic: {topic}" in result.stdout
    assert "✅ Deep Research complete." in result.stdout
    assert "✅ Research parsed successfully." in result.stdout
    assert "✅ Horizon scan complete." in result.stdout

    # Verify service calls
    mock_llm_service.generate_deep_research.assert_called_once_with(topic)
    mock_llm_service.parse_research.assert_called_once_with("Deep research text")

    # Verify agent instantiation and method calls
    mock_agents["Social"].assert_called_once_with(["parsed topic"])
    mock_agents["Tech"].assert_called_once_with(["parsed topic"])
    # ... and so on for all 6 agents

    # The mock instance is shared, so we expect 6 calls total for each method
    assert mock_agents["instance"].generate_domain_map.call_count == 6
    assert mock_agents["instance"].scan_for_signals.call_count == 6

    # Verify that output files were created in the specified directory
    assert output_dir.is_dir()
    files = list(output_dir.iterdir())
    md_files = [f for f in files if f.suffix == '.md']
    json_files = [f for f in files if f.suffix == '.json']

    # Expect two JSON artifacts now: parsed_research_*.json and horizon_scan_results_*.json
    assert len(md_files) == 1 and md_files[0].name.startswith("dr_")
    assert len(json_files) == 2
    parsed_files = [f for f in json_files if f.name.startswith("parsed_research_")]
    report_files = [f for f in json_files if f.name.startswith("horizon_scan_results_")]
    assert len(parsed_files) == 1
    assert len(report_files) == 1

    # Verify parsed JSON content matches mocked return
    import json as _json
    with open(parsed_files[0], 'r') as pf:
        parsed_content = _json.load(pf)
    assert parsed_content == {"topics": ["parsed topic"]}


def test_horizon_scan_cli_timing_flags(mock_llm_service, mock_agents, tmp_path):
    topic = "timing-topic"
    outdir = tmp_path / "timing_output"
    result = runner.invoke(app, ["--topic", topic, "--output-dir", str(outdir), "--poll-interval", "3", "--max-cycles", "5", "--debug-dr"])    
    assert result.exit_code == 0
    # Verify configure_timings invoked with expected args on mocked llm_service
    mock_llm_service.configure_timings.assert_called_once()
    args, kwargs = mock_llm_service.configure_timings.call_args
    # Accept positional or keyword invocation
    if kwargs:
        assert kwargs.get('poll_interval') == 3
        assert kwargs.get('max_cycles') == 5
        assert kwargs.get('debug') is True
    else:
        # Positional ordering: poll_interval, max_cycles, debug
        assert args[0] == 3 and args[1] == 5 and args[2] is True

def test_horizon_scan_requires_topic():
    result = runner.invoke(app)  # No --topic provided
    assert result.exit_code != 0
    assert "Missing option '--topic'" in result.stdout


def test_horizon_scan_uses_cache(tmp_path, mock_llm_service, mock_agents):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    topic = "Cached Topic"
    dr_text = "# DR Cached\nContent"
    parsed = {"topics": ["cached topic"]}
    cache_service.save(cache_dir, topic, dr_text, parsed)

    runner_result = runner.invoke(app, [
        "--topic", topic,
        "--output-dir", str(tmp_path / "out"),
        "--cache-dir", str(cache_dir)
    ])
    assert runner_result.exit_code == 0
    # LLM generation should have been skipped
    mock_llm_service.generate_deep_research.assert_not_called()
    mock_llm_service.parse_research.assert_not_called()


def test_horizon_scan_refresh_cache(tmp_path, mock_llm_service, mock_agents):
    cache_dir = tmp_path / "cache2"
    cache_dir.mkdir()
    topic = "Refresh Topic"
    # Pre-populate cache
    cache_service.save(cache_dir, topic, "old dr", {"topics": ["old"]})
    # Run with refresh; should invoke services
    runner_result = runner.invoke(app, [
        "--topic", topic,
        "--output-dir", str(tmp_path / "out2"),
        "--cache-dir", str(cache_dir),
        "--refresh-cache"
    ])
    assert runner_result.exit_code == 0
    mock_llm_service.generate_deep_research.assert_called_once()
    mock_llm_service.parse_research.assert_called_once()
def test_logging_cli_flags(tmp_path, mock_llm_service, mock_agents, capsys):
    from unittest.mock import patch, MagicMock
    topic = "Log Topic"
    with patch('logger_service.init_logger') as mock_init, \
         patch('logger_service.log_event') as mock_log_event:
        mock_init.return_value = MagicMock()
        result = runner.invoke(app, [
            "--topic", topic,
            "--output-dir", str(tmp_path / "out"),
            "--log-json",
            "--log-level", "DEBUG"
        ])
    assert result.exit_code == 0
    # Ensure logger initialized with json flag
    mock_init.assert_called_once()
    # Extract event names from calls
    events = [c.kwargs.get('event') or (c.args[0] if c.args else None) for c in mock_log_event.call_args_list]
    assert 'start' in events
    assert 'scan_complete' in events
    # Ensure services were used despite cache presence (refresh)
