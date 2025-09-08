import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

# Add the project root to the path to allow imports from the app
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

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
    
    assert len(md_files) == 1
    assert md_files[0].name.startswith("dr_")
    
    assert len(json_files) == 1
    assert json_files[0].name.startswith("horizon_scan_results_")

def test_horizon_scan_requires_topic():
    # Act
    result = runner.invoke(app) # No --topic provided
    
    # Assert
    assert result.exit_code != 0
    assert "Missing option '--topic'" in result.stdout
