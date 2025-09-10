import sys, os, json
from pathlib import Path
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import main as horizon_main


@pytest.fixture
def temp_output(tmp_path: Path):
    return tmp_path


def test_horizon_scan_creates_results_dir_and_order(temp_output: Path):
    # Prepare a dummy deep research file
    dr_file = temp_output / "existing_dr.md"
    dr_file.write_text("# Dummy Deep Research\n\nContent", encoding="utf-8")

    topics_list = [
        "AI and machine learning in animal communication",
        "Bioacoustics and sensing technologies"
    ]

    # Mock parsing + domain map generation to avoid API / LLM calls
    fake_parsed = {"topics": topics_list}
    fake_domain_map = {
        "topics": {
            "Peripheral": {"Social": ["social peripheral"], "Tech": [], "Economic": [], "Environmental": [], "Political": [], "Values": []},
            "Adjacent": {"Social": [], "Tech": [], "Economic": [], "Environmental": [], "Political": [], "Values": []},
        }
    }

    with patch("main.llm_service.parse_research", return_value=fake_parsed), \
         patch("main.llm_service.generate_domain_map", return_value=fake_domain_map):
        # Run with skip_web_search to avoid network
        horizon_main.horizon_scan(
            topic="Test Topic",
            output_dir=temp_output,
            dr_file=dr_file,
            skip_web_search=True,
            no_scenario_scoring=True,
            log_level="ERROR"
        )

    results_dir = temp_output / "results"
    assert results_dir.exists(), "results directory not created"
    reports = list(results_dir.glob("horizon_scan_results_*.json"))
    assert reports, "No horizon scan results file written"
    data = json.loads(reports[0].read_text(encoding="utf-8"))
    expected_order = ["Social", "Tech", "Economic", "Environmental", "Political", "Values"]
    assert list(data["signals"].keys()) == expected_order
    # Ensure summary present
    assert "summary" in data and "total_signals" in data["summary"]