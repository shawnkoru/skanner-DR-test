import pytest
from unittest.mock import patch, Mock

# Add the project root to the path to allow imports from the app
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.base_agent import STEEPV_Agent

# A concrete implementation for testing purposes
class AgentUnderTest(STEEPV_Agent):
    """Concrete test agent whose category resolves to 'UnderTest'."""
    pass

@pytest.fixture
def mock_llm_service():
    with patch('agents.base_agent.llm_service') as mock_llm:
        yield mock_llm

@pytest.fixture
def mock_web_search_service():
    with patch('agents.base_agent.web_search_service') as mock_web:
        yield mock_web

def test_generate_domain_map_calls_llm_service(mock_llm_service):
    # Arrange
    agent = AgentUnderTest(topics=["initial topic"])
    expected_map = {"topics": {"Core": {"UnderTest": []}, "Adjacent": {"UnderTest": []}, "Peripheral": {"UnderTest": []}}}
    mock_llm_service.generate_domain_map.return_value = expected_map
    
    # Act
    agent.generate_domain_map()
    
    # Assert
    mock_llm_service.generate_domain_map.assert_called_once_with(["initial topic"], "UnderTest")
    assert agent.domain_map == expected_map

def test_generate_domain_map_does_not_call_if_no_topics(mock_llm_service):
    # Arrange
    agent = AgentUnderTest(topics=[])
    
    # Act
    agent.generate_domain_map()
    
    # Assert
    mock_llm_service.generate_domain_map.assert_not_called()

def test_scan_for_signals_calls_web_search(mock_web_search_service):
    # Arrange
    agent = AgentUnderTest(topics=["..."])
    agent.domain_map = {
        "topics": {
            "Peripheral": {"UnderTest": ["peripheral topic"]},
            "Adjacent": {"UnderTest": ["adjacent topic"]}
        }
    }
    mock_web_search_service.search.return_value = [
        {"title": "Signal 1", "snippet": "Desc 1", "link": "url1"}
    ]
    
    # Act
    signals = agent.scan_for_signals()
    
    # Assert
    assert mock_web_search_service.search.call_count == 2
    mock_web_search_service.search.assert_any_call("peripheral topic")
    mock_web_search_service.search.assert_any_call("adjacent topic")
    assert len(signals) == 2
    assert signals[0]["title"] == "Signal 1"

def test_scan_for_signals_handles_no_domain_map(mock_web_search_service):
    # Arrange
    agent = AgentUnderTest(topics=["..."])
    agent.domain_map = None # Explicitly set to None
    
    # Act
    signals = agent.scan_for_signals()
    
    # Assert
    mock_web_search_service.search.assert_not_called()
    assert signals == []

def test_scan_for_signals_handles_malformed_domain_map(mock_web_search_service):
    # Arrange
    agent = AgentUnderTest(topics=["..."])
    # Domain map is missing the 'topics' key
    agent.domain_map = {"some_other_key": {}}
    
    # Act
    signals = agent.scan_for_signals()
    
    # Assert
    mock_web_search_service.search.assert_not_called()
    assert signals == []

def test_scan_for_signals_handles_malformed_search_results(mock_web_search_service):
    # Arrange
    agent = AgentUnderTest(topics=["..."])
    agent.domain_map = {
        "topics": { "Peripheral": {"UnderTest": ["a topic"]}, "Adjacent": {} }
    }
    # Search result is missing expected keys
    mock_web_search_service.search.return_value = [{"unexpected_key": "value"}]
    
    # Act
    signals = agent.scan_for_signals()
    
    # Assert
    assert len(signals) == 1
    assert signals[0]["title"] == "N/A"
    assert signals[0]["description"] == "N/A"
    assert signals[0]["sourceURL"] == "N/A"
