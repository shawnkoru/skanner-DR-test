import pytest
from unittest.mock import patch, MagicMock
import json

# Add project root
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import llm_service
from llm_service import generate_deep_research, parse_research, generate_domain_map, _heuristic_topics, _call_responses_api


# ---- Helpers to mock OpenAI Responses objects ---- #
class _MockPart:
    def __init__(self, text: str, p_type: str = "output_text"):
        self.type = p_type
        self.text = text

class _MockMessageItem:
    def __init__(self, text: str):
        self.type = 'message'
        self.content = [_MockPart(text)]

class _MockStatusResponse:
    def __init__(self, status: str, text: str = ""):
        self.status = status
        self.output = [] if not text else [_MockMessageItem(text)]
        self.id = "resp_123"
    def model_dump_json(self):
        return json.dumps({"status": self.status, "output_len": len(self.output)})


@pytest.fixture
def mock_openai_responses():
    with patch('llm_service.client.responses') as mock_resp:
        yield mock_resp


def test_generate_deep_research_polls_until_complete(mock_openai_responses):
    # Arrange: create() returns object with id; retrieve() returns in_progress then completed
    create_obj = MagicMock()
    create_obj.id = "resp_123"
    mock_openai_responses.create.return_value = create_obj
    mock_openai_responses.retrieve.side_effect = [
        _MockStatusResponse('in_progress'),
        _MockStatusResponse('completed', text="Final deep research body")
    ]
    with patch('llm_service.time.sleep') as mock_sleep:  # avoid real wait
        # Act
        result = generate_deep_research("Quantum Cats")
    # Assert
    assert "Final deep research body" in result
    assert mock_openai_responses.retrieve.call_count == 2
    mock_sleep.assert_called()  # we did at least one poll wait


def test_parse_research_returns_structured_json_when_available():
    # Patch _call_responses_api to return direct JSON
    payload = {"topics": ["A"], "entities": [], "concepts": []}
    with patch('llm_service._call_responses_api', return_value=json.dumps(payload)) as mocked:
        result = parse_research("Some DR text")
    mocked.assert_called_once()
    assert result == payload


def test_parse_research_heuristic_topics_when_empty():
    # Simulate failure to parse (returns non-JSON) so heuristics trigger
    deep_text = """# Executive Summary\n\n## Synthetic Bio Interfaces\n## Planetary Scale Sensing\nIntroduction text here\n## Edge Quantum Accelerators\nConclusion text\n"""
    with patch('llm_service._call_responses_api', return_value="NOT JSON"):
        result = parse_research(deep_text)
    assert 'topics' in result
    # Heuristic should capture headings excluding intro/conclusion words
    assert any("Synthetic Bio Interfaces" in t for t in result['topics'])
    assert any("Planetary Scale Sensing" in t for t in result['topics'])


def test__heuristic_topics_deduplicates_and_limits():
    text = """# Alpha\n# Alpha\n## Beta\nRandom line\n## Gamma\n## Delta\n"""
    topics = _heuristic_topics(text, max_topics=3)
    assert len(topics) == 3
    assert topics[0] == 'Alpha'
    assert 'Beta' in topics


def test_generate_domain_map_fallback_distribution():
    # Force raw LLM response to be empty so fallback kicks in assigning topics to each band
    topics = ["Topic One", "Topic Two", "Topic Three"]
    with patch('llm_service._call_responses_api', return_value="{}"):
        domain_map = generate_domain_map(topics, "Social")
    assert 'topics' in domain_map
    for band in ["Core", "Adjacent", "Peripheral"]:
        assert domain_map['topics'][band]['Social'], f"Band {band} missing fallback topic"


def test__call_responses_api_failed_status(monkeypatch):
    # Simulate a create followed by a failed retrieval.
    mock_responses = MagicMock()
    create_obj = MagicMock(); create_obj.id = "resp_fail"
    mock_responses.create.return_value = create_obj
    class FailResp:
        status = 'failed'
        output = []
        id = 'resp_fail'
        error = {'message': 'boom'}
        def model_dump_json(self):
            return json.dumps({'status': 'failed'})
    mock_responses.retrieve.return_value = FailResp()
    monkeypatch.setattr(llm_service.client, 'responses', mock_responses)
    out = _call_responses_api("sys msg", "user query")
    assert out.startswith("Error: Task failed")


def test__call_responses_api_timeout(monkeypatch):
    # Force timeout by always returning in_progress beyond MAX_POLL_CYCLES
    mock_responses = MagicMock()
    create_obj = MagicMock(); create_obj.id = "resp_timeout"
    mock_responses.create.return_value = create_obj
    class InProgress:
        status = 'in_progress'
        output = []
        id = 'resp_timeout'
        def model_dump_json(self):
            return json.dumps({'status': 'in_progress'})
    mock_responses.retrieve.return_value = InProgress()
    monkeypatch.setattr(llm_service.client, 'responses', mock_responses)
    monkeypatch.setattr(llm_service, 'MAX_POLL_CYCLES', 2)
    # Avoid real sleep
    monkeypatch.setattr(llm_service.time, 'sleep', lambda *_: None)
    out = _call_responses_api("sys msg", "user query")
    assert "timed out" in out.lower()
