import pytest
from unittest.mock import patch, Mock
import requests
import json
import uuid

# Add the project root to the path to allow imports from the app
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import llm_service
from llm_service import _call_responses_api, generate_deep_research, parse_research, generate_domain_map

# Fixtures
@pytest.fixture
def mock_requests_post():
    with patch('requests.post') as mock_post:
        yield mock_post

def test_generate_deep_research_calls_api(mock_requests_post):
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"text": "Deep research content"}
    mock_requests_post.return_value = mock_response
    
    # Act
    result = generate_deep_research("test topic")
    
    # Assert
    mock_requests_post.assert_called_once()
    assert result == "Deep research content"

def test_parse_research_calls_api_and_parses_json(mock_requests_post):
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 200
    expected_dict = {"topics": ["a", "b"]}
    mock_response.json.return_value = {"text": json.dumps(expected_dict)}
    mock_requests_post.return_value = mock_response
    
    # Act
    result = parse_research("some text")
    
    # Assert
    mock_requests_post.assert_called_once()
    assert result == expected_dict

def test_generate_domain_map_calls_api_and_parses_json(mock_requests_post):
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 200
    expected_dict = {"Core": ["topic1"]}
    # The API returns a string, which we then parse. So the mock should return a string.
    mock_response.json.return_value = {"text": json.dumps(expected_dict)}
    mock_requests_post.return_value = mock_response
    
    # Act
    result = generate_domain_map(["topic1"], "Social")
    
    # Assert
    mock_requests_post.assert_called_once()
    assert result == expected_dict

def test_call_responses_api_constructs_correct_payload(mock_requests_post):
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"text": "Success"}
    mock_requests_post.return_value = mock_response
    
    prompt_text = "My test prompt"
    
    # Act
    _call_responses_api(prompt_text)
    
    # Assert
    mock_requests_post.assert_called_once()
    args, kwargs = mock_requests_post.call_args
    
    assert args[0] == "https://api.openai.com/v1/responses"
    assert "json" in kwargs
    
    payload = kwargs["json"]
    assert payload["model"] == llm_service.config.OPENAI_MODEL
    assert "prompt" in payload
    assert payload["prompt"]["text"] == prompt_text
    assert "id" in payload["prompt"]
    # Check if the id is a valid UUID
    try:
        uuid.UUID(payload["prompt"]["id"])
    except ValueError:
        pytest.fail("The generated ID is not a valid UUID.")

def test_call_responses_api_handles_http_error(mock_requests_post):
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = '{"error": "Bad Request"}'
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Client Error")
    mock_requests_post.return_value = mock_response
    
    # Act & Assert
    with pytest.raises(requests.exceptions.HTTPError):
        _call_responses_api("test prompt")

def test_call_responses_api_handles_unexpected_json_structure(mock_requests_post):
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 200
    unexpected_payload = {"data": "some other structure"}
    mock_response.json.return_value = unexpected_payload
    # Also set the .text attribute on the mock response
    mock_response.text = json.dumps(unexpected_payload)
    mock_requests_post.return_value = mock_response
    
    # Act
    result = _call_responses_api("test prompt")
    
    # Assert
    assert result == json.dumps(unexpected_payload)

def test_call_responses_api_handles_choices_structure(mock_requests_post):
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 200
    choices_payload = {"choices": [{"text": "content from choices"}]}
    mock_response.json.return_value = choices_payload
    mock_requests_post.return_value = mock_response
    
    # Act
    result = _call_responses_api("test prompt")
    
    # Assert
    assert result == "content from choices"
