import pytest
from unittest.mock import patch, Mock
import requests

# Add the project root to the path to allow imports from the app
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from web_search_service import search
import config

@pytest.fixture
def mock_requests_post():
    with patch('requests.post') as mock_post:
        yield mock_post

def test_search_calls_parallel_ai_api_with_correct_payload(mock_requests_post):
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": [{"title": "Test Result"}]}
    mock_requests_post.return_value = mock_response
    
    query = "test query"
    
    # Act
    search(query)
    
    # Assert
    mock_requests_post.assert_called_once()
    args, kwargs = mock_requests_post.call_args
    
    assert args[0] == "https://api.parallel.ai/v1/search"
    assert "headers" in kwargs
    assert "json" in kwargs
    
    assert kwargs["headers"]["Authorization"] == f"Bearer {config.PARALLEL_AI_API_KEY}"
    assert kwargs["json"]["query"] == query

def test_search_returns_results_on_success(mock_requests_post):
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 200
    expected_results = [{"title": "Result 1"}, {"title": "Result 2"}]
    mock_response.json.return_value = {"results": expected_results}
    mock_requests_post.return_value = mock_response
    
    # Act
    results = search("any query")
    
    # Assert
    assert results == expected_results

def test_search_raises_http_error_on_failure(mock_requests_post):
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
    mock_requests_post.return_value = mock_response
    
    # Act & Assert
    with pytest.raises(requests.exceptions.HTTPError):
        search("any query")

def test_search_returns_empty_list_if_no_results_key(mock_requests_post):
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 200
    # Simulate a response without the 'results' key
    mock_response.json.return_value = {"data": "some other format"}
    mock_requests_post.return_value = mock_response
    
    # Act
    results = search("any query")
    
    # Assert
    assert results == []
