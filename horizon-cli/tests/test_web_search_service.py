import pytest
from unittest.mock import patch, Mock
import requests

# Add the project root to the path to allow imports from the app
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from web_search_service import search, DEFAULT_MAX_RETRIES, PARALLEL_ENDPOINT_PRIMARY
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
    
    assert args[0] == PARALLEL_ENDPOINT_PRIMARY
    assert "headers" in kwargs
    assert "json" in kwargs
    body = kwargs["json"]
    assert body["objective"] == query
    assert body["search_queries"] == [query]
    assert body["processor"] == "base"
    assert "max_results" in body and "max_chars_per_result" in body
    # Header changed to x-api-key
    assert kwargs["headers"]["x-api-key"] == f"{config.PARALLEL_AI_API_KEY}"

def test_search_returns_results_on_success(mock_requests_post):
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 200
    # New API shape requires normalization
    raw_results = [
        {"title": "Result 1", "url": "https://example.com/1", "excerpts": ["Line a", "Line b"]},
        {"title": "Result 2", "url": "https://example.com/2", "excerpts": ["Line c"]}
    ]
    mock_response.json.return_value = {"results": raw_results}
    mock_requests_post.return_value = mock_response
    
    # Act
    results = search("any query")
    
    # Assert
    assert len(results) == 2
    assert results[0]["title"] == "Result 1"
    assert results[0]["link"] == "https://example.com/1"
    assert "Line a" in results[0]["snippet"]

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


def test_search_retries_and_returns_empty_after_failures(mock_requests_post):
    # Simulate transient connection errors then final failure
    from requests.exceptions import ConnectionError
    mock_requests_post.side_effect = [ConnectionError("boom"), ConnectionError("boom2"), ConnectionError("boom3")]
    results = search("query", max_retries=3)
    assert results == []
    assert mock_requests_post.call_count == 3


def test_search_retries_then_succeeds(mock_requests_post):
    from requests.exceptions import Timeout
    success = Mock()
    success.status_code = 200
    success.json.return_value = {"results": [{"title": "Recovered"}]}
    success.raise_for_status.return_value = None
    mock_requests_post.side_effect = [Timeout("t1"), Timeout("t2"), success]
    results = search("query", max_retries=5)
    assert len(results) == 1 and results[0]["title"] == "Recovered"
    assert mock_requests_post.call_count == 3

def test_search_retries_on_http_5xx_then_succeeds(mock_requests_post):
    # Prepare two 500 HTTPError responses then success
    import requests
    # First failing response
    resp1 = Mock()
    err1 = requests.exceptions.HTTPError("500 Server Error")
    err1.response = Mock()
    err1.response.status_code = 500
    resp1.status_code = 500
    resp1.raise_for_status.side_effect = err1

    # Second failing response
    resp2 = Mock()
    err2 = requests.exceptions.HTTPError("500 Server Error")
    err2.response = Mock()
    err2.response.status_code = 500
    resp2.status_code = 500
    resp2.raise_for_status.side_effect = err2

    # Success response
    success = Mock()
    success.status_code = 200
    success.raise_for_status.return_value = None
    success.json.return_value = {"results": [{"title": "Recovered After 5xx"}]}

    mock_requests_post.side_effect = [resp1, resp2, success]

    results = search("query", max_retries=5)
    assert len(results) == 1 and results[0]["title"] == "Recovered After 5xx"
    assert mock_requests_post.call_count == 3

def test_search_5xx_exhausts_retries_returns_empty(mock_requests_post):
    import requests
    attempts = []
    def make_resp():
        r = Mock()
        err = requests.exceptions.HTTPError("500 Server Error")
        err.response = Mock(); err.response.status_code = 500
        r.status_code = 500
        r.raise_for_status.side_effect = err
        return r
    mock_requests_post.side_effect = [make_resp(), make_resp(), make_resp()]
    results = search("query", max_retries=3)
    assert results == []
    assert mock_requests_post.call_count == 3

def test_search_non_retryable_404_raises(mock_requests_post):
    import requests
    resp = Mock()
    err = requests.exceptions.HTTPError("404 Not Found")
    err.response = Mock(); err.response.status_code = 404
    resp.status_code = 404
    resp.raise_for_status.side_effect = err
    mock_requests_post.return_value = resp
    with pytest.raises(requests.exceptions.HTTPError):
        search("query")

def test_search_immediate_exhaustion_returns_empty(mock_requests_post):
    # Set max_retries=0 to force while condition false immediately and hit final return path
    results = search("query", max_retries=0)
    assert results == []
    mock_requests_post.assert_not_called()
