import config
import requests

def search(query: str) -> list:
    """
    Provides a simple function like search(query: str) that returns a clean list of URLs and snippets.
    """
    headers = {
        "Authorization": f"Bearer {config.PARALLEL_AI_API_KEY}"
    }
    
    response = requests.post(
        "https://api.parallel.ai/v1/search",
        headers=headers,
        json={"query": query}
    )
    
    response.raise_for_status()
    
    # Assuming the response has a 'results' key with a list of search results
    return response.json().get("results", [])
